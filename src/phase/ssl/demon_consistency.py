from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from phase.config import DemonLossConfig

NYQUIST_MARGIN = 0.98


class DemonSpectrum(nn.Module):
    """Batched, differentiable DEMON, matching `phase.features.demon` in structure.

    Bandpass to the cavitation band, take the analytic envelope, low-pass and
    decimate it, then transform the envelope. Filtering is done in the frequency
    domain rather than with `sosfiltfilt` so the whole thing stays on the GPU and
    runs on a batch.
    """

    def __init__(
        self,
        config: DemonLossConfig,
        sample_rate: int,
        max_modulation_hz: float = 100.0,
        n_bins: int = 256,
    ):
        super().__init__()
        self.sample_rate = sample_rate
        self.max_modulation_hz = max_modulation_hz
        self.n_bins = n_bins
        nyquist = sample_rate / 2
        self.band_high = min(config.band_high, nyquist * NYQUIST_MARGIN)
        self.band_low = min(config.band_low, self.band_high * 0.5)
        self.envelope_lowpass = 500.0

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        n = waveform.shape[-1]
        freqs = torch.fft.rfftfreq(n, 1.0 / self.sample_rate, device=waveform.device)

        spectrum = torch.fft.rfft(waveform, dim=-1)
        band = (freqs >= self.band_low) & (freqs <= self.band_high)

        analytic = torch.zeros_like(spectrum)
        analytic[..., band] = spectrum[..., band] * 2.0
        envelope = torch.fft.irfft(analytic, n=n, dim=-1).abs()

        envelope_spectrum = torch.fft.rfft(envelope - envelope.mean(dim=-1, keepdim=True), dim=-1)
        envelope_spectrum = envelope_spectrum * (freqs <= self.envelope_lowpass)

        keep = freqs <= self.max_modulation_hz
        modulation = envelope_spectrum[..., keep].abs()

        # a 30 s window at 32 kHz resolves the modulation axis to 0.03 Hz, far finer
        # than shaft and blade lines are broad; pool to a fixed grid so the head stays
        # small and the target does not change with window length
        if modulation.shape[-1] != self.n_bins:
            modulation = F.adaptive_avg_pool1d(modulation.unsqueeze(1), self.n_bins).squeeze(1)
        return F.normalize(modulation, dim=-1)


class DemonConsistency(nn.Module):
    """Protects propeller modulation from being discarded by the encoder.

    `PLAN.md` describes "an auxiliary loss penalising divergence between the DEMON
    spectrum implied by the two views' embeddings", which admits more than one
    reading. Penalising divergence between the two views' *measured* DEMON spectra
    is close to vacuous: Phase 4 already established that every augmentation
    preserves those peaks, so the two agree by construction and the gradient would
    say nothing about the encoder.

    Implemented instead as a prediction task. A small head reconstructs the true
    DEMON spectrum of a view from that view's embedding, so the loss is only low
    when the embedding still carries the modulation. That cannot be satisfied by a
    constant, and it is the reading that matches the stated intent of protecting
    the cue rather than merely observing it is present.

    Both views are supervised and their predictions are additionally tied to each
    other, which is the consistency half of the name.
    """

    def __init__(
        self, config: DemonLossConfig, embedding_dim: int, sample_rate: int, n_samples: int
    ):
        super().__init__()
        self.config = config
        self.weight = config.weight
        self.spectrum = DemonSpectrum(config, sample_rate)

        with torch.no_grad():
            probe = self.spectrum(torch.zeros(1, n_samples))
        self.n_bins = probe.shape[-1]

        self.head = nn.Sequential(
            nn.Linear(embedding_dim, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, self.n_bins),
        )

    def predict(self, embedding: torch.Tensor) -> torch.Tensor:
        return F.normalize(F.softplus(self.head(embedding)), dim=-1)

    def forward(
        self,
        embedding_q: torch.Tensor,
        embedding_k: torch.Tensor,
        waveform_q: torch.Tensor,
        waveform_k: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        with torch.no_grad():
            target_q = self.spectrum(waveform_q)
            target_k = self.spectrum(waveform_k)

        predicted_q = self.predict(embedding_q)
        predicted_k = self.predict(embedding_k)

        reconstruction = F.mse_loss(predicted_q, target_q) + F.mse_loss(predicted_k, target_k)
        consistency = F.mse_loss(predicted_q, predicted_k)
        loss = reconstruction + consistency

        stats = {
            "demon_reconstruction": float(reconstruction.detach()),
            "demon_consistency": float(consistency.detach()),
            "demon_target_cosine": float(
                F.cosine_similarity(target_q, target_k, dim=-1).mean().detach()
            ),
            "demon_predicted_cosine": float(
                F.cosine_similarity(predicted_q, target_q, dim=-1).mean().detach()
            ),
        }
        return self.weight * loss, stats
