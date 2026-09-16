from __future__ import annotations

import torch
import torch.nn as nn
from torchaudio.transforms import MelSpectrogram

from phase.config import FeatureConfig


class LogMelFrontEnd(nn.Module):
    """Batched log-Mel on the training device.

    Features are computed after augmentation, so this has to live on the GPU
    alongside the encoder rather than in the DataLoader.
    """

    def __init__(self, config: FeatureConfig, sample_rate: int):
        super().__init__()
        if config.name != "logmel":
            raise ValueError(f"the training front end handles logmel, not {config.name!r}")
        params = config.params
        self.log_offset = params.log_offset
        self.normalize = config.normalize
        self.highpass_hz = config.highpass_hz
        self.sample_rate = sample_rate

        self.mel = MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=params.n_fft,
            hop_length=params.hop_length,
            n_mels=params.n_mels,
            f_min=params.fmin,
            f_max=min(params.fmax, sample_rate / 2),
            power=params.power,
        )

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        if self.highpass_hz > 0:
            spectrum = torch.fft.rfft(waveform, dim=-1)
            freqs = torch.fft.rfftfreq(
                waveform.shape[-1], 1.0 / self.sample_rate, device=waveform.device
            )
            spectrum = spectrum * (freqs >= self.highpass_hz)
            waveform = torch.fft.irfft(spectrum, n=waveform.shape[-1], dim=-1)

        features = torch.log(self.mel(waveform) + self.log_offset)
        if self.normalize == "per_sample_zscore":
            flat = features.flatten(1)
            mean = flat.mean(dim=1).view(-1, 1, 1)
            std = flat.std(dim=1).clamp_min(1e-8).view(-1, 1, 1)
            features = (features - mean) / std
        return features.unsqueeze(1)
