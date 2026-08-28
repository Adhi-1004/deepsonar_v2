from __future__ import annotations

from pathlib import Path

import numpy as np
import soundfile as sf
import torch

from phase.augment.base import Augmentation, apply_transfer, frequencies, pair, rms, uniform
from phase.data.manifest import resolve


class ContaminationError(RuntimeError):
    pass


def wenz_spectrum(
    freqs: torch.Tensor, sea_state: torch.Tensor, shipping: float = 0.5
) -> torch.Tensor:
    khz = (freqs / 1000.0).clamp_min(1e-4).unsqueeze(0)
    state = sea_state.unsqueeze(1)

    turbulence = 17.0 - 30.0 * torch.log10(khz)
    traffic = (
        40.0 + 20.0 * (shipping - 0.5) + 26.0 * torch.log10(khz) - 60.0 * torch.log10(khz + 0.03)
    )
    wind = (
        50.0
        + 7.5 * torch.sqrt(state.clamp_min(0.0))
        + 20.0 * torch.log10(khz)
        - 40.0 * torch.log10(khz + 0.4)
    )
    thermal = -15.0 + 20.0 * torch.log10(khz)

    shape = torch.broadcast_shapes(turbulence.shape, traffic.shape, wind.shape, thermal.shape)
    parts = torch.stack(
        [term.expand(shape) for term in (turbulence, traffic, wind, thermal)], dim=0
    )
    return torch.logsumexp(parts * (np.log(10.0) / 10.0), dim=0) * (10.0 / np.log(10.0))


class NoiseBank:
    def __init__(self, root: str | Path, allowed: list[str] | None = None):
        self.root = resolve(root)
        self.allowed = set(allowed) if allowed is not None else None
        self.clips: list[Path] = sorted(self.root.rglob("*.wav"))
        if self.allowed is not None:
            blocked = [p for p in self.clips if p.as_posix() not in self.allowed]
            self.clips = [p for p in self.clips if p.as_posix() in self.allowed]
            if blocked and not self.clips:
                raise ContaminationError(
                    f"every clip under {self.root} is outside the held-out noise list; "
                    "using evaluation audio as mixing noise would contaminate the results"
                )

    def draw(self, n: int, length: int, generator: torch.Generator) -> torch.Tensor:
        if not self.clips:
            raise ContaminationError(f"no permitted noise clips under {self.root}")
        picks = torch.randint(len(self.clips), (n,), generator=generator)
        out = torch.zeros(n, length)
        for position, index in enumerate(picks.tolist()):
            audio, _ = sf.read(str(self.clips[index]), dtype="float32", always_2d=False)
            audio = np.asarray(audio, dtype=np.float32)
            if audio.ndim > 1:
                audio = audio.mean(axis=1)
            if audio.size < length:
                audio = np.tile(audio, int(np.ceil(length / max(audio.size, 1))))
            start = int(
                torch.randint(max(1, audio.size - length), (1,), generator=generator).item()
            )
            out[position] = torch.from_numpy(audio[start : start + length].copy())
        return out


class NoiseMixing(Augmentation):
    name = "noise_mixing"

    def __init__(self, op, bank: NoiseBank | None = None):
        super().__init__(op)
        self.bank = bank
        self.sources = list(self.params.get("sources", ["wenz_synthetic"]))
        if bank is None and "shipsear_class_e" in self.sources:
            self.sources = [s for s in self.sources if s != "shipsear_class_e"]

    def sample_params(
        self, n: int, sample_rate: int, generator: torch.Generator, device: torch.device
    ) -> dict[str, torch.Tensor]:
        snr_low, snr_high = pair(self.params.get("snr_db", [0.0, 20.0]))
        state_low, state_high = pair(self.params.get("sea_state", [1, 5]))
        return {
            "snr_db": uniform(snr_low, snr_high, n, generator, device),
            "sea_state": uniform(state_low, state_high, n, generator, device),
            "use_recorded": torch.rand(n, generator=generator, device=device),
        }

    def synthetic(
        self,
        n: int,
        length: int,
        sample_rate: int,
        sea_state: torch.Tensor,
        generator: torch.Generator,
    ) -> torch.Tensor:
        white = torch.randn(n, length, generator=generator, device=sea_state.device)
        freqs = frequencies(length, sample_rate, sea_state.device)
        level = wenz_spectrum(freqs, sea_state)
        response = torch.pow(10.0, (level - level.amax(dim=1, keepdim=True)) / 20.0)
        return apply_transfer(white, response)

    def transform(
        self, batch: torch.Tensor, sample_rate: int, drawn: dict[str, torch.Tensor]
    ) -> torch.Tensor:
        n, length = batch.shape
        generator = torch.Generator(device="cpu")
        generator.manual_seed(int(torch.randint(2**31 - 1, (1,)).item()))

        noise = self.synthetic(n, length, sample_rate, drawn["sea_state"], generator)
        if self.bank is not None and "shipsear_class_e" in self.sources:
            recorded = self.bank.draw(n, length, generator).to(batch.device)
            pick = (drawn["use_recorded"] < 0.5).unsqueeze(1)
            noise = torch.where(pick, recorded, noise)

        scale = rms(batch) / rms(noise) * torch.pow(10.0, -drawn["snr_db"].unsqueeze(1) / 20.0)
        return batch + noise * scale
