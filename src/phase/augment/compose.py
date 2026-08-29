from __future__ import annotations

import torch

from phase.augment.base import (
    Augmentation,
    SpectralAugmentation,
    apply_transfer,
    draw,
    match_rms,
)
from phase.augment.doppler import Doppler
from phase.augment.generic import Gain, GaussianNoise, PolarityFlip, RandomCrop, TimeMask
from phase.augment.lloyds_mirror import LloydsMirror
from phase.augment.multipath import Multipath
from phase.augment.noise_mixing import NoiseBank, NoiseMixing
from phase.augment.transmission_loss import TransmissionLoss
from phase.config import AugmentConfig, AugmentOp

REGISTRY = {
    "lloyds_mirror": LloydsMirror,
    "multipath": Multipath,
    "transmission_loss": TransmissionLoss,
    "doppler": Doppler,
    "gain": Gain,
    "gaussian_noise": GaussianNoise,
    "time_mask": TimeMask,
    "polarity_flip": PolarityFlip,
    "random_crop": RandomCrop,
}


def build_op(op: AugmentOp, bank: NoiseBank | None = None) -> Augmentation:
    if op.name == "noise_mixing":
        return NoiseMixing(op, bank)
    if op.name not in REGISTRY:
        raise KeyError(f"no augmentation registered under {op.name!r}")
    return REGISTRY[op.name](op)


class Pipeline:
    def __init__(self, config: AugmentConfig, bank: NoiseBank | None = None):
        self.config = config
        self.ops = [build_op(op, bank) for op in config.ops]

    @property
    def names(self) -> list[str]:
        return [op.name for op in self.ops]

    def _runs(self) -> list[tuple[bool, list[Augmentation]]]:
        runs: list[tuple[bool, list[Augmentation]]] = []
        for op in self.ops:
            spectral = isinstance(op, SpectralAugmentation)
            if runs and runs[-1][0] and spectral:
                runs[-1][1].append(op)
            else:
                runs.append((spectral, [op]))
        return runs

    def _apply_spectral_run(
        self,
        batch: torch.Tensor,
        sample_rate: int,
        generator: torch.Generator,
        ops: list[Augmentation],
    ) -> torch.Tensor:
        n, length = batch.shape
        combined = torch.ones(n, length // 2 + 1, dtype=torch.complex64, device=batch.device)

        for op in ops:
            drawn = op.sample_params(n, sample_rate, generator, batch.device)
            response = op.response(length, sample_rate, drawn, batch.device).to(torch.complex64)
            if op.prob < 1.0:
                keep = (draw(n, generator, batch.device) < op.prob).unsqueeze(1)
                response = torch.where(keep, response, torch.ones_like(response))
            combined = combined * response

        return match_rms(apply_transfer(batch, combined), batch)

    def __call__(
        self, batch: torch.Tensor, sample_rate: int, generator: torch.Generator
    ) -> torch.Tensor:
        out = batch
        for spectral, ops in self._runs():
            if spectral and len(ops) > 1:
                out = self._apply_spectral_run(out, sample_rate, generator, ops)
            else:
                for op in ops:
                    out = op(out, sample_rate, generator)
        return out

    def views(
        self, batch: torch.Tensor, sample_rate: int, generator: torch.Generator
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return (
            self(batch, sample_rate, generator),
            self(batch, sample_rate, generator),
        )
