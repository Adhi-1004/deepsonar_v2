from __future__ import annotations

from typing import Any

import torch

from phase.config import AugmentOp


class Augmentation:
    name = "base"

    def __init__(self, op: AugmentOp):
        self.op = op
        self.prob = op.prob
        self.params = op.params

    def sample_params(
        self, n: int, sample_rate: int, generator: torch.Generator, device: torch.device
    ) -> dict[str, torch.Tensor]:
        raise NotImplementedError

    def transform(
        self, batch: torch.Tensor, sample_rate: int, drawn: dict[str, torch.Tensor]
    ) -> torch.Tensor:
        raise NotImplementedError

    def __call__(
        self, batch: torch.Tensor, sample_rate: int, generator: torch.Generator
    ) -> torch.Tensor:
        if batch.dim() != 2:
            raise ValueError(f"{self.name} expects (batch, samples), got {tuple(batch.shape)}")

        drawn = self.sample_params(batch.shape[0], sample_rate, generator, batch.device)
        transformed = self.transform(batch, sample_rate, drawn)

        if self.prob >= 1.0:
            return transformed
        keep = (
            torch.rand(batch.shape[0], generator=generator, device=batch.device) < self.prob
        ).unsqueeze(1)
        return torch.where(keep, transformed, batch)


class SpectralAugmentation(Augmentation):
    """An augmentation expressible as a per-item frequency response.

    Several of these compose into a single rFFT/irFFT pair rather than one each.
    """

    def response(
        self, n_samples: int, sample_rate: int, drawn: dict[str, torch.Tensor], device: torch.device
    ) -> torch.Tensor:
        raise NotImplementedError

    def transform(
        self, batch: torch.Tensor, sample_rate: int, drawn: dict[str, torch.Tensor]
    ) -> torch.Tensor:
        response = self.response(batch.shape[-1], sample_rate, drawn, batch.device)
        return match_rms(apply_transfer(batch, response), batch)


def uniform(
    low: float, high: float, n: int, generator: torch.Generator, device: torch.device
) -> torch.Tensor:
    return torch.rand(n, generator=generator, device=device) * (high - low) + low


def log_uniform(
    low: float, high: float, n: int, generator: torch.Generator, device: torch.device
) -> torch.Tensor:
    span = torch.rand(n, generator=generator, device=device)
    return torch.exp(
        span * (torch.log(torch.tensor(high)) - torch.log(torch.tensor(low)))
        + torch.log(torch.tensor(low))
    )


def randint(
    low: int, high: int, n: int, generator: torch.Generator, device: torch.device
) -> torch.Tensor:
    return torch.randint(low, high + 1, (n,), generator=generator, device=device)


def pair(value: Any) -> tuple[float, float]:
    if isinstance(value, (list, tuple)):
        return float(value[0]), float(value[1])
    return float(value), float(value)


def frequencies(n_samples: int, sample_rate: int, device: torch.device) -> torch.Tensor:
    return torch.fft.rfftfreq(n_samples, 1.0 / sample_rate, device=device)


def apply_transfer(batch: torch.Tensor, response: torch.Tensor) -> torch.Tensor:
    spectrum = torch.fft.rfft(batch, dim=-1)
    return torch.fft.irfft(spectrum * response, n=batch.shape[-1], dim=-1)


def rms(batch: torch.Tensor) -> torch.Tensor:
    return batch.pow(2).mean(dim=-1, keepdim=True).clamp_min(1e-20).sqrt()


def match_rms(transformed: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
    return transformed * (rms(reference) / rms(transformed))
