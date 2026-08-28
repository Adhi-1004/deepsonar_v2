from __future__ import annotations

import torch

from phase.augment.base import SpectralAugmentation, frequencies, pair, uniform

SPREADING_EXPONENT = {"cylindrical": 10.0, "spherical": 20.0}


def thorp_absorption(khz: torch.Tensor) -> torch.Tensor:
    squared = khz**2
    return (
        0.11 * squared / (1 + squared)
        + 44.0 * squared / (4100 + squared)
        + 2.75e-4 * squared
        + 0.003
    )


class TransmissionLoss(SpectralAugmentation):
    name = "transmission_loss"

    def sample_params(
        self, n: int, sample_rate: int, generator: torch.Generator, device: torch.device
    ) -> dict[str, torch.Tensor]:
        range_low, range_high = pair(self.params.get("range_m", [200.0, 6000.0]))
        return {"range": uniform(range_low, range_high, n, generator, device)}

    def response(
        self, n_samples: int, sample_rate: int, drawn: dict[str, torch.Tensor], device: torch.device
    ) -> torch.Tensor:
        spreading = str(self.params.get("spreading", "cylindrical"))
        exponent = SPREADING_EXPONENT[spreading]

        freqs = frequencies(n_samples, sample_rate, device)
        alpha = thorp_absorption(freqs / 1000.0).unsqueeze(0)

        kilometres = (drawn["range"] / 1000.0).unsqueeze(1)
        loss_db = (
            exponent * torch.log10(drawn["range"].clamp_min(1.0)).unsqueeze(1) + alpha * kilometres
        )

        return torch.pow(10.0, -(loss_db - loss_db.amin(dim=1, keepdim=True)) / 20.0).to(
            torch.complex64
        )
