from __future__ import annotations

import torch

from phase.augment.base import Augmentation, pair, uniform


class Doppler(Augmentation):
    name = "doppler"

    def sample_params(
        self, n: int, sample_rate: int, generator: torch.Generator, device: torch.device
    ) -> dict[str, torch.Tensor]:
        low, high = pair(self.params.get("rate", [0.98, 1.02]))
        return {"rate": uniform(low, high, n, generator, device)}

    def transform(
        self, batch: torch.Tensor, sample_rate: int, drawn: dict[str, torch.Tensor]
    ) -> torch.Tensor:
        n, length = batch.shape
        rate = drawn["rate"].unsqueeze(1)

        positions = torch.arange(length, device=batch.device, dtype=batch.dtype).unsqueeze(0) * rate
        positions = positions.clamp(0, length - 1)

        left = positions.floor().long()
        right = (left + 1).clamp_max(length - 1)
        weight = (positions - left).to(batch.dtype)

        lower = torch.gather(batch, 1, left)
        upper = torch.gather(batch, 1, right)
        return lower + (upper - lower) * weight
