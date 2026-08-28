from __future__ import annotations

import torch

from phase.augment.base import SpectralAugmentation, frequencies, pair, uniform


class LloydsMirror(SpectralAugmentation):
    name = "lloyds_mirror"

    def sample_params(
        self, n: int, sample_rate: int, generator: torch.Generator, device: torch.device
    ) -> dict[str, torch.Tensor]:
        source_low, source_high = pair(self.params.get("source_depth_m", [2.0, 10.0]))
        receiver_low, receiver_high = pair(self.params.get("receiver_depth_m", [10.0, 60.0]))
        range_low, range_high = pair(self.params.get("range_m", [200.0, 6000.0]))

        return {
            "source_depth": uniform(source_low, source_high, n, generator, device),
            "receiver_depth": uniform(receiver_low, receiver_high, n, generator, device),
            "range": uniform(range_low, range_high, n, generator, device),
        }

    def delay(self, drawn: dict[str, torch.Tensor]) -> torch.Tensor:
        speed = float(self.params.get("sound_speed", 1500.0))
        source, receiver, distance = drawn["source_depth"], drawn["receiver_depth"], drawn["range"]

        direct = torch.sqrt(distance**2 + (source - receiver) ** 2)
        reflected = torch.sqrt(distance**2 + (source + receiver) ** 2)
        return (reflected - direct) / speed

    def response(
        self, n_samples: int, sample_rate: int, drawn: dict[str, torch.Tensor], device: torch.device
    ) -> torch.Tensor:
        reflection = float(self.params.get("reflection_coefficient", -1.0))
        freqs = frequencies(n_samples, sample_rate, device)
        phase = 2 * torch.pi * freqs.unsqueeze(0) * self.delay(drawn).unsqueeze(1)
        return 1.0 + reflection * torch.exp(-1j * phase)
