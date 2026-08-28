from __future__ import annotations

import torch

from phase.augment.base import SpectralAugmentation, frequencies, pair, randint, uniform


class Multipath(SpectralAugmentation):
    name = "multipath"

    def sample_params(
        self, n: int, sample_rate: int, generator: torch.Generator, device: torch.device
    ) -> dict[str, torch.Tensor]:
        arrivals_low, arrivals_high = pair(self.params.get("n_arrivals", [2, 5]))
        delay_low, delay_high = pair(self.params.get("delay_ms", [1.0, 40.0]))
        decay_low, decay_high = pair(self.params.get("decay_db", [3.0, 18.0]))

        most = int(arrivals_high)
        count = randint(int(arrivals_low), most, n, generator, device)
        order = torch.arange(most, device=device).unsqueeze(0)
        active = order < count.unsqueeze(1)

        delays = uniform(delay_low, delay_high, n * most, generator, device).reshape(n, most)
        decays = uniform(decay_low, decay_high, n, generator, device).unsqueeze(1)

        return {
            "delay_s": (delays / 1000.0) * active,
            "gains": torch.pow(10.0, -(decays * order) / 20.0) * active,
        }

    def response(
        self, n_samples: int, sample_rate: int, drawn: dict[str, torch.Tensor], device: torch.device
    ) -> torch.Tensor:
        freqs = frequencies(n_samples, sample_rate, device).unsqueeze(0)
        delays, gains = drawn["delay_s"], drawn["gains"]

        total = torch.zeros(delays.shape[0], freqs.shape[-1], dtype=torch.complex64, device=device)
        for arrival in range(delays.shape[1]):
            gain = gains[:, arrival].unsqueeze(1)
            if not torch.any(gain > 0):
                continue
            phase = 2 * torch.pi * freqs * delays[:, arrival].unsqueeze(1)
            total = total + gain * torch.exp(-1j * phase)
        return total
