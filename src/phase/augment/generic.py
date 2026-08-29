from __future__ import annotations

import torch

from phase.augment.base import Augmentation, draw, pair, rms, uniform


class Gain(Augmentation):
    name = "gain"

    def sample_params(self, n, sample_rate, generator, device):
        low, high = pair(self.params.get("db_low", -6.0)), pair(self.params.get("db_high", 6.0))
        return {"db": uniform(low[0], high[0], n, generator, device)}

    def transform(self, batch, sample_rate, drawn):
        return batch * torch.pow(10.0, drawn["db"].unsqueeze(1) / 20.0)


class GaussianNoise(Augmentation):
    name = "gaussian_noise"

    def sample_params(self, n, sample_rate, generator, device):
        low = float(self.params.get("snr_db_low", 5.0))
        high = float(self.params.get("snr_db_high", 25.0))
        return {"snr_db": uniform(low, high, n, generator, device)}

    def transform(self, batch, sample_rate, drawn):
        noise = torch.randn(batch.shape, device=batch.device)
        scale = rms(batch) / rms(noise) * torch.pow(10.0, -drawn["snr_db"].unsqueeze(1) / 20.0)
        return batch + noise * scale


class TimeMask(Augmentation):
    name = "time_mask"

    def sample_params(self, n, sample_rate, generator, device):
        fraction = float(self.params.get("max_fraction", 0.1))
        count = int(self.params.get("n_masks", 2))
        return {
            "width": draw((n, count), generator, device) * fraction,
            "start": draw((n, count), generator, device),
        }

    def transform(self, batch, sample_rate, drawn):
        length = batch.shape[-1]
        positions = torch.arange(length, device=batch.device).unsqueeze(0)
        out = batch
        for index in range(drawn["width"].shape[1]):
            width = (drawn["width"][:, index] * length).long().unsqueeze(1)
            start = (
                (drawn["start"][:, index] * (length - width.squeeze(1)).clamp_min(1))
                .long()
                .unsqueeze(1)
            )
            inside = (positions >= start) & (positions < start + width)
            out = out.masked_fill(inside, 0.0)
        return out


class PolarityFlip(Augmentation):
    name = "polarity_flip"

    def sample_params(self, n, sample_rate, generator, device):
        return {}

    def transform(self, batch, sample_rate, drawn):
        return -batch


class RandomCrop(Augmentation):
    name = "random_crop"

    def sample_params(self, n, sample_rate, generator, device):
        return {}

    def transform(self, batch, sample_rate, drawn):
        return batch
