from __future__ import annotations

import librosa
import numpy as np

from phase.config import CQTConfig


def compute(waveform: np.ndarray, sample_rate: int, config: CQTConfig) -> np.ndarray:
    nyquist = sample_rate / 2
    max_bins = int(np.floor(config.bins_per_octave * np.log2(nyquist / config.fmin)))
    n_bins = max(1, min(config.n_bins, max_bins))

    spectrum = librosa.cqt(
        y=np.asarray(waveform, dtype=np.float32),
        sr=sample_rate,
        hop_length=config.hop_length,
        fmin=config.fmin,
        n_bins=n_bins,
        bins_per_octave=config.bins_per_octave,
    )
    return librosa.amplitude_to_db(np.abs(spectrum), ref=np.max).astype(np.float32)
