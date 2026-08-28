from __future__ import annotations

import librosa
import numpy as np

from phase.config import LogMelConfig


def compute(waveform: np.ndarray, sample_rate: int, config: LogMelConfig) -> np.ndarray:
    fmax = min(config.fmax, sample_rate / 2)
    spectrogram = librosa.feature.melspectrogram(
        y=np.asarray(waveform, dtype=np.float32),
        sr=sample_rate,
        n_fft=config.n_fft,
        hop_length=config.hop_length,
        n_mels=config.n_mels,
        fmin=config.fmin,
        fmax=fmax,
        power=config.power,
    )
    return np.log(spectrogram + config.log_offset).astype(np.float32)
