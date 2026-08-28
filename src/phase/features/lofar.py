from __future__ import annotations

import numpy as np
from scipy.signal import stft

from phase.config import LofarConfig


def tpsw(spectrum: np.ndarray, length: int, gap: int) -> np.ndarray:
    kernel = np.ones(2 * length + 1)
    kernel[length - gap : length + gap + 1] = 0.0
    kernel /= kernel.sum()

    padded = np.pad(spectrum, ((length, length), (0, 0)), mode="edge")
    background = np.empty_like(spectrum)
    for index in range(spectrum.shape[1]):
        background[:, index] = np.convolve(padded[:, index], kernel, mode="valid")
    return background


def compute(waveform: np.ndarray, sample_rate: int, config: LofarConfig) -> np.ndarray:
    nperseg = int(round(config.frame_ms * sample_rate / 1000))
    noverlap = nperseg - int(round(config.shift_ms * sample_rate / 1000))

    _, _, transform = stft(
        np.asarray(waveform, dtype=np.float64),
        fs=sample_rate,
        window=config.window,
        nperseg=nperseg,
        noverlap=max(0, noverlap),
        boundary=None,
        padded=False,
    )
    magnitude = np.abs(transform)

    background = tpsw(magnitude, config.tpsw_length, config.tpsw_gap)
    normalised = magnitude / np.maximum(background, 1e-12)
    return np.log(normalised + 1e-6).astype(np.float32)
