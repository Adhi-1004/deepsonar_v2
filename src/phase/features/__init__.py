from __future__ import annotations

import numpy as np
from scipy.signal import butter, sosfiltfilt

from phase.config import FeatureConfig
from phase.features import cqt, demon, lofar, logmel

BACKENDS = {
    "logmel": logmel.compute,
    "cqt": cqt.compute,
    "demon": demon.compute,
    "lofar": lofar.compute,
}


def normalise(array: np.ndarray, mode: str) -> np.ndarray:
    if mode == "none":
        return array
    std = array.std()
    if std < 1e-8:
        return np.zeros_like(array)
    return ((array - array.mean()) / std).astype(np.float32)


def highpass(waveform: np.ndarray, sample_rate: int, cutoff: float) -> np.ndarray:
    if cutoff <= 0:
        return waveform
    sos = butter(4, cutoff, btype="highpass", fs=sample_rate, output="sos")
    return sosfiltfilt(sos, np.asarray(waveform, dtype=np.float64)).astype(np.float32)


def compute(waveform: np.ndarray, sample_rate: int, config: FeatureConfig) -> np.ndarray:
    waveform = highpass(waveform, sample_rate, config.highpass_hz)
    array = BACKENDS[config.name](waveform, sample_rate, config.params)
    return normalise(array, config.normalize)
