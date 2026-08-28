from __future__ import annotations

import numpy as np
from scipy.signal import butter, hilbert, sosfiltfilt

from phase.config import DemonConfig

NYQUIST_MARGIN = 0.98


def analysis_band(sample_rate: int, config: DemonConfig) -> tuple[float, float]:
    nyquist = sample_rate / 2
    high = min(config.band_high, nyquist * NYQUIST_MARGIN)
    low = min(config.band_low, high * 0.5)
    return low, high


def envelope(waveform: np.ndarray, sample_rate: int, config: DemonConfig) -> np.ndarray:
    low, high = analysis_band(sample_rate, config)
    sos = butter(4, [low, high], btype="bandpass", fs=sample_rate, output="sos")
    band = sosfiltfilt(sos, np.asarray(waveform, dtype=np.float64))

    detected = np.abs(hilbert(band))
    smoother = butter(4, config.envelope_lowpass, btype="lowpass", fs=sample_rate, output="sos")
    return sosfiltfilt(smoother, detected)


def compute(waveform: np.ndarray, sample_rate: int, config: DemonConfig) -> np.ndarray:
    detected = envelope(waveform, sample_rate, config)

    decimation = max(1, int(sample_rate // (4 * config.envelope_lowpass)))
    reduced = detected[::decimation]
    reduced_rate = sample_rate / decimation
    reduced = reduced - reduced.mean()
    if reduced.size < 8:
        return np.zeros(1, dtype=np.float32)

    windowed = reduced * np.hanning(reduced.size)
    spectrum = np.abs(np.fft.rfft(windowed, n=config.n_fft))
    freqs = np.fft.rfftfreq(config.n_fft, 1 / reduced_rate)

    keep = spectrum[freqs <= config.max_modulation_hz]
    peak = keep.max()
    return (keep / peak).astype(np.float32) if peak > 0 else keep.astype(np.float32)


def peak_frequencies(
    spectrum: np.ndarray, sample_rate: int, config: DemonConfig, top: int = 5
) -> np.ndarray:
    decimation = max(1, int(sample_rate // (4 * config.envelope_lowpass)))
    reduced_rate = sample_rate / decimation
    freqs = np.fft.rfftfreq(config.n_fft, 1 / reduced_rate)
    freqs = freqs[freqs <= config.max_modulation_hz][: spectrum.size]

    interior = np.arange(1, spectrum.size - 1)
    if interior.size == 0:
        return np.zeros(0, dtype=np.float32)
    local = interior[
        (spectrum[interior] > spectrum[interior - 1])
        & (spectrum[interior] > spectrum[interior + 1])
    ]
    if local.size == 0:
        return np.zeros(0, dtype=np.float32)
    strongest = local[np.argsort(spectrum[local])[::-1][:top]]
    return np.sort(freqs[strongest]).astype(np.float32)
