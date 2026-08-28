from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf

from phase.config import FeatureConfig
from phase.data.manifest import resolve
from phase.features import compute
from phase.features import demon as demon_module


def read_window(path: Path, seconds: float, start: float = 0.0) -> tuple[np.ndarray, int]:
    with sf.SoundFile(str(path)) as handle:
        rate = handle.samplerate
        handle.seek(int(start * rate))
        audio = handle.read(frames=int(seconds * rate), dtype="float32", always_2d=False)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    return np.asarray(audio, dtype=np.float32), rate


def dc_before_after(raw: Path, clean: Path, out: str | Path, seconds: float = 20.0) -> Path:
    before, rate = read_window(raw, seconds)
    after, clean_rate = read_window(clean, seconds)

    fig, axes = plt.subplots(2, 2, figsize=(12, 6))
    for column, (audio, sr, title) in enumerate(
        [(before, rate, "raw"), (after, clean_rate, "cleaned")]
    ):
        time = np.arange(audio.size) / sr
        axes[0, column].plot(time, audio, linewidth=0.4)
        axes[0, column].axhline(audio.mean(), color="crimson", linewidth=1.0)
        axes[0, column].set_title(f"{title}  DC = {audio.mean():+.5f}")
        axes[0, column].set_xlabel("s")

        spectrum = np.abs(np.fft.rfft((audio - audio.mean()) * np.hanning(audio.size))) ** 2
        freqs = np.fft.rfftfreq(audio.size, 1 / sr)
        keep = freqs < 200
        axes[1, column].semilogy(freqs[keep], spectrum[keep] + 1e-20, linewidth=0.7)
        axes[1, column].set_xlabel("Hz")
        axes[1, column].set_ylabel("power")

    fig.suptitle(f"DC offset removal — {raw.name}")
    fig.tight_layout()
    target = resolve(out)
    target.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(target, dpi=130)
    plt.close(fig)
    return target


def feature_panel(
    path: Path, configs: dict[str, FeatureConfig], out: str | Path, seconds: float = 30.0
) -> Path:
    audio, rate = read_window(path, seconds)

    fig, axes = plt.subplots(1, len(configs), figsize=(5 * len(configs), 4))
    axes = np.atleast_1d(axes)
    for axis, (name, config) in zip(axes, configs.items(), strict=False):
        feature = compute(audio, rate, config)
        if feature.ndim == 1:
            band = demon_module.analysis_band(rate, config.params)
            freqs = np.linspace(0, config.params.max_modulation_hz, feature.size)
            axis.plot(freqs, feature, linewidth=0.8)
            axis.set_xlabel("modulation Hz")
            axis.set_title(f"{name}  band {band[0]:.0f}-{band[1]:.0f} Hz")
        else:
            axis.imshow(feature, aspect="auto", origin="lower", cmap="magma")
            axis.set_title(f"{name}  {feature.shape}")
            axis.set_xlabel("frame")

    fig.suptitle(path.name)
    fig.tight_layout()
    target = resolve(out)
    target.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(target, dpi=130)
    plt.close(fig)
    return target


def augmentation_panel(
    waveform: np.ndarray,
    sample_rate: int,
    ops: dict[str, Any],
    demon_config: Any,
    out: str | Path,
) -> Path:
    import torch

    from phase.features import demon as demon_module

    generator = torch.Generator().manual_seed(0)
    batch = torch.from_numpy(np.asarray(waveform, dtype=np.float32)).unsqueeze(0)

    panels = [("original", waveform)]
    for name, augmentation in ops.items():
        augmentation.prob = 1.0
        panels.append((name, augmentation(batch, sample_rate, generator)[0].numpy()))

    baseline = demon_module.compute(waveform, sample_rate, demon_config)
    reference = demon_module.peak_frequencies(baseline, sample_rate, demon_config, top=3)

    fig, axes = plt.subplots(2, len(panels), figsize=(3.1 * len(panels), 6.4), squeeze=False)
    for column, (name, audio) in enumerate(panels):
        spectrum, _, _, _ = axes[0, column].specgram(
            audio, NFFT=2048, Fs=sample_rate, noverlap=1024, cmap="magma"
        )
        axes[0, column].set_ylim(0, 8000)
        axes[0, column].set_title(name, fontsize=10)
        axes[0, column].set_xlabel("s", fontsize=8)
        if column == 0:
            axes[0, column].set_ylabel("Hz")

        curve = demon_module.compute(audio, sample_rate, demon_config)
        freqs = np.linspace(0, demon_config.max_modulation_hz, curve.size)
        axes[1, column].plot(freqs, curve, linewidth=0.8, color="#B85F18")
        for peak in reference:
            axes[1, column].axvline(peak, color="#2E7089", linestyle="--", linewidth=0.9)
        axes[1, column].set_xlim(0, 60)
        axes[1, column].set_xlabel("modulation Hz", fontsize=8)
        if column == 0:
            axes[1, column].set_ylabel("DEMON")

    fig.suptitle(
        "Physics augmentations — dashed lines mark the original shaft and blade rates", fontsize=11
    )
    fig.tight_layout()
    target = resolve(out)
    target.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(target, dpi=130)
    plt.close(fig)
    return target
