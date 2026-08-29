from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf

from phase.config import FeatureConfig
from phase.data.manifest import resolve
from phase.features import compute


def read_window(root: Path, window: dict[str, Any]) -> np.ndarray:
    audio, _ = sf.read(
        str(root / window["path"]),
        start=window["start"],
        frames=window["frames"],
        dtype="float32",
        always_2d=False,
    )
    audio = np.asarray(audio, dtype=np.float32)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if audio.size < window["frames"]:
        audio = np.pad(audio, (0, window["frames"] - audio.size))
    return audio


def build(index: dict[str, Any], root: Path, config: FeatureConfig, prefix: str) -> list[str]:
    sample_rate = index["sample_rate"]
    windows = index["windows"]
    if not windows:
        return ["nothing to cache"]

    probe = compute(read_window(root, windows[0]), sample_rate, config)
    store = resolve(prefix + "_features.npy")
    store.parent.mkdir(parents=True, exist_ok=True)

    array = np.lib.format.open_memmap(
        store, mode="w+", dtype=np.float32, shape=(len(windows), *probe.shape)
    )
    for position, window in enumerate(windows):
        array[position] = compute(read_window(root, window), sample_rate, config)
        if (position + 1) % 250 == 0:
            print(f"  cached {position + 1}/{len(windows)}", file=sys.stderr)
    array.flush()

    meta = resolve(prefix + "_features.json")
    meta.write_text(
        json.dumps(
            {
                "feature": config.name,
                "normalize": config.normalize,
                "sample_rate": sample_rate,
                "shape": [len(windows), *probe.shape],
                "labels": [w["class"] for w in windows],
                "splits": [w["split"] for w in windows],
                "recording_ids": [w["recording_id"] for w in windows],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    size = store.stat().st_size / 1e9
    return [
        f"wrote {store.name} shape {(len(windows), *probe.shape)} ({size:.2f} GB)",
        f"wrote {meta.name}",
    ]


def load(prefix: str) -> tuple[np.ndarray, dict[str, Any]]:
    array = np.load(resolve(prefix + "_features.npy"), mmap_mode="r")
    meta = json.loads(resolve(prefix + "_features.json").read_text(encoding="utf-8"))
    return array, meta
