from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf
from torch.utils.data import Dataset

from phase.data.manifest import resolve


class WindowDataset(Dataset):
    def __init__(
        self,
        index: dict[str, Any] | str | Path,
        root: str | Path,
        split: str | None = None,
        transform: Any = None,
        classes: list[str] | None = None,
    ):
        if not isinstance(index, dict):
            index = json.loads(resolve(index).read_text(encoding="utf-8"))

        self.root = resolve(root)
        self.sample_rate = index["sample_rate"]
        self.frame = index["frame"]
        self.transform = transform
        self.windows = [w for w in index["windows"] if split is None or w["split"] == split]
        if not self.windows:
            raise ValueError(f"no windows for split {split!r}")

        self.classes = classes or sorted({w["class"] for w in index["windows"]})
        self.class_to_index = {name: i for i, name in enumerate(self.classes)}

    def __len__(self) -> int:
        return len(self.windows)

    def read(self, position: int) -> np.ndarray:
        window = self.windows[position]
        audio, _ = sf.read(
            str(self.root / window["path"]),
            start=window["start"],
            frames=window["frames"],
            dtype="float32",
            always_2d=False,
        )
        audio = np.asarray(audio, dtype=np.float32)
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        if audio.size < self.frame:
            audio = np.pad(audio, (0, self.frame - audio.size))
        return audio

    def __getitem__(self, position: int) -> tuple[Any, int]:
        window = self.windows[position]
        audio = self.read(position)
        label = self.class_to_index[window["class"]]
        if self.transform is not None:
            return self.transform(audio, self.sample_rate), label
        return audio, label

    def recording_ids(self) -> list[str]:
        return [w["recording_id"] for w in self.windows]
