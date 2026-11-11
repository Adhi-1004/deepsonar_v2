"""data package — re-exports and feature-cache helpers."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from phase.data.manifest import resolve


# ---------------------------------------------------------------------------
# Feature cache helpers (used by 06_evaluate.py and linear probe scripts)
# ---------------------------------------------------------------------------


def load(path: str | Path) -> tuple[np.ndarray, dict[str, Any]]:
    """Load a pre-computed feature cache produced by ``04_segment_and_cache.py``.

    Returns
    -------
    features : ndarray of shape (N, *feature_shape) – raw (un-flattened) arrays
    meta     : dict with keys ``labels``, ``splits``, ``recording_ids``, ``classes``
    """
    p = resolve(path)
    if not p.exists():
        raise FileNotFoundError(f"feature cache not found: {p}")

    if p.suffix == ".npz":
        archive = np.load(str(p), allow_pickle=False)
        features = archive["features"]
        meta = {
            "labels": archive["labels"].tolist(),
            "splits": archive["splits"].tolist(),
            "recording_ids": archive["recording_ids"].tolist()
            if "recording_ids" in archive else None,
            "classes": archive["classes"].tolist()
            if "classes" in archive else sorted(set(archive["labels"].tolist())),
        }
        return features, meta

    # JSON index pointing to per-file .npy files (alternative layout)
    if p.suffix == ".json":
        index = json.loads(p.read_text(encoding="utf-8"))
        root = p.parent
        rows = index["windows"]
        features = np.stack([np.load(str(root / r["npy"])) for r in rows])
        meta = {
            "labels": [r["class"] for r in rows],
            "splits": [r["split"] for r in rows],
            "recording_ids": [r.get("recording_id") for r in rows],
            "classes": index.get("classes", sorted({r["class"] for r in rows})),
        }
        return features, meta

    raise ValueError(f"Unsupported cache format: {p.suffix}. Expected .npz or .json")


def save(
    features: np.ndarray,
    labels: list[str],
    splits: list[str],
    recording_ids: list[str | None],
    classes: list[str],
    path: str | Path,
) -> Path:
    """Save a feature cache as a compressed .npz file."""
    out = resolve(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        str(out),
        features=features,
        labels=np.array(labels),
        splits=np.array(splits),
        recording_ids=np.array([r or "" for r in recording_ids]),
        classes=np.array(classes),
    )
    return out

