from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf
import torch

from phase.augment.lloyds_mirror import LloydsMirror
from phase.augment.multipath import Multipath
from phase.augment.transmission_loss import TransmissionLoss
from phase.config import AugmentOp, DemonConfig
from phase.data.manifest import resolve
from phase.features import demon as demon_module

TRANSIT = re.compile(r"\s+")

SOURCE_DEPTH_M = 5.0

BANDS = [(10, 100), (100, 500), (500, 2000), (2000, 8000)]


@dataclass(frozen=True)
class Pair:
    stem: str
    label: int
    original: Path
    rendered: Path
    range_m: float
    receiver_depth_m: float


def load_geometry(root: Path) -> dict[str, tuple[int, float, float]]:
    geometry: dict[str, tuple[int, float, float]] = {}
    for name in ("train_list.txt", "test_list.txt"):
        listing = root / "DS3500" / name
        if not listing.is_file():
            continue
        for line in listing.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            fields = TRANSIT.split(line.strip())
            if len(fields) < 4:
                continue
            stem = Path(fields[0].replace("\\", "/")).stem
            geometry[stem] = (int(fields[1]), float(fields[2]), float(fields[3]))
    return geometry


def matched_pairs(root: str | Path) -> list[Pair]:
    root = resolve(root)
    geometry = load_geometry(root)

    originals = {p.stem: p for p in (root / "ShipsEar" / "shipsear_5s_16k").rglob("*.wav")}
    rendered = {p.stem: p for p in (root / "DS3500").glob("*.wav")}

    pairs = []
    for stem, (label, range_km, depth_km) in sorted(geometry.items()):
        if stem in originals and stem in rendered:
            pairs.append(
                Pair(
                    stem=stem,
                    label=label,
                    original=originals[stem],
                    rendered=rendered[stem],
                    range_m=range_km * 1000.0,
                    receiver_depth_m=depth_km * 1000.0,
                )
            )
    return pairs


def read(path: Path) -> tuple[np.ndarray, int]:
    audio, rate = sf.read(str(path), dtype="float32", always_2d=False)
    audio = np.asarray(audio, dtype=np.float32)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    return audio, rate


def analytic_ops(range_m: float, receiver_depth_m: float, source_depth_m: float = SOURCE_DEPTH_M):
    lloyd = LloydsMirror(
        AugmentOp(
            name="lloyds_mirror",
            prob=1.0,
            params={
                "source_depth_m": source_depth_m,
                "receiver_depth_m": receiver_depth_m,
                "range_m": range_m,
                "sound_speed": 1500.0,
                "reflection_coefficient": -1.0,
            },
        )
    )
    loss = TransmissionLoss(
        AugmentOp(
            name="transmission_loss",
            prob=1.0,
            params={"range_m": range_m, "spreading": "cylindrical"},
        )
    )
    return lloyd, loss


def render_analytic(
    batch: torch.Tensor,
    sample_rate: int,
    range_m: float,
    receiver_depth_m: float,
    generator: torch.Generator,
    with_multipath: bool = False,
) -> torch.Tensor:
    out = batch
    for op in analytic_ops(range_m, receiver_depth_m):
        out = op(out, sample_rate, generator)
    if with_multipath:
        multipath = Multipath(
            AugmentOp(name="multipath", prob=1.0, params={"n_arrivals": 3, "delay_ms": [1.0, 20.0]})
        )
        out = multipath(out, sample_rate, generator)
    return out


def log_spectrum(
    audio: np.ndarray, sample_rate: int, n_fft: int = 2048
) -> tuple[np.ndarray, np.ndarray]:
    hop = n_fft // 2
    frames = max(1, 1 + (audio.size - n_fft) // hop)
    window = np.hanning(n_fft)
    stack = np.stack(
        [np.abs(np.fft.rfft(audio[i * hop : i * hop + n_fft] * window)) for i in range(frames)]
    )
    freqs = np.fft.rfftfreq(n_fft, 1 / sample_rate)
    return 20 * np.log10(stack + 1e-10), freqs


def spectral_distance(
    a: np.ndarray, b: np.ndarray, sample_rate: int
) -> tuple[float, dict[str, float]]:
    left, freqs = log_spectrum(a, sample_rate)
    right, _ = log_spectrum(b, sample_rate)
    frames = min(left.shape[0], right.shape[0])
    left, right = left[:frames], right[:frames]

    left = left - left.mean()
    right = right - right.mean()
    difference = left - right

    overall = float(np.sqrt((difference**2).mean()))
    per_band = {}
    for low, high in BANDS:
        mask = (freqs >= low) & (freqs < high)
        if mask.any():
            per_band[f"{low}-{high}Hz"] = float(np.sqrt((difference[:, mask] ** 2).mean()))
    return overall, per_band


def demon_agreement(
    a: np.ndarray, b: np.ndarray, sample_rate: int, config: DemonConfig
) -> dict[str, float]:
    left = demon_module.compute(a, sample_rate, config)
    right = demon_module.compute(b, sample_rate, config)
    size = min(left.size, right.size)
    left, right = left[:size], right[:size]

    denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
    cosine = float(left @ right / denominator) if denominator > 0 else 0.0

    peaks_left = demon_module.peak_frequencies(left, sample_rate, config, top=3)
    peaks_right = demon_module.peak_frequencies(right, sample_rate, config, top=3)
    if peaks_left.size and peaks_right.size:
        shift = float(np.mean([np.min(np.abs(peaks_right - p)) for p in peaks_left]))
    else:
        shift = float("nan")
    return {"demon_cosine": cosine, "demon_peak_shift_hz": shift}


VARIANTS = {
    "none": (),
    "lloyd": ("lloyd",),
    "tl": ("tl",),
    "lloyd+tl": ("lloyd", "tl"),
    "lloyd+tl+multipath": ("lloyd", "tl", "multipath"),
}


def apply_variant(
    audio: np.ndarray,
    sample_rate: int,
    terms: tuple[str, ...],
    range_m: float,
    receiver_depth_m: float,
    generator: torch.Generator,
) -> np.ndarray:
    if not terms:
        return audio
    batch = torch.from_numpy(audio).unsqueeze(0)
    lloyd, loss = analytic_ops(range_m, receiver_depth_m)
    for term in terms:
        if term == "lloyd":
            batch = lloyd(batch, sample_rate, generator)
        elif term == "tl":
            batch = loss(batch, sample_rate, generator)
        elif term == "multipath":
            multipath = Multipath(
                AugmentOp(
                    name="multipath",
                    prob=1.0,
                    params={"n_arrivals": 3, "delay_ms": [1.0, 20.0], "decay_db": 8.0},
                )
            )
            batch = multipath(batch, sample_rate, generator)
    return batch[0].numpy()


def validate(
    pairs: list[Pair],
    demon_config: DemonConfig,
    variants: dict[str, tuple[str, ...]] | None = None,
    seed: int = 0,
) -> list[dict[str, Any]]:
    variants = variants or VARIANTS
    generator = torch.Generator().manual_seed(seed)
    rows: list[dict[str, Any]] = []

    for index, pair in enumerate(pairs, 1):
        original, sample_rate = read(pair.original)
        rendered, _ = read(pair.rendered)

        for name, terms in variants.items():
            candidate = apply_variant(
                original, sample_rate, terms, pair.range_m, pair.receiver_depth_m, generator
            )
            overall, per_band = spectral_distance(candidate, rendered, sample_rate)
            row = {
                "stem": pair.stem,
                "variant": name,
                "label": pair.label,
                "range_km": pair.range_m / 1000.0,
                "depth_km": pair.receiver_depth_m / 1000.0,
                "lsd_db": overall,
                **{f"lsd_{k}": v for k, v in per_band.items()},
                **demon_agreement(candidate, rendered, sample_rate, demon_config),
            }
            rows.append(row)
        if index % 25 == 0:
            print(f"  {index}/{len(pairs)} pairs", flush=True)
    return rows


def aggregate(rows: list[dict[str, Any]], by: str | None = None) -> dict[str, Any]:
    keys = [
        k for k in rows[0] if isinstance(rows[0][k], float) and k not in {"range_km", "depth_km"}
    ]
    out: dict[str, Any] = {}

    groups: dict[Any, list[dict[str, Any]]] = {}
    for row in rows:
        key = (row["variant"], row[by]) if by else row["variant"]
        groups.setdefault(key, []).append(row)

    for key, members in sorted(groups.items(), key=lambda kv: str(kv[0])):
        summary = {}
        for metric in keys:
            values = np.array([m[metric] for m in members], dtype=np.float64)
            values = values[np.isfinite(values)]
            if values.size:
                summary[metric] = {
                    "mean": float(values.mean()),
                    "std": float(values.std(ddof=1)) if values.size > 1 else 0.0,
                }
        summary["n"] = len(members)
        out[str(key)] = summary
    return out
