from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

from phase.data.manifest import resolve


class SegmentError(RuntimeError):
    pass


def windows_for(duration: float, seconds: float, hop: float) -> int:
    if duration < seconds:
        return 0
    return int((duration - seconds) // hop) + 1


def build(
    manifest: dict[str, Any],
    splits: dict[str, Any] | None,
    seconds: float,
    hop: float,
    sample_rate: int,
    suffix: str | None = None,
    fold: str = "fold_0",
) -> dict[str, Any]:
    by_path = {r["path"]: r for r in manifest["rows"]}

    assignment: dict[str, str] = {}
    if splits is not None:
        if fold not in splits["folds"]:
            raise SegmentError(f"{fold} is not in {sorted(splits['folds'])}")
        for name, rows in splits["folds"][fold].items():
            for row in rows:
                assignment[row["path"]] = name

    frame = int(round(seconds * sample_rate))
    stride = int(round(hop * sample_rate))

    entries: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    for path, row in sorted(by_path.items()):
        count = windows_for(row["duration"], seconds, hop)
        if count == 0:
            skipped.append({"path": path, "class": row["class"], "duration": row["duration"]})
            continue
        split = assignment.get(path)
        if splits is not None and split is None:
            continue
        stored = Path(path).with_suffix(suffix).as_posix() if suffix else path
        for index in range(count):
            entries.append(
                {
                    "path": stored,
                    "start": index * stride,
                    "frames": frame,
                    "class": row["class"],
                    "recording_id": row["recording_id"],
                    "split": split,
                }
            )

    per_class = Counter(e["class"] for e in entries)
    per_split = Counter(e["split"] for e in entries if e["split"])
    skipped_hours = sum(s["duration"] for s in skipped) / 3600

    return {
        "dataset": manifest["dataset"],
        "fold": fold if splits is not None else None,
        "seconds": seconds,
        "hop": hop,
        "sample_rate": sample_rate,
        "frame": frame,
        "n_windows": len(entries),
        "n_files_used": len({e["path"] for e in entries}),
        "n_files_skipped": len(skipped),
        "skipped_hours": round(skipped_hours, 4),
        "per_class": dict(sorted(per_class.items())),
        "per_split": dict(sorted(per_split.items())),
        "skipped": skipped,
        "windows": entries,
    }


def write(index: dict[str, Any], path: str | Path) -> Path:
    out = resolve(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(index, indent=2), encoding="utf-8")
    return out


def render(index: dict[str, Any]) -> str:
    lines = [
        "dataset        " + index["dataset"],
        f"window         {index['seconds']} s, hop {index['hop']} s "
        f"({index['frame']} frames @ {index['sample_rate']} Hz)",
        f"fold           {index['fold']}",
        f"windows        {index['n_windows']}",
        f"files used     {index['n_files_used']}",
        f"files skipped  {index['n_files_skipped']} "
        f"({index['skipped_hours']} h too short for one window)",
        "per class      " + str(index["per_class"]),
        "per split      " + str(index["per_split"]),
    ]
    return "\n".join(lines)
