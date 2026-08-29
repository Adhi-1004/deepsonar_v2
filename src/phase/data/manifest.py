from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

import soundfile as sf

from phase.config import REPO_ROOT, DatasetConfig

AUDIO_SUFFIXES = {".wav", ".flac", ".mp3", ".ogg", ".m4a", ".aif", ".aiff"}

TRANSIT_DIR = re.compile(r"^(\d{8})([a-z]*)-(\d+)$")

ESC50_NAME = re.compile(r"^(\d+)-(\d+)-([A-Z])-(\d+)$")

DEEPSHIP_METAFILES = {
    "Cargo": [
        "data/raw/deepship/deepship-raw/Cargo/cargo-metafile",
        "data/raw/deepship_subset_vasundharauppuluri/DeepShip-main/Cargo/cargo-metafile",
    ],
    "Passenger": [
        "data/raw/deepship/deepship-raw/Passenger/passengership-metafile",
        "data/raw/deepship_subset_vasundharauppuluri/DeepShip-main/Passengership/passengership-metafile",
    ],
    "Tanker": [
        "data/raw/deepship_subset_vasundharauppuluri/DeepShip-main/Tanker/tanker-metafile",
    ],
    "Tug": [
        "data/raw/deepship_subset_vasundharauppuluri/DeepShip-main/Tug/tug-metafile",
    ],
}


def resolve(path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else REPO_ROOT / path


def load_metafile(
    paths: list[str],
) -> tuple[dict[tuple[str, str], str], dict[tuple[str, int], str]]:
    by_datetime: dict[tuple[str, str], str] = {}
    by_index: dict[tuple[str, int], str] = {}
    for candidate in paths:
        path = resolve(candidate)
        if not path.is_file():
            continue
        for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
            fields = [f.strip() for f in line.split(",")]
            if len(fields) < 6 or not fields[0].isdigit():
                continue
            index, ship, date, time = int(fields[0]), fields[2], fields[3], fields[4]
            if not ship:
                continue
            by_datetime.setdefault((date, time), ship)
            by_index.setdefault((date, index), ship)
    return by_datetime, by_index


def deepship_ship_names(root: Path) -> tuple[dict[str, str], dict[str, str]]:
    ships: dict[str, str] = {}
    how: dict[str, str] = {}
    for cls, metafiles in DEEPSHIP_METAFILES.items():
        class_dir = root / cls
        if not class_dir.is_dir():
            continue
        by_datetime, by_index = load_metafile(metafiles)
        for transit in sorted(d for d in class_dir.iterdir() if d.is_dir()):
            match = TRANSIT_DIR.match(transit.name)
            audio = sorted(p for p in transit.iterdir() if p.suffix.lower() in AUDIO_SUFFIXES)
            if not match or not audio:
                continue
            date, _, index = match.groups()
            key = f"{cls}/{transit.name}"
            ship = by_datetime.get((date, audio[0].stem))
            if ship:
                how[key] = "date_time"
            else:
                ship = by_index.get((date, int(index)))
                if ship:
                    how[key] = "date_index"
            if ship:
                ships[key] = f"{cls}/{ship}"
    return ships, how


def probe(path: Path) -> dict[str, Any] | None:
    try:
        info = sf.info(str(path))
    except Exception:
        return None
    return {
        "duration": info.frames / info.samplerate if info.samplerate else 0.0,
        "sample_rate": info.samplerate,
        "channels": info.channels,
    }


def build(config: DatasetConfig, root: Path | None = None) -> dict[str, Any]:
    root = resolve(root or config.root)
    if not root.is_dir():
        raise SystemExit(f"no such directory: {root}")

    ships: dict[str, str] = {}
    how: dict[str, str] = {}
    if config.name == "deepship":
        ships, how = deepship_ship_names(root)

    rows: list[dict[str, Any]] = []
    unreadable: list[str] = []
    for path in sorted(p for p in root.rglob("*") if p.suffix.lower() in AUDIO_SUFFIXES):
        relative = path.relative_to(root)
        info = probe(path)
        if info is None:
            unreadable.append(relative.as_posix())
            continue

        parts = relative.parts
        label = parts[0] if len(parts) > 1 else "_root"
        transit = f"{label}/{parts[1]}" if len(parts) > 2 else relative.with_suffix("").as_posix()

        if config.name == "esc50":
            fold, clip_id, _, target = ESC50_NAME.match(path.stem).groups()
            label = target
            transit = f"fold{fold}/clip{clip_id}"

        if not config.split_safe:
            recording_id: str | None = None
            grouping = "none"
        elif config.name == "deepship":
            recording_id = ships.get(transit, transit)
            grouping = how.get(transit, "transit_fallback")
        else:
            recording_id = transit
            grouping = "path"

        rows.append(
            {
                "path": relative.as_posix(),
                "dataset": config.name,
                "class": label,
                "recording_id": recording_id,
                "transit": transit,
                "grouping": grouping,
                "duration": round(info["duration"], 4),
                "sample_rate": info["sample_rate"],
                "channels": info["channels"],
            }
        )

    if not rows:
        raise SystemExit(f"no audio under {root}")

    grouping_counts = Counter(r["grouping"] for r in rows)
    per_class = {}
    for cls in sorted({r["class"] for r in rows}):
        group = [r for r in rows if r["class"] == cls]
        per_class[cls] = {
            "files": len(group),
            "hours": round(sum(r["duration"] for r in group) / 3600, 4),
            "recordings": len({r["recording_id"] for r in group if r["recording_id"]}),
            "transits": len({r["transit"] for r in group}),
        }

    return {
        "dataset": config.name,
        "root": root.as_posix(),
        "split_safe": config.split_safe,
        "classes": sorted({r["class"] for r in rows}),
        "n_files": len(rows),
        "n_unreadable": len(unreadable),
        "total_hours": round(sum(r["duration"] for r in rows) / 3600, 4),
        "n_recordings": len({r["recording_id"] for r in rows if r["recording_id"]}),
        "grouping_source": dict(grouping_counts),
        "per_class": per_class,
        "rows": rows,
    }


def write(manifest: dict[str, Any], path: str | Path) -> Path:
    out = resolve(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return out


def render(manifest: dict[str, Any]) -> str:
    lines = [
        "dataset        " + manifest["dataset"],
        "root           " + manifest["root"],
        f"split safe     {manifest['split_safe']}",
        f"files          {manifest['n_files']} ({manifest['n_unreadable']} unreadable)",
        f"total hours    {manifest['total_hours']}",
        f"recordings     {manifest['n_recordings']}",
        "grouping       " + str(manifest["grouping_source"]),
        "",
        "per class",
    ]
    for cls, stats in manifest["per_class"].items():
        lines.append(
            f"  {cls:16s} {stats['files']:>5d} files  {stats['hours']:>8.3f} h  "
            f"{stats['recordings']:>4d} recordings  {stats['transits']:>4d} transits"
        )
    return "\n".join(lines)
