from __future__ import annotations

import argparse
import json
import re
import statistics
import sys
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import soundfile as sf

from phase.config import REPO_ROOT

AUDIO_SUFFIXES = {".wav", ".flac", ".mp3", ".ogg", ".m4a", ".aif", ".aiff"}

TOKEN_SPLIT = re.compile(r"[-_.\s]+")

MIN_GROUPS_PER_CLASS = 3

PUBLISHED_SPECS: dict[str, dict[str, Any]] = {
    "deepship": {
        "hours": 47.07,
        "recordings": 265,
        "classes": ["cargo", "passenger", "tanker", "tug"],
        "sample_rate": 32000,
        "citation": "Irfan et al. 2021, ESWA 183:115270",
    },
    "ds3500": {
        "files": 1948,
        "classes": ["A", "B", "C", "D", "E"],
        "class_counts": {"A": 345, "B": 235, "C": 785, "D": 395, "E": 188},
        "sample_rate": 16000,
        "seconds_per_file": 5.0,
        "citation": "DS3500, BELLHOP rendering of ShipsEar, 36 geometries",
    },
    "esc50": {
        "files": 2000,
        "sample_rate": 44100,
        "seconds_per_file": 5.0,
        "citation": "Piczak 2015, ESC-50",
    },
}


@dataclass
class GroupingCandidate:
    strategy: str
    n_groups: int
    class_pure: bool
    files_per_group_min: int
    files_per_group_median: float
    files_per_group_max: int
    seconds_per_group_median: float
    min_groups_per_class: int
    splittable: bool
    examples: list[str] = field(default_factory=list)


def iter_audio(root: Path) -> list[Path]:
    return sorted(p for p in root.rglob("*") if p.suffix.lower() in AUDIO_SUFFIXES)


def probe(path: Path) -> dict[str, Any] | None:
    try:
        info = sf.info(str(path))
    except Exception:
        return None
    return {
        "frames": info.frames,
        "sample_rate": info.samplerate,
        "channels": info.channels,
        "duration": info.frames / info.samplerate if info.samplerate else 0.0,
        "format": info.format,
        "subtype": info.subtype,
    }


def class_of(relative: Path, source: str = "top_dir") -> str:
    if source == "stem_token0":
        tokens = [t for t in TOKEN_SPLIT.split(relative.stem) if t]
        return tokens[0] if tokens else "_root"
    return relative.parts[0] if len(relative.parts) > 1 else "_root"


def _strategies(relative: Path, cls: str) -> dict[str, str]:
    parts = relative.parts
    stem = relative.stem
    tokens = [t for t in TOKEN_SPLIT.split(stem) if t]

    out = {
        "file_stem": str(relative.with_suffix("")),
        "parent_dir": str(relative.parent) if relative.parent != Path(".") else "_root",
    }
    for depth in range(1, min(len(parts), 4)):
        out[f"dir_level_{depth}"] = "/".join(parts[:depth])
    if len(tokens) > 1:
        out["stem_minus_last_token"] = cls + "/" + "_".join(tokens[:-1])
    if len(tokens) > 2:
        out["stem_first_two_tokens"] = cls + "/" + "_".join(tokens[:2])
    if tokens:
        out["stem_first_token"] = cls + "/" + tokens[0]
    return out


def analyse_grouping(rows: list[dict[str, Any]]) -> tuple[list[GroupingCandidate], dict[str, Any]]:
    by_strategy: dict[str, dict[str, list[dict[str, Any]]]] = defaultdict(lambda: defaultdict(list))
    for row in rows:
        for strategy, key in _strategies(Path(row["relative"]), row["class"]).items():
            by_strategy[strategy][key].append(row)

    candidates: list[GroupingCandidate] = []
    for strategy, groups in by_strategy.items():
        sizes = [len(v) for v in groups.values()]
        seconds = [sum(r["duration"] for r in v) for v in groups.values()]
        class_pure = all(len({r["class"] for r in v}) == 1 for v in groups.values())
        groups_per_class: Counter[str] = Counter()
        for members in groups.values():
            groups_per_class[members[0]["class"]] += 1
        fewest = min(groups_per_class.values())
        candidates.append(
            GroupingCandidate(
                strategy=strategy,
                n_groups=len(groups),
                class_pure=class_pure,
                files_per_group_min=min(sizes),
                files_per_group_median=statistics.median(sizes),
                files_per_group_max=max(sizes),
                seconds_per_group_median=round(statistics.median(seconds), 2),
                min_groups_per_class=fewest,
                splittable=class_pure and fewest >= MIN_GROUPS_PER_CLASS,
                examples=sorted(groups)[:5],
            )
        )

    candidates.sort(key=lambda c: c.n_groups)
    n_files = len(rows)
    n_classes = len({r["class"] for r in rows})
    median_duration = statistics.median(r["duration"] for r in rows)

    grouped = [
        c
        for c in candidates
        if c.class_pure and n_classes < c.n_groups < n_files and c.files_per_group_median > 1
    ]
    usable = [c for c in grouped if c.splittable]

    evidence = []
    if usable:
        best = max(usable, key=lambda c: c.files_per_group_median)
        verdict = "recoverable_from_structure"
        recommended = best.strategy
        evidence.append(
            f"{best.strategy} yields {best.n_groups} class-pure groups over {n_files} files, "
            f"median {best.files_per_group_median} files and "
            f"{best.seconds_per_group_median} s per group"
        )
    elif grouped:
        best = max(grouped, key=lambda c: c.min_groups_per_class)
        verdict = "too_coarse_to_split"
        recommended = None
        evidence.append(
            f"{best.strategy} yields {best.n_groups} class-pure groups, but the thinnest class "
            f"has only {best.min_groups_per_class} of them, fewer than the "
            f"{MIN_GROUPS_PER_CLASS} needed to place a group in each of train, val and test"
        )
    elif median_duration >= 60.0:
        verdict = "per_transit_files"
        recommended = "file_stem"
        evidence.append(
            f"no grouping structure above the file level, but median file duration is "
            f"{median_duration:.1f} s, long enough that each file is plausibly a whole transit"
        )
    else:
        verdict = "not_recoverable"
        recommended = None
        evidence.append(
            f"no class-pure grouping above the file level and median file duration is only "
            f"{median_duration:.1f} s, so individual recordings cannot be reconstructed"
        )

    summary: dict[str, Any] = {
        "verdict": verdict,
        "recommended_recording_id": recommended,
        "evidence": evidence,
        "median_file_duration_s": round(median_duration, 3),
    }
    return candidates, summary


def compare_to_spec(name: str, report: dict[str, Any]) -> dict[str, Any] | None:
    spec = PUBLISHED_SPECS.get(name)
    if spec is None:
        return None

    checks: list[dict[str, Any]] = []

    def check(label: str, expected: Any, observed: Any, ok: bool) -> None:
        checks.append({"field": label, "expected": expected, "observed": observed, "ok": bool(ok)})

    if "files" in spec:
        check("files", spec["files"], report["n_files"], report["n_files"] == spec["files"])
    if "hours" in spec:
        check(
            "hours",
            spec["hours"],
            report["total_hours"],
            report["total_hours"] >= 0.9 * spec["hours"],
        )
    if "classes" in spec:
        observed = sorted(report["per_class"])
        expected = sorted(spec["classes"])
        lowered_ok = [c.lower() for c in observed] == [c.lower() for c in expected]
        check("classes", expected, observed, lowered_ok)
    if "class_counts" in spec:
        observed = {k: v["files"] for k, v in report["per_class"].items()}
        check("class_counts", spec["class_counts"], observed, observed == spec["class_counts"])
    if "sample_rate" in spec:
        rates = report["sample_rates"]
        observed = max(rates, key=rates.get) if rates else None
        check(
            "sample_rate", spec["sample_rate"], observed, str(spec["sample_rate"]) == str(observed)
        )
    if "recordings" in spec:
        observed = report["recording_identity"]["n_recordings_if_recommended"]
        check(
            "recordings",
            spec["recordings"],
            observed,
            observed is not None and observed >= 0.5 * spec["recordings"],
        )

    return {
        "citation": spec["citation"],
        "checks": checks,
        "n_passed": sum(c["ok"] for c in checks),
        "n_checks": len(checks),
    }


def audit(
    root: Path,
    name: str | None = None,
    max_files: int | None = None,
    class_from: str = "top_dir",
) -> dict[str, Any]:
    root = root if root.is_absolute() else REPO_ROOT / root
    if not root.is_dir():
        raise SystemExit(f"no such directory: {root}")

    name = name or root.name.lower()
    paths = iter_audio(root)
    if max_files:
        paths = paths[:max_files]
    if not paths:
        raise SystemExit(f"no audio files under {root}")

    rows: list[dict[str, Any]] = []
    unreadable: list[str] = []
    for index, path in enumerate(paths, 1):
        info = probe(path)
        relative = path.relative_to(root)
        if info is None:
            unreadable.append(relative.as_posix())
            continue
        rows.append(
            {
                "relative": relative.as_posix(),
                "class": class_of(relative, class_from),
                "depth": len(relative.parts),
                **info,
            }
        )
        if index % 1000 == 0:
            print(f"  probed {index}/{len(paths)}", file=sys.stderr)

    if not rows:
        raise SystemExit(f"every audio file under {root} was unreadable")

    durations = [r["duration"] for r in rows]
    per_class: dict[str, Any] = {}
    for cls in sorted({r["class"] for r in rows}):
        group = [r for r in rows if r["class"] == cls]
        per_class[cls] = {
            "files": len(group),
            "hours": round(sum(r["duration"] for r in group) / 3600, 4),
            "median_duration_s": round(statistics.median(r["duration"] for r in group), 3),
        }

    report: dict[str, Any] = {
        "dataset": name,
        "root": Path(root).as_posix(),
        "class_from": class_from,
        "n_files": len(rows),
        "n_unreadable": len(unreadable),
        "unreadable_examples": unreadable[:10],
        "total_hours": round(sum(durations) / 3600, 4),
        "duration_s": {
            "min": round(min(durations), 3),
            "median": round(statistics.median(durations), 3),
            "mean": round(statistics.fmean(durations), 3),
            "max": round(max(durations), 3),
        },
        "sample_rates": dict(Counter(str(r["sample_rate"]) for r in rows).most_common()),
        "channels": dict(Counter(str(r["channels"]) for r in rows).most_common()),
        "formats": dict(Counter(r["format"] + "/" + str(r["subtype"]) for r in rows).most_common()),
        "depth": dict(Counter(str(r["depth"]) for r in rows).most_common()),
        "max_depth": max(r["depth"] for r in rows),
        "per_class": per_class,
        "example_paths": [r["relative"] for r in rows[:15]],
    }

    candidates, summary = analyse_grouping(rows)
    recommended = summary["recommended_recording_id"]
    summary["n_recordings_if_recommended"] = next(
        (c.n_groups for c in candidates if c.strategy == recommended), None
    )
    summary["candidates"] = [asdict(c) for c in candidates]
    report["recording_identity"] = summary

    spec = compare_to_spec(name, report)
    if spec is not None:
        report["spec_comparison"] = spec

    return report


def render(report: dict[str, Any]) -> str:
    duration = report["duration_s"]
    lines = [
        "dataset          " + str(report["dataset"]),
        "root             " + str(report["root"]),
        f"files            {report['n_files']} ({report['n_unreadable']} unreadable)",
        f"total hours      {report['total_hours']}",
        f"duration s       min {duration['min']} / median {duration['median']} / max {duration['max']}",
        "sample rates     " + str(report["sample_rates"]),
        "channels         " + str(report["channels"]),
        f"directory depth  {report['depth']} (max {report['max_depth']})",
        "",
        "per class",
    ]
    for cls, stats in report["per_class"].items():
        lines.append(
            f"  {cls:24s} {stats['files']:>7d} files  {stats['hours']:>9.3f} h  "
            f"median {stats['median_duration_s']:>8.2f} s"
        )

    identity = report["recording_identity"]
    lines += [
        "",
        "recording identity",
        "  verdict        " + str(identity["verdict"]),
        "  recommended    " + str(identity["recommended_recording_id"]),
        "  n recordings   " + str(identity["n_recordings_if_recommended"]),
    ]
    for note in identity["evidence"]:
        lines.append("  evidence       " + note)
    lines.append("  candidates")
    for candidate in identity["candidates"]:
        lines.append(
            f"    {candidate['strategy']:26s} groups {candidate['n_groups']:>7d}  "
            f"pure {str(candidate['class_pure']):5s}  "
            f"files/group med {candidate['files_per_group_median']:>7.1f}  "
            f"min/class {candidate['min_groups_per_class']:>5d}  "
            f"splittable {str(candidate['splittable'])}"
        )

    if "spec_comparison" in report:
        spec = report["spec_comparison"]
        lines += ["", "published spec (" + spec["citation"] + ")"]
        for check in spec["checks"]:
            mark = "ok  " if check["ok"] else "MISS"
            lines.append(f"  {mark} {check['field']:14s} expected {check['expected']}")
            lines.append(f"       {'':14s} observed {check['observed']}")
        lines.append(f"  {spec['n_passed']}/{spec['n_checks']} checks passed")

    lines += ["", "example paths"]
    lines += ["  " + p for p in report["example_paths"]]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Audit a raw dataset directory")
    parser.add_argument("--root", required=True)
    parser.add_argument("--report", default=None)
    parser.add_argument("--name", default=None)
    parser.add_argument("--max-files", type=int, default=None)
    parser.add_argument("--class-from", default="top_dir", choices=["top_dir", "stem_token0"])
    args = parser.parse_args(argv)

    report = audit(
        Path(args.root),
        name=args.name,
        max_files=args.max_files,
        class_from=args.class_from,
    )
    print(render(report))

    if args.report:
        out = Path(args.report)
        out = out if out.is_absolute() else REPO_ROOT / out
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print("\nwrote " + out.as_posix())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
