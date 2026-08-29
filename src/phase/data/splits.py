from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.model_selection import StratifiedGroupKFold

from phase.config import REPO_ROOT

MIN_GROUPS_PER_CLASS = 3


class SplitError(RuntimeError):
    pass


def resolve(path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else REPO_ROOT / path


def load_manifest(path: str | Path) -> dict[str, Any]:
    return json.loads(resolve(path).read_text(encoding="utf-8"))


def check_groupable(manifest: dict[str, Any]) -> None:
    if not manifest["split_safe"]:
        raise SplitError(
            f"{manifest['dataset']} is marked split_safe: false and must never be split by "
            "recording; it is for physics validation and cross-dataset transfer only"
        )
    missing = [r["path"] for r in manifest["rows"] if not r["recording_id"]]
    if missing:
        raise SplitError(f"{len(missing)} rows have no recording_id, e.g. {missing[:3]}")

    per_class: dict[str, set[str]] = {}
    for row in manifest["rows"]:
        per_class.setdefault(row["class"], set()).add(row["recording_id"])
    thin = {c: len(g) for c, g in per_class.items() if len(g) < MIN_GROUPS_PER_CLASS}
    if thin:
        raise SplitError(
            f"classes with fewer than {MIN_GROUPS_PER_CLASS} recordings cannot be split "
            f"three ways: {thin}"
        )

    straddling = {
        rid
        for rid, labels in (
            (rid, {r["class"] for r in manifest["rows"] if r["recording_id"] == rid})
            for rid in {r["recording_id"] for r in manifest["rows"]}
        )
        if len(labels) > 1
    }
    if straddling:
        raise SplitError(
            f"{len(straddling)} recording_ids span more than one class, so they are not "
            f"unique identities: {sorted(straddling)[:3]}"
        )


def build(manifest: dict[str, Any], n_folds: int = 5, seed: int = 0) -> dict[str, Any]:
    check_groupable(manifest)

    rows = manifest["rows"]
    labels = np.array([r["class"] for r in rows])
    groups = np.array([r["recording_id"] for r in rows])
    indices = np.arange(len(rows))

    outer = StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    folds: dict[str, Any] = {}
    for fold, (train_val, test) in enumerate(outer.split(indices, labels, groups)):
        inner = StratifiedGroupKFold(n_splits=n_folds - 1, shuffle=True, random_state=seed + fold)
        train, val = next(iter(inner.split(train_val, labels[train_val], groups[train_val])))
        assignment = {
            "train": [rows[i] for i in train_val[train]],
            "val": [rows[i] for i in train_val[val]],
            "test": [rows[i] for i in test],
        }
        for name, subset in assignment.items():
            if not subset:
                raise SplitError(f"fold {fold} split {name} is empty")
        folds[f"fold_{fold}"] = {
            name: [
                {"path": r["path"], "class": r["class"], "recording_id": r["recording_id"]}
                for r in subset
            ]
            for name, subset in assignment.items()
        }

    return {
        "dataset": manifest["dataset"],
        "split_safe": manifest["split_safe"],
        "classes": manifest["classes"],
        "grouped_by": "recording_id",
        "n_folds": n_folds,
        "seed": seed,
        "folds": folds,
    }


def summarise(splits: dict[str, Any]) -> str:
    lines = [
        "dataset      " + splits["dataset"],
        f"folds        {splits['n_folds']} (seed {splits['seed']})",
        "grouped by   " + splits["grouped_by"],
        "",
    ]
    for fold_name, fold in splits["folds"].items():
        lines.append(fold_name)
        for name, rows in fold.items():
            per_class = Counter(r["class"] for r in rows)
            recordings = len({r["recording_id"] for r in rows})
            counts = " ".join(f"{c}={per_class[c]}" for c in sorted(per_class))
            lines.append(
                f"  {name:6s} {len(rows):>5d} files  {recordings:>4d} recordings   {counts}"
            )
    return "\n".join(lines)


def write(splits: dict[str, Any], path: str | Path, overwrite: bool = False) -> Path:
    out = resolve(path)
    if out.exists() and not overwrite:
        raise SplitError(
            f"{out} already exists; splits are immutable once written and must not be "
            "regenerated mid-project"
        )
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(splits, indent=2), encoding="utf-8")
    return out
