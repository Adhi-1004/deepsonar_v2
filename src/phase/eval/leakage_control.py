from __future__ import annotations

from typing import Any

import numpy as np

from phase.config import OptimConfig
from phase.eval.linear_probe import evaluate_linear_probe
from phase.seed import set_seed


def shuffled_labels_meta(meta: dict[str, Any], seed: int = 0) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    recording_to_indices: dict[str, list[int]] = {}
    for idx, rec_id in enumerate(meta["recording_ids"]):
        recording_to_indices.setdefault(rec_id, []).append(idx)

    unique_recordings = list(recording_to_indices.keys())
    shuffled_recordings = unique_recordings.copy()
    rng.shuffle(shuffled_recordings)

    rec_to_class = {
        rec: meta["labels"][indices[0]] for rec, indices in recording_to_indices.items()
    }
    permuted_rec_to_class = {
        orig: rec_to_class[perm]
        for orig, perm in zip(unique_recordings, shuffled_recordings, strict=True)
    }

    new_labels = [permuted_rec_to_class[rec] for rec in meta["recording_ids"]]
    return {**meta, "labels": new_labels}


def run_leakage_control(
    features: np.ndarray,
    meta: dict[str, Any],
    optim_config: OptimConfig,
    checkpoint_path: str | None = None,
    seeds: list[int] | None = None,
    backbone: str = "resnet18",
    device: str = "cpu",
) -> dict[str, Any]:
    if seeds is None:
        seeds = [0, 1, 2]

    control_runs = []
    for seed in seeds:
        set_seed(seed)
        shuffled_meta = shuffled_labels_meta(meta, seed=seed)
        result = evaluate_linear_probe(
            features,
            shuffled_meta,
            optim_config,
            checkpoint_path=checkpoint_path,
            backbone=backbone,
            device=device,
        )
        control_runs.append({
            "seed": seed,
            "accuracy": result["accuracy"],
            "macro_f1": result["macro_f1"],
        })

    accuracies = [r["accuracy"] for r in control_runs]
    return {
        "n_seeds": len(seeds),
        "mean_accuracy": float(np.mean(accuracies)),
        "std_accuracy": float(np.std(accuracies)),
        "chance_level": 1.0 / len(set(meta["labels"])),
        "runs": control_runs,
    }

