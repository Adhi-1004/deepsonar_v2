from __future__ import annotations

from typing import Any

import numpy as np

from phase.config import OptimConfig
from phase.eval.linear_probe import evaluate_linear_probe
from phase.eval.metrics import aggregate
from phase.seed import set_seed


def subsample_train_indices(
    meta: dict[str, Any], fraction: float, seed: int
) -> list[int]:
    rng = np.random.default_rng(seed)
    train_indices = [i for i, s in enumerate(meta["splits"]) if s == "train"]
    if fraction >= 1.0 or not train_indices:
        return train_indices

    rec_to_indices: dict[str, list[int]] = {}
    rec_to_class: dict[str, str] = {}
    for idx in train_indices:
        rec_id = meta["recording_ids"][idx]
        cls = meta["labels"][idx]
        rec_to_indices.setdefault(rec_id, []).append(idx)
        rec_to_class[rec_id] = cls

    classes = sorted({meta["labels"][i] for i in train_indices})
    selected_indices: list[int] = []
    target_count = max(len(classes), int(round(len(train_indices) * fraction)))

    # Ensure every class has at least one recording
    for cls in classes:
        cls_recs = [r for r, c in rec_to_class.items() if c == cls]
        chosen = rng.choice(cls_recs)
        selected_indices.extend(rec_to_indices[chosen])

    # Add remaining recordings until target count is met
    remaining_recs = [r for r in rec_to_class if r not in {meta["recording_ids"][i] for i in selected_indices}]
    rng.shuffle(remaining_recs)
    for rec in remaining_recs:
        if len(selected_indices) >= target_count:
            break
        selected_indices.extend(rec_to_indices[rec])

    return sorted(selected_indices)


def run_label_efficiency(
    features: np.ndarray,
    meta: dict[str, Any],
    optim_config: OptimConfig,
    checkpoint_path: str | None = None,
    pretrained_imagenet: bool = False,
    fractions: list[float] | None = None,
    seeds: list[int] | None = None,
    backbone: str = "resnet18",
    device: str = "cpu",
) -> dict[str, Any]:
    if fractions is None:
        fractions = [0.01, 0.05, 0.1, 0.25, 0.5, 1.0]
    if seeds is None:
        seeds = [0, 1, 2]

    results_by_fraction: dict[str, Any] = {}

    for frac in fractions:
        seed_runs = []
        for seed in seeds:
            set_seed(seed)
            sub_train = set(subsample_train_indices(meta, frac, seed))

            # Build modified splits: retain val and test, filter train
            modified_splits = []
            for i, split in enumerate(meta["splits"]):
                if split == "train":
                    modified_splits.append("train" if i in sub_train else "unused")
                else:
                    modified_splits.append(split)

            sub_meta = {**meta, "splits": modified_splits}

            run_result = evaluate_linear_probe(
                features,
                sub_meta,
                optim_config,
                checkpoint_path=checkpoint_path,
                pretrained_imagenet=pretrained_imagenet,
                backbone=backbone,
                device=device,
            )
            seed_runs.append({
                "accuracy": run_result["accuracy"],
                "macro_f1": run_result["macro_f1"],
                "best_val_accuracy": run_result["best_val_accuracy"],
            })

        summary = aggregate(seed_runs)
        results_by_fraction[str(frac)] = {
            "fraction": frac,
            "runs": seed_runs,
            "summary": summary,
        }

    return {
        "fractions": fractions,
        "seeds": seeds,
        "results": results_by_fraction,
    }

