"""Recording-ID shuffle control for leakage detection (Phase 7).

If splits were truly leaking recording identities, shuffling the
recording-ID assignment would *not* decrease accuracy – the model
would be exploiting within-recording redundancy rather than
cross-recording generalisation.

This module re-runs a linear probe after randomly permuting the
``recording_id`` labels (while keeping class labels fixed), so the
training set nominally contains the same ships as the test set by
chance. Any drop versus the real split is evidence that our
grouped splits are actually protecting against leakage.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from phase.config import OptimConfig
from phase.eval.linear_probe import train_linear_probe
from phase.eval.metrics import aggregate
from phase.data.manifest import resolve


def shuffle_recording_ids(meta: dict[str, Any], seed: int = 0) -> dict[str, Any]:
    """Return a copy of meta with recording_ids randomly permuted within each class.

    The permutation is within-class so class balance per split is preserved, but
    the mapping from recording → split assignment is randomised.  This simulates
    a 'naive' split that ignores recording identity.

    If ``meta`` has no ``recording_ids`` the meta is returned unchanged (the
    control is vacuous without grouping information).
    """
    if not meta.get("recording_ids"):
        return meta

    rng = np.random.default_rng(seed)
    labels = meta["labels"]
    splits = meta["splits"]
    rids = list(meta["recording_ids"])

    classes = sorted(set(labels))
    new_rids = list(rids)
    for cls in classes:
        cls_indices = [i for i, l in enumerate(labels) if l == cls]
        cls_rids = [rids[i] for i in cls_indices]
        shuffled = rng.permutation(cls_rids).tolist()
        for i, idx in enumerate(cls_indices):
            new_rids[idx] = shuffled[i]

    out = dict(meta)
    out["recording_ids"] = new_rids
    return out


def run_leakage_control(
    embeddings: np.ndarray,
    meta: dict[str, Any],
    optim_config: OptimConfig,
    seeds: list[int],
    n_permutations: int = 3,
    device: str = "cpu",
) -> dict[str, Any]:
    """Run the leakage-control experiment.

    For each seed × permutation, shuffles recording IDs and trains a linear
    probe.  Compares resulting accuracy to the unshuffled baseline from the
    same seed.

    Parameters
    ----------
    n_permutations:
        How many independent shuffle seeds to run per evaluation seed.

    Returns
    -------
    dict with keys ``baseline``, ``shuffled``, ``delta_summary``.
    """
    # --- Baseline (true grouped split) ---
    baseline_rows: list[dict[str, float]] = []
    for seed in seeds:
        result = train_linear_probe(embeddings, meta, optim_config, seed=seed, device=device)
        scores = result.get("test", result["best_val"])
        baseline_rows.append({k: v for k, v in scores.items() if isinstance(v, float)})
        print(f"  [baseline] seed={seed}" + "".join(f"  {k}={v:.4f}" for k, v in baseline_rows[-1].items()))

    # --- Shuffled control ---
    shuffled_rows: list[dict[str, float]] = []
    for seed in seeds:
        for perm in range(n_permutations):
            shuffled_meta = shuffle_recording_ids(meta, seed=seed * 100 + perm)
            result = train_linear_probe(
                embeddings, shuffled_meta, optim_config, seed=seed, device=device
            )
            scores = result.get("test", result["best_val"])
            shuffled_rows.append({k: v for k, v in scores.items() if isinstance(v, float)})
            print(
                f"  [shuffled] seed={seed} perm={perm}"
                + "".join(f"  {k}={v:.4f}" for k, v in shuffled_rows[-1].items())
            )

    baseline_agg = aggregate(baseline_rows)
    shuffled_agg = aggregate(shuffled_rows)

    # Delta: positive means real split is better (leakage protection helps)
    delta: dict[str, dict[str, float]] = {}
    for metric in baseline_agg:
        if metric in shuffled_agg:
            delta[metric] = {
                "real_mean": baseline_agg[metric]["mean"],
                "shuffled_mean": shuffled_agg[metric]["mean"],
                "delta": baseline_agg[metric]["mean"] - shuffled_agg[metric]["mean"],
            }

    return {
        "protocol": "leakage_control",
        "seeds": seeds,
        "n_permutations": n_permutations,
        "baseline_rows": baseline_rows,
        "shuffled_rows": shuffled_rows,
        "baseline_summary": baseline_agg,
        "shuffled_summary": shuffled_agg,
        "delta": delta,
    }


def write(result: dict[str, Any], path: str | Path) -> Path:
    out = resolve(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return out

