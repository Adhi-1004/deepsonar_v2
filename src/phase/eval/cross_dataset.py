"""Cross-dataset transfer evaluation (Phase 7).

Trains a linear probe on one dataset's embeddings and evaluates on another's.
The claim is that PHASE degrades less than generic-augmentation MoCo.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from phase.config import OptimConfig
from phase.eval.metrics import aggregate, summarise
from phase.data.manifest import resolve


@torch.no_grad()
def _evaluate(model: nn.Module, loader: DataLoader, device: str) -> dict[str, float]:
    model.eval()
    preds, truths = [], []
    for inputs, labels in loader:
        preds.append(model(inputs.to(device)).argmax(dim=1).cpu().numpy())
        truths.append(labels.numpy())
    return summarise(np.concatenate(truths), np.concatenate(preds))


def _make_loader(
    embeddings: np.ndarray, labels: list[str], lookup: dict[str, int], batch_size: int
) -> DataLoader:
    targets = np.array([lookup.get(l, -1) for l in labels], dtype=np.int64)
    valid = targets >= 0
    ds = TensorDataset(
        torch.from_numpy(embeddings[valid]).float(),
        torch.from_numpy(targets[valid]),
    )
    return DataLoader(ds, batch_size=batch_size, shuffle=False)


def train_cross_dataset(
    source_embeddings: np.ndarray,
    source_meta: dict[str, Any],
    target_embeddings: np.ndarray,
    target_meta: dict[str, Any],
    optim_config: OptimConfig,
    seed: int = 0,
    device: str = "cpu",
    log: Any = None,
) -> dict[str, Any]:
    """Train on source, evaluate on target.

    Uses only the source training split to fit the linear head.
    Evaluates on **all** target data (no split assumed – cross-dataset
    means we don't have held-out recordings from the target domain).

    Parameters
    ----------
    source_embeddings / source_meta:
        Pre-extracted embeddings for the training dataset.
    target_embeddings / target_meta:
        Pre-extracted embeddings for the target (unseen) dataset.
        ``target_meta["labels"]`` must use the same class names as ``source_meta``.
    """
    # Build class mapping from source classes
    classes = sorted(set(source_meta["labels"]))
    lookup = {name: i for i, name in enumerate(classes)}
    source_splits = np.array(source_meta["splits"])
    source_targets = np.array([lookup[l] for l in source_meta["labels"]], dtype=np.int64)

    # Only train on source training split
    train_mask = np.flatnonzero(source_splits == "train")
    train_ds = TensorDataset(
        torch.from_numpy(np.ascontiguousarray(source_embeddings[train_mask])).float(),
        torch.from_numpy(source_targets[train_mask]),
    )
    train_loader = DataLoader(
        train_ds, batch_size=optim_config.batch_size, shuffle=True, drop_last=False
    )

    dim = source_embeddings.shape[1]
    head = nn.Linear(dim, len(classes)).to(device)

    if optim_config.optimizer == "adamw":
        optimiser = torch.optim.AdamW(
            head.parameters(), lr=optim_config.lr, weight_decay=optim_config.weight_decay
        )
    else:
        optimiser = torch.optim.SGD(
            head.parameters(), lr=optim_config.lr, momentum=optim_config.momentum,
            weight_decay=optim_config.weight_decay,
        )
    criterion = nn.CrossEntropyLoss()

    for epoch in range(optim_config.epochs):
        lr = optim_config.lr * 0.5 * (
            1 + __import__("math").cos(
                __import__("math").pi * max(0, epoch - optim_config.warmup_epochs)
                / max(1, optim_config.epochs - optim_config.warmup_epochs)
            )
        ) if epoch >= optim_config.warmup_epochs else (
            optim_config.lr * (epoch + 1) / max(1, optim_config.warmup_epochs)
        )
        for group in optimiser.param_groups:
            group["lr"] = lr

        head.train()
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimiser.zero_grad(set_to_none=True)
            criterion(head(inputs), targets).backward()
            optimiser.step()

        if log is not None:
            log({"epoch": epoch, "lr": lr, "seed": seed})

    # Evaluate on target
    target_loader = _make_loader(
        target_embeddings, target_meta["labels"], lookup, optim_config.batch_size
    )
    target_scores = _evaluate(head, target_loader, device)

    # Also evaluate on source test split for reference
    source_test_mask = np.flatnonzero(source_splits == "test")
    result: dict[str, Any] = {
        "classes": classes,
        "seed": seed,
        "target_scores": target_scores,
    }
    if source_test_mask.size > 0:
        source_test_ds = TensorDataset(
            torch.from_numpy(np.ascontiguousarray(source_embeddings[source_test_mask])).float(),
            torch.from_numpy(source_targets[source_test_mask]),
        )
        source_test_loader = DataLoader(
            source_test_ds, batch_size=optim_config.batch_size, shuffle=False
        )
        result["source_test_scores"] = _evaluate(head, source_test_loader, device)

    return result


def run_cross_dataset(
    source_embeddings: np.ndarray,
    source_meta: dict[str, Any],
    target_embeddings: np.ndarray,
    target_meta: dict[str, Any],
    seeds: list[int],
    optim_config: OptimConfig,
    device: str = "cpu",
    log: Any = None,
) -> dict[str, Any]:
    """Multi-seed cross-dataset sweep."""
    runs: list[dict[str, Any]] = []
    for seed in seeds:
        result = train_cross_dataset(
            source_embeddings, source_meta, target_embeddings, target_meta,
            optim_config, seed=seed, device=device, log=log,
        )
        runs.append(result)
        print(
            f"  seed={seed}  target:"
            + "".join(f"  {k}={v:.4f}" for k, v in result["target_scores"].items())
        )

    target_rows = [r["target_scores"] for r in runs]
    source_rows = [r.get("source_test_scores", {}) for r in runs if "source_test_scores" in r]

    return {
        "protocol": "cross_dataset",
        "seeds": seeds,
        "runs": runs,
        "target_summary": aggregate(target_rows),
        "source_summary": aggregate(source_rows) if source_rows else {},
    }


def write(result: dict[str, Any], path: str | Path) -> Path:
    out = resolve(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return out

