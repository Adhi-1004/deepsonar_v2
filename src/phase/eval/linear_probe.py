"""Linear probe evaluation on frozen encoder embeddings (Phase 7).

Fits a single linear layer on top of frozen MoCo encoder embeddings.
Cheaper than full fine-tune and a standard SSL quality signal.
"""
from __future__ import annotations

import math
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from phase.config import OptimConfig
from phase.eval.metrics import summarise


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------


def _make_loaders(
    embeddings: np.ndarray,
    meta: dict[str, Any],
    batch_size: int,
    label_fraction: float = 1.0,
    seed: int = 0,
) -> tuple[dict[str, DataLoader], list[str]]:
    classes = sorted(set(meta["labels"]))
    lookup = {name: i for i, name in enumerate(classes)}
    targets = np.array([lookup[n] for n in meta["labels"]], dtype=np.int64)
    splits = np.array(meta["splits"])

    train_mask = np.flatnonzero(splits == "train")
    if label_fraction < 1.0:
        rng = np.random.default_rng(seed)
        # Stratified sub-sample by class, grouped by recording if present
        keep: list[int] = []
        recording_ids = meta.get("recording_ids")
        for cls_idx in range(len(classes)):
            cls_mask = train_mask[targets[train_mask] == cls_idx]
            if recording_ids is not None:
                rids = np.array(recording_ids)
                unique_rids = list(dict.fromkeys(rids[cls_mask]))
                n_rids = max(1, round(label_fraction * len(unique_rids)))
                chosen_rids = set(rng.choice(unique_rids, n_rids, replace=False))
                keep.extend(int(i) for i in cls_mask if rids[i] in chosen_rids)
            else:
                n = max(1, round(label_fraction * len(cls_mask)))
                keep.extend(rng.choice(cls_mask, n, replace=False).tolist())
        train_mask = np.array(sorted(keep))

    loaders: dict[str, DataLoader] = {}
    for name in ("train", "val", "test"):
        if name == "train":
            mask = train_mask
        else:
            mask = np.flatnonzero(splits == name)
        if mask.size == 0:
            continue
        ds = TensorDataset(
            torch.from_numpy(np.ascontiguousarray(embeddings[mask])).float(),
            torch.from_numpy(targets[mask]),
        )
        loaders[name] = DataLoader(
            ds, batch_size=batch_size, shuffle=(name == "train"), drop_last=False
        )
    return loaders, classes


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------


def _cosine_lr(step: int, total: int, warmup: int, base: float) -> float:
    if step < warmup:
        return base * (step + 1) / max(1, warmup)
    progress = (step - warmup) / max(1, total - warmup)
    return base * 0.5 * (1 + math.cos(math.pi * progress))


@torch.no_grad()
def _evaluate(model: nn.Module, loader: DataLoader, device: str) -> dict[str, float]:
    model.eval()
    preds, truths = [], []
    for inputs, labels in loader:
        preds.append(model(inputs.to(device)).argmax(dim=1).cpu().numpy())
        truths.append(labels.numpy())
    return summarise(np.concatenate(truths), np.concatenate(preds))


def train_linear_probe(
    embeddings: np.ndarray,
    meta: dict[str, Any],
    optim_config: OptimConfig,
    label_fraction: float = 1.0,
    seed: int = 0,
    device: str = "cpu",
    log: Any = None,
) -> dict[str, Any]:
    """Train a single linear layer on frozen embeddings.

    Parameters
    ----------
    embeddings:
        Pre-extracted feature matrix ``(N, D)``.
    meta:
        Dict with ``labels`` (list[str]), ``splits`` (list[str]), and
        optionally ``recording_ids`` (list[str]) for grouped sub-sampling.
    label_fraction:
        Fraction of training recordings to use (label-efficiency curves).
    """
    loaders, classes = _make_loaders(
        embeddings, meta, optim_config.batch_size, label_fraction, seed
    )
    dim = embeddings.shape[1]
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

    history: list[dict[str, Any]] = []
    best: dict[str, Any] = {"val_accuracy": -1.0}

    for epoch in range(optim_config.epochs):
        lr = _cosine_lr(epoch, optim_config.epochs, optim_config.warmup_epochs, optim_config.lr)
        for group in optimiser.param_groups:
            group["lr"] = lr

        head.train()
        total_loss, seen = 0.0, 0
        for inputs, targets in loaders["train"]:
            inputs, targets = inputs.to(device), targets.to(device)
            optimiser.zero_grad(set_to_none=True)
            loss = criterion(head(inputs), targets)
            loss.backward()
            optimiser.step()
            total_loss += loss.item() * targets.size(0)
            seen += targets.size(0)

        record: dict[str, Any] = {
            "epoch": epoch, "lr": lr, "train_loss": total_loss / max(1, seen),
        }
        if "val" in loaders:
            record.update(
                {f"val_{k}": v for k, v in _evaluate(head, loaders["val"], device).items()}
            )
            if record.get("val_accuracy", -1) > best.get("val_accuracy", -1):
                best = dict(record)

        history.append(record)
        if log is not None:
            log(record)

    result: dict[str, Any] = {"classes": classes, "history": history, "best_val": best}
    if "test" in loaders:
        result["test"] = _evaluate(head, loaders["test"], device)
    return result

