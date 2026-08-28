from __future__ import annotations

import math
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from phase.config import OptimConfig
from phase.eval.metrics import summarise
from phase.models.resnet import SpectrogramClassifier


def make_loaders(
    features: np.ndarray, meta: dict[str, Any], batch_size: int
) -> tuple[dict[str, DataLoader], list[str]]:
    classes = sorted(set(meta["labels"]))
    lookup = {name: index for index, name in enumerate(classes)}
    targets = np.array([lookup[name] for name in meta["labels"]], dtype=np.int64)
    splits = np.array(meta["splits"])

    loaders = {}
    for name in ("train", "val", "test"):
        mask = np.flatnonzero(splits == name)
        if mask.size == 0:
            continue
        dataset = TensorDataset(
            torch.from_numpy(np.ascontiguousarray(features[mask])).float(),
            torch.from_numpy(targets[mask]),
        )
        loaders[name] = DataLoader(
            dataset, batch_size=batch_size, shuffle=name == "train", drop_last=False
        )
    return loaders, classes


def cosine_lr(step: int, total: int, warmup: int, base: float) -> float:
    if step < warmup:
        return base * (step + 1) / max(1, warmup)
    progress = (step - warmup) / max(1, total - warmup)
    return base * 0.5 * (1 + math.cos(math.pi * progress))


def train_supervised(
    features: np.ndarray,
    meta: dict[str, Any],
    optim_config: OptimConfig,
    backbone: str = "resnet18",
    device: str = "cpu",
    log: Any = None,
) -> dict[str, Any]:
    loaders, classes = make_loaders(features, meta, optim_config.batch_size)
    model = SpectrogramClassifier(backbone, len(classes)).to(device)

    if optim_config.optimizer == "adamw":
        optimiser = torch.optim.AdamW(
            model.parameters(), lr=optim_config.lr, weight_decay=optim_config.weight_decay
        )
    else:
        optimiser = torch.optim.SGD(
            model.parameters(),
            lr=optim_config.lr,
            momentum=optim_config.momentum,
            weight_decay=optim_config.weight_decay,
        )
    criterion = nn.CrossEntropyLoss()

    history = []
    best = {"val_accuracy": -1.0}
    for epoch in range(optim_config.epochs):
        model.train()
        total_loss = 0.0
        seen = 0
        for inputs, targets in loaders["train"]:
            inputs, targets = inputs.to(device), targets.to(device)
            for group in optimiser.param_groups:
                group["lr"] = cosine_lr(
                    epoch, optim_config.epochs, optim_config.warmup_epochs, optim_config.lr
                )
            optimiser.zero_grad(set_to_none=True)
            loss = criterion(model(inputs), targets)
            loss.backward()
            optimiser.step()
            total_loss += loss.item() * targets.size(0)
            seen += targets.size(0)

        record = {"epoch": epoch, "train_loss": total_loss / max(1, seen)}
        if "val" in loaders:
            record.update(
                {f"val_{k}": v for k, v in evaluate(model, loaders["val"], device).items()}
            )
            if record.get("val_accuracy", -1) > best["val_accuracy"]:
                best = {"epoch": epoch, **{k: v for k, v in record.items() if k.startswith("val_")}}
        history.append(record)
        if log is not None:
            log(record)

    result = {"classes": classes, "history": history, "best_val": best}
    if "test" in loaders:
        result["test"] = evaluate(model, loaders["test"], device)
    return result


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: str) -> dict[str, float]:
    model.eval()
    predictions = []
    truths = []
    for inputs, targets in loader:
        logits = model(inputs.to(device))
        predictions.append(logits.argmax(dim=1).cpu().numpy())
        truths.append(targets.numpy())
    return summarise(np.concatenate(truths), np.concatenate(predictions))
