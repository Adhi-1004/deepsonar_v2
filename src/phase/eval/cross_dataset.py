from __future__ import annotations

from typing import Any

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from phase.config import OptimConfig
from phase.eval.finetune import make_loaders
from phase.eval.linear_probe import extract_embeddings, train_linear_head
from phase.eval.metrics import summarise
from phase.models.transfer import load_classifier


def evaluate_cross_dataset(
    source_features: np.ndarray,
    source_meta: dict[str, Any],
    target_features: np.ndarray,
    target_meta: dict[str, Any],
    optim_config: OptimConfig,
    checkpoint_path: str | None = None,
    backbone: str = "resnet18",
    device: str = "cpu",
) -> dict[str, Any]:
    source_loaders, source_classes = make_loaders(source_features, source_meta, optim_config.batch_size)
    target_loaders, target_classes = make_loaders(target_features, target_meta, optim_config.batch_size)

    # Common classes across datasets
    common_classes = sorted(set(source_classes) & set(target_classes))
    if not common_classes:
        common_classes = source_classes

    classifier = load_classifier(
        backbone,
        len(source_classes),
        checkpoint_path=checkpoint_path,
        device=device,
    )
    encoder = classifier.backbone
    for p in encoder.parameters():
        p.requires_grad = False

    train_x, train_y = extract_embeddings(encoder, source_loaders["train"], device)
    val_x, val_y = (
        extract_embeddings(encoder, source_loaders["val"], device)
        if "val" in source_loaders
        else (torch.empty(0), torch.empty(0, dtype=torch.long))
    )

    # Target test evaluation
    target_test_loader = target_loaders.get("test", target_loaders.get("train"))
    target_x, target_y = extract_embeddings(encoder, target_test_loader, device)

    result = train_linear_head(
        train_x,
        train_y,
        val_x,
        val_y,
        target_x,
        target_y,
        len(source_classes),
        optim_config,
        device=device,
    )
    return {
        "source_classes": source_classes,
        "target_classes": target_classes,
        "cross_accuracy": result["accuracy"],
        "cross_macro_f1": result["macro_f1"],
    }

