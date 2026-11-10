from __future__ import annotations

import math
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from phase.config import OptimConfig
from phase.eval.finetune import cosine_lr, make_loaders
from phase.eval.metrics import confusion, per_class_f1, summarise
from phase.models.transfer import load_classifier


@torch.no_grad()
def extract_embeddings(
    backbone: nn.Module, loader: DataLoader, device: str
) -> tuple[torch.Tensor, torch.Tensor]:
    backbone.eval()
    all_embeddings = []
    all_targets = []
    for inputs, targets in loader:
        inputs = inputs.to(device)
        if inputs.dim() == 3:
            inputs = inputs.unsqueeze(1)
        out = backbone(inputs)
        all_embeddings.append(out.detach().cpu())
        all_targets.append(targets)
    if not all_embeddings:
        return torch.empty(0), torch.empty(0, dtype=torch.long)
    return torch.cat(all_embeddings), torch.cat(all_targets)


def train_linear_head(
    train_x: torch.Tensor,
    train_y: torch.Tensor,
    val_x: torch.Tensor,
    val_y: torch.Tensor,
    test_x: torch.Tensor,
    test_y: torch.Tensor,
    n_classes: int,
    optim_config: OptimConfig,
    device: str = "cpu",
) -> dict[str, Any]:
    dim = train_x.shape[1]
    head = nn.Linear(dim, n_classes).to(device)
    criterion = nn.CrossEntropyLoss()

    if optim_config.optimizer == "adamw":
        optimiser = torch.optim.AdamW(
            head.parameters(), lr=optim_config.lr, weight_decay=optim_config.weight_decay
        )
    else:
        optimiser = torch.optim.SGD(
            head.parameters(),
            lr=optim_config.lr,
            momentum=optim_config.momentum,
            weight_decay=optim_config.weight_decay,
        )

    train_dataset = TensorDataset(train_x, train_y)
    train_loader = DataLoader(
        train_dataset, batch_size=optim_config.batch_size, shuffle=True, drop_last=False
    )

    best_val_acc = -1.0
    best_state = None

    for epoch in range(optim_config.epochs):
        head.train()
        lr = cosine_lr(epoch, optim_config.epochs, optim_config.warmup_epochs, optim_config.lr)
        for param_group in optimiser.param_groups:
            param_group["lr"] = lr

        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            optimiser.zero_grad(set_to_none=True)
            loss = criterion(head(batch_x), batch_y)
            loss.backward()
            optimiser.step()

        if val_x.numel() > 0:
            head.eval()
            with torch.no_grad():
                val_preds = head(val_x.to(device)).argmax(dim=1).cpu().numpy()
                val_acc = float((val_preds == val_y.numpy()).mean())
                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    best_state = {k: v.cpu().clone() for k, v in head.state_dict().items()}

    if best_state is not None:
        head.load_state_dict(best_state)

    head.eval()
    with torch.no_grad():
        test_preds = head(test_x.to(device)).argmax(dim=1).cpu().numpy()
        test_truth = test_y.numpy()

    return {
        "best_val_accuracy": best_val_acc,
        **summarise(test_truth, test_preds),
        "predictions": test_preds,
        "truth": test_truth,
    }


def evaluate_linear_probe(
    features: np.ndarray,
    meta: dict[str, Any],
    optim_config: OptimConfig,
    checkpoint_path: str | None = None,
    pretrained_imagenet: bool = False,
    backbone: str = "resnet18",
    device: str = "cpu",
) -> dict[str, Any]:
    loaders, classes = make_loaders(features, meta, optim_config.batch_size)
    classifier = load_classifier(
        backbone,
        len(classes),
        checkpoint_path=checkpoint_path,
        pretrained=pretrained_imagenet,
        device=device,
    )
    encoder = classifier.backbone
    for p in encoder.parameters():
        p.requires_grad = False

    train_x, train_y = extract_embeddings(encoder, loaders["train"], device)
    val_x, val_y = (
        extract_embeddings(encoder, loaders["val"], device)
        if "val" in loaders
        else (torch.empty(0), torch.empty(0, dtype=torch.long))
    )
    test_x, test_y = (
        extract_embeddings(encoder, loaders["test"], device)
        if "test" in loaders
        else (val_x, val_y)
    )

    result = train_linear_head(
        train_x,
        train_y,
        val_x,
        val_y,
        test_x,
        test_y,
        len(classes),
        optim_config,
        device=device,
    )
    result["classes"] = classes
    result["per_class_f1"] = per_class_f1(result["truth"], result["predictions"], classes)
    result["confusion"] = confusion(result["truth"], result["predictions"], classes).tolist()
    return result

