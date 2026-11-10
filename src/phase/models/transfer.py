from __future__ import annotations

import torch
import torch.nn as nn

from phase.models.resnet import SpectrogramClassifier


def load_classifier(
    backbone: str,
    n_classes: int,
    checkpoint_path: str | None = None,
    pretrained: bool = False,
    device: str = "cpu",
) -> SpectrogramClassifier:
    model = SpectrogramClassifier(backbone, n_classes, pretrained=pretrained)
    if checkpoint_path is not None:
        state = torch.load(checkpoint_path, map_location=device, weights_only=False)
        weights = state.get("model", state)
        backbone_weights = {}
        for k, v in weights.items():
            if k.startswith("encoder_q.0."):
                backbone_weights[k[len("encoder_q.0."):]] = v
            elif k.startswith("backbone."):
                backbone_weights[k[len("backbone."):]] = v
        if backbone_weights:
            model.backbone.load_state_dict(backbone_weights, strict=False)
        elif "backbone" in state:
            model.backbone.load_state_dict(state["backbone"])
    return model.to(device)

