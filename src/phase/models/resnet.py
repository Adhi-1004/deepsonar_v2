from __future__ import annotations

import torch
import torch.nn as nn
from torchvision.models import resnet18, resnet34

BACKBONES = {"resnet18": resnet18, "resnet34": resnet34}

EMBEDDING_DIM = {"resnet18": 512, "resnet34": 512}


def build_backbone(name: str, in_channels: int = 1, pretrained: bool = False) -> nn.Module:
    model = BACKBONES[name](weights="IMAGENET1K_V1" if pretrained else None)

    if in_channels != 3:
        original = model.conv1
        model.conv1 = nn.Conv2d(
            in_channels,
            original.out_channels,
            kernel_size=original.kernel_size,
            stride=original.stride,
            padding=original.padding,
            bias=False,
        )
        if pretrained:
            with torch.no_grad():
                model.conv1.weight.copy_(original.weight.mean(dim=1, keepdim=True))

    model.fc = nn.Identity()
    return model


class SpectrogramClassifier(nn.Module):
    def __init__(
        self, backbone: str, n_classes: int, in_channels: int = 1, pretrained: bool = False
    ):
        super().__init__()
        self.backbone = build_backbone(backbone, in_channels, pretrained)
        self.head = nn.Linear(EMBEDDING_DIM[backbone], n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 3:
            x = x.unsqueeze(1)
        return self.head(self.backbone(x))

    def embed(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 3:
            x = x.unsqueeze(1)
        return self.backbone(x)
