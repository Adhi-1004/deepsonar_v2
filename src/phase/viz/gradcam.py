from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from phase.data.manifest import resolve


class GradCAM:
    def __init__(self, model: nn.Module, target_layer: nn.Module):
        self.model = model
        self.target_layer = target_layer
        self.gradients: torch.Tensor | None = None
        self.activations: torch.Tensor | None = None

        target_layer.register_forward_hook(self._save_activation)
        target_layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, module: nn.Module, input: Any, output: torch.Tensor):
        self.activations = output.detach()

    def _save_gradient(self, module: nn.Module, grad_input: Any, grad_output: tuple[torch.Tensor, ...]):
        self.gradients = grad_output[0].detach()

    def generate(self, x: torch.Tensor, class_idx: int | None = None) -> np.ndarray:
        self.model.eval()
        if x.dim() == 2:
            x = x.unsqueeze(0).unsqueeze(0)
        elif x.dim() == 3:
            x = x.unsqueeze(0)

        output = self.model(x)
        if class_idx is None:
            class_idx = int(output.argmax(dim=1).item())

        self.model.zero_grad()
        loss = output[0, class_idx]
        loss.backward()

        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = F.relu((weights * self.activations).sum(dim=1, keepdim=True))
        cam = F.interpolate(cam, size=(x.shape[-2], x.shape[-1]), mode="bilinear", align_corners=False)
        cam = cam.squeeze().cpu().numpy()
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        return cam


def plot_gradcam(
    spectrogram: np.ndarray,
    cam: np.ndarray,
    out: str | Path,
    title: str = "Spectrogram Grad-CAM",
    sample_rate: int = 32000,
) -> Path:
    fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)

    axes[0].imshow(spectrogram, aspect="auto", origin="lower", cmap="magma")
    axes[0].set_title("Input Log-Mel Spectrogram")
    axes[0].set_ylabel("Mel Bin")

    axes[1].imshow(spectrogram, aspect="auto", origin="lower", cmap="magma", alpha=0.6)
    axes[1].imshow(cam, aspect="auto", origin="lower", cmap="jet", alpha=0.4)
    axes[1].set_title("Grad-CAM Attention Overlay")
    axes[1].set_xlabel("Time Frame")
    axes[1].set_ylabel("Mel Bin")

    fig.suptitle(title, fontsize=12)
    fig.tight_layout()

    target = resolve(out)
    target.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(target, dpi=140)
    plt.close(fig)
    return target

