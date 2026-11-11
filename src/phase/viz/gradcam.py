"""Grad-CAM saliency maps over spectrograms (Phase 8).

Highlights which time-frequency regions the encoder attends to.
If the physics claim is correct, activations should concentrate on
tonal lines (propeller shaft/blade harmonics) rather than on
broadband noise or low-frequency drift.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

from phase.data.manifest import resolve


# ---------------------------------------------------------------------------
# Grad-CAM core
# ---------------------------------------------------------------------------


class GradCAM:
    """Grad-CAM for a CNN whose last feature map is at ``target_layer``.

    Parameters
    ----------
    model:
        The CNN (encoder or full classifier). Should output class logits.
    target_layer:
        The convolutional layer whose feature maps and gradients are hooked.
        Typically the last residual block before the global average pool.
    """

    def __init__(self, model: nn.Module, target_layer: nn.Module):
        self.model = model
        self._activations: torch.Tensor | None = None
        self._gradients: torch.Tensor | None = None

        self._fwd_hook = target_layer.register_forward_hook(self._save_activation)
        self._bwd_hook = target_layer.register_full_backward_hook(self._save_gradient)

    # ------------------------------------------------------------------
    def _save_activation(
        self, module: nn.Module, input: Any, output: torch.Tensor
    ) -> None:
        self._activations = output.detach()

    def _save_gradient(
        self, module: nn.Module, grad_input: Any, grad_output: tuple[torch.Tensor, ...]
    ) -> None:
        self._gradients = grad_output[0].detach()

    # ------------------------------------------------------------------
    def __call__(
        self, spectrogram: torch.Tensor, class_index: int | None = None
    ) -> np.ndarray:
        """Compute Grad-CAM heat map.

        Parameters
        ----------
        spectrogram:
            Tensor of shape ``(1, C, H, W)`` – a single spectrogram.
        class_index:
            Which output logit to differentiate against. If ``None``, uses
            the predicted class (argmax).

        Returns
        -------
        cam : ndarray of shape ``(H, W)`` with values in ``[0, 1]``.
        """
        self.model.eval()
        logits = self.model(spectrogram)
        if class_index is None:
            class_index = int(logits.argmax(dim=1).item())

        self.model.zero_grad()
        logits[0, class_index].backward()

        # Global-average-pool gradients → weights
        weights = self._gradients.mean(dim=(2, 3), keepdim=True)  # (1, C, 1, 1)
        cam = (weights * self._activations).sum(dim=1, keepdim=True)  # (1, 1, h, w)
        cam = torch.relu(cam).squeeze().cpu().numpy()

        # Resize to spectrogram spatial dims
        import torch.nn.functional as F
        h, w = spectrogram.shape[-2], spectrogram.shape[-1]
        cam_t = torch.from_numpy(cam).unsqueeze(0).unsqueeze(0)
        cam_up = F.interpolate(cam_t, size=(h, w), mode="bilinear", align_corners=False)
        cam_np = cam_up.squeeze().numpy()

        # Normalise to [0, 1]
        if cam_np.max() > 0:
            cam_np = cam_np / cam_np.max()
        return cam_np

    def remove_hooks(self) -> None:
        self._fwd_hook.remove()
        self._bwd_hook.remove()


# ---------------------------------------------------------------------------
# Convenience: find the last layer4 in a ResNet backbone
# ---------------------------------------------------------------------------


def _last_conv_layer(model: nn.Module) -> nn.Module:
    """Return the last residual block (layer4) of a ResNet-style backbone."""
    # Walk through named modules and return the last nn.Sequential named layer4
    target = None
    for name, module in model.named_modules():
        if "layer4" in name and isinstance(module, nn.Sequential):
            target = module
    if target is None:
        # Fallback: return the last Conv2d found
        for name, module in model.named_modules():
            if isinstance(module, nn.Conv2d):
                target = module
    if target is None:
        raise RuntimeError("Could not locate a target layer for Grad-CAM")
    return target


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------


def gradcam_panel(
    model: nn.Module,
    spectrograms: list[np.ndarray],
    class_names: list[str],
    true_classes: list[str],
    sample_rate: int,
    out: str | Path,
    class_index: int | None = None,
    n_cols: int = 4,
) -> Path:
    """Render a grid of spectrogram + Grad-CAM overlay.

    Parameters
    ----------
    model:
        A ``SpectrogramClassifier`` or similar that takes ``(1,1,H,W)`` input
        and returns ``(1, n_classes)`` logits.
    spectrograms:
        List of ``(H, W)`` log-mel arrays (already z-scored).
    class_names:
        Global list of class names (index → name).
    true_classes:
        List of ground-truth class names for each spectrogram.
    out:
        Output PNG path.
    class_index:
        Fixed class index to explain. If ``None``, explains the predicted class.
    """
    # Identify target layer
    backbone = None
    for name, module in model.named_modules():
        if hasattr(module, "layer4"):
            backbone = module
            break
    if backbone is None:
        backbone = model

    target_layer = _last_conv_layer(backbone)
    grad_cam = GradCAM(model, target_layer)

    n = len(spectrograms)
    n_cols = min(n_cols, n)
    n_rows = (n + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(3.5 * n_cols, 3.5 * n_rows))
    axes = np.atleast_2d(axes)

    for idx, spec in enumerate(spectrograms):
        row, col = divmod(idx, n_cols)
        ax = axes[row, col]

        # Build tensor
        tensor = torch.from_numpy(spec).float().unsqueeze(0).unsqueeze(0)  # (1,1,H,W)
        cam = grad_cam(tensor, class_index)
        predicted = class_names[int(model(tensor).argmax(dim=1).item())]

        ax.imshow(spec, aspect="auto", origin="lower", cmap="magma")
        ax.imshow(cam, aspect="auto", origin="lower", cmap="jet", alpha=0.45)
        ax.set_title(
            f"true: {true_classes[idx]}\npred: {predicted}",
            fontsize=9,
            color="white" if true_classes[idx] == predicted else "crimson",
        )
        ax.set_xticks([])
        ax.set_yticks([])

    # Hide unused axes
    for idx in range(n, n_rows * n_cols):
        row, col = divmod(idx, n_cols)
        axes[row, col].set_visible(False)

    grad_cam.remove_hooks()
    fig.suptitle("Grad-CAM — heat map over log-Mel spectrogram", fontsize=12)
    fig.tight_layout()

    target = resolve(out)
    target.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(target, dpi=130)
    plt.close(fig)
    return target

