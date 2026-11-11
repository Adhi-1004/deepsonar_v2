"""ImageNet and AudioSet pretrained transfer baselines (Phase 7).

Provides two baselines required by PLAN.md §3 Phase 7:
  3. ImageNet-pretrained ResNet-18 transfer
  4. AudioSet-pretrained PANN (Pretrained Audio Neural Net) transfer

Both are wrapped in the same ``TransferEncoder`` interface so the
evaluation pipeline treats them identically to the MoCo encoder.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from phase.models.resnet import EMBEDDING_DIM, build_backbone


class TransferEncoder(nn.Module):
    """Frozen pretrained encoder for linear-probe / fine-tune evaluation.

    Parameters
    ----------
    name:
        ``"imagenet_resnet18"``, ``"imagenet_resnet34"``, or
        ``"audioset_pann"`` (requires ``timm`` with the PANN weights).
    freeze:
        If ``True``, all backbone parameters are frozen (linear probe).
        If ``False``, the full model is trainable (fine-tune).
    """

    def __init__(self, name: str = "imagenet_resnet18", freeze: bool = True):
        super().__init__()
        self.name = name

        if name in ("imagenet_resnet18", "imagenet_resnet34"):
            backbone_key = name.replace("imagenet_", "")
            self.backbone = build_backbone(backbone_key, in_channels=1, pretrained=True)
            self.embedding_dim = EMBEDDING_DIM[backbone_key]

        elif name == "audioset_pann":
            self.backbone, self.embedding_dim = _build_pann()

        else:
            raise ValueError(
                f"Unknown transfer model {name!r}. "
                "Choose from imagenet_resnet18, imagenet_resnet34, audioset_pann."
            )

        if freeze:
            for param in self.backbone.parameters():
                param.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return embedding tensor of shape ``(B, D)``."""
        if x.dim() == 3:
            x = x.unsqueeze(1)
        return self.backbone(x)

    def embed(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward(x)


# ---------------------------------------------------------------------------
# PANN (AudioSet) helper
# ---------------------------------------------------------------------------


def _build_pann() -> tuple[nn.Module, int]:
    """Load a CNN14 pretrained on AudioSet via timm if available.

    Falls back to an ImageNet ResNet-18 with a warning if timm or the
    weights are unavailable — the caller is responsible for noting this
    in the results log.
    """
    try:
        import timm

        # CNN14 is available as 'cnn14' in timm with AudioSet weights
        model = timm.create_model("cnn14", pretrained=True, num_classes=0, global_pool="avg")
        embedding_dim = model.num_features
        # Adapt to single-channel input (spectrograms are mono)
        first_conv = None
        for name, module in model.named_modules():
            if isinstance(module, nn.Conv2d) and module.in_channels == 3:
                first_conv = (name, module)
                break
        if first_conv is not None:
            attr_path = first_conv[0].split(".")
            parent = model
            for part in attr_path[:-1]:
                parent = getattr(parent, part)
            original = first_conv[1]
            new_conv = nn.Conv2d(
                1, original.out_channels,
                original.kernel_size, original.stride, original.padding, bias=False,
            )
            with torch.no_grad():
                new_conv.weight.copy_(original.weight.mean(dim=1, keepdim=True))
            setattr(parent, attr_path[-1], new_conv)

        return model, int(embedding_dim)

    except Exception as exc:
        import warnings

        warnings.warn(
            f"AudioSet PANN unavailable ({exc}). "
            "Falling back to ImageNet ResNet-18 for the AudioSet baseline slot. "
            "Record this in docs/decisions.md.",
            stacklevel=2,
        )
        backbone = build_backbone("resnet18", in_channels=1, pretrained=True)
        return backbone, EMBEDDING_DIM["resnet18"]


# ---------------------------------------------------------------------------
# Embedding extraction helper (used by eval scripts)
# ---------------------------------------------------------------------------


@torch.no_grad()
def extract(
    encoder: TransferEncoder,
    loader: torch.utils.data.DataLoader,
    device: str = "cpu",
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run a DataLoader through the frozen encoder, return (embeddings, labels)."""
    encoder.eval().to(device)
    all_emb, all_labels = [], []
    for waveforms, labels in loader:
        waveforms = waveforms.to(device)
        all_emb.append(encoder(waveforms).cpu())
        all_labels.append(labels)
    return torch.cat(all_emb), torch.cat(all_labels)

