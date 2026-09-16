from __future__ import annotations

import torch
import torch.nn.functional as F


def info_nce(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    return F.cross_entropy(logits, labels)


def alignment(query: torch.Tensor, key: torch.Tensor) -> torch.Tensor:
    """E||f(x) - f(x+)||^2 over positive pairs. Lower means views map together."""
    return (query - key).norm(dim=1).pow(2).mean()


def uniformity(embeddings: torch.Tensor, t: float = 2.0) -> torch.Tensor:
    """log E exp(-t||f(x) - f(y)||^2). Lower means the sphere is covered evenly."""
    distances = torch.pdist(embeddings, p=2).pow(2)
    return distances.mul(-t).exp().mean().clamp_min(1e-12).log()
