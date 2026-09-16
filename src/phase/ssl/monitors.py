from __future__ import annotations

import numpy as np
import torch

from phase.ssl.losses import alignment, uniformity


@torch.no_grad()
def collapse_statistics(embeddings: torch.Tensor) -> dict[str, float]:
    """Two independent collapse signals.

    Per-dimension standard deviation goes to zero when every input maps to one
    point. Effective rank falls when the representation occupies a subspace far
    smaller than its nominal width, which is the subtler failure and the one a
    k-NN probe can still look healthy through.
    """
    centred = embeddings - embeddings.mean(dim=0, keepdim=True)
    per_dimension = centred.std(dim=0)

    covariance = (centred.T @ centred) / max(1, centred.shape[0] - 1)
    eigenvalues = torch.linalg.eigvalsh(covariance.float()).clamp_min(0)
    total = eigenvalues.sum()
    if total <= 0:
        return {"embedding_std": 0.0, "effective_rank": 0.0, "rank_fraction": 0.0}

    proportions = (eigenvalues / total).clamp_min(1e-12)
    entropy = -(proportions * proportions.log()).sum()
    effective_rank = float(entropy.exp())

    return {
        "embedding_std": float(per_dimension.mean()),
        "embedding_std_min": float(per_dimension.min()),
        "effective_rank": effective_rank,
        "rank_fraction": effective_rank / embeddings.shape[1],
    }


@torch.no_grad()
def knn_accuracy(
    train_embeddings: torch.Tensor,
    train_labels: torch.Tensor,
    test_embeddings: torch.Tensor,
    test_labels: torch.Tensor,
    k: int = 20,
    temperature: float = 0.07,
) -> float:
    """Cosine k-NN probe. The earliest signal that pretraining is doing anything."""
    if train_embeddings.numel() == 0 or test_embeddings.numel() == 0:
        return float("nan")

    train = torch.nn.functional.normalize(train_embeddings, dim=1)
    test = torch.nn.functional.normalize(test_embeddings, dim=1)
    classes = int(train_labels.max().item()) + 1
    k = min(k, train.shape[0])

    similarity = test @ train.T
    weights, indices = similarity.topk(k, dim=1)
    weights = (weights / temperature).exp()

    neighbours = train_labels[indices]
    scores = torch.zeros(test.shape[0], classes, device=test.device)
    scores.scatter_add_(1, neighbours, weights)
    return float((scores.argmax(dim=1) == test_labels).float().mean())


@torch.no_grad()
def health(query: torch.Tensor, key: torch.Tensor) -> dict[str, float]:
    stats = {
        "alignment": float(alignment(query, key)),
        "uniformity": float(uniformity(query)),
    }
    stats.update(collapse_statistics(query))
    return stats


def is_collapsed(stats: dict[str, float], knn: float, chance: float) -> bool:
    """Uniformity plateauing high while k-NN sits at chance is the classic signature."""
    if not np.isfinite(knn):
        return False
    return stats["rank_fraction"] < 0.05 or (knn <= chance * 1.1 and stats["alignment"] < 1e-3)
