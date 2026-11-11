"""Embedding extraction and t-SNE / UMAP visualisation (Phase 8).

Produces two scatter plots:
  - coloured by ship *class*  → shows whether the encoder separates ship types
  - coloured by *recording*   → shows whether the encoder ignores recording identity
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
from torch.utils.data import DataLoader

from phase.data.manifest import resolve
from phase.features.frontend import LogMelFrontEnd


# ---------------------------------------------------------------------------
# Embedding extraction
# ---------------------------------------------------------------------------


@torch.no_grad()
def extract_embeddings(
    encoder: nn.Module,
    frontend: LogMelFrontEnd,
    loader: DataLoader,
    device: str = "cpu",
    limit: int = 4096,
) -> tuple[np.ndarray, list[str], list[str]]:
    """Run all windows through the (frozen) encoder backbone.

    Returns
    -------
    embeddings : ndarray of shape (N, D)
    labels     : list of class-name strings, length N
    rec_ids    : list of recording-id strings, length N
    """
    encoder.eval()
    frontend.eval()
    all_emb, all_labels, all_rids = [], [], []
    seen = 0

    classes = loader.dataset.classes
    for waveforms, label_indices in loader:
        waveforms = waveforms.to(device)
        features = frontend(waveforms)
        # Use the backbone (strip projection head if present)
        if hasattr(encoder, "embed"):
            emb = encoder.embed(features)
        elif isinstance(encoder, nn.Sequential):
            emb = encoder[0](features)
        else:
            emb = encoder(features)

        all_emb.append(emb.cpu().numpy())
        all_labels.extend(classes[i] for i in label_indices.tolist())
        all_rids.extend(loader.dataset.recording_ids()[seen : seen + len(label_indices)])
        seen += len(label_indices)
        if seen >= limit:
            break

    embeddings = np.concatenate(all_emb, axis=0)[:limit]
    return embeddings, all_labels[:limit], all_rids[:limit]


# ---------------------------------------------------------------------------
# 2-D projection helpers
# ---------------------------------------------------------------------------


def _tsne(embeddings: np.ndarray, seed: int = 0) -> np.ndarray:
    from sklearn.manifold import TSNE

    return TSNE(n_components=2, random_state=seed, perplexity=min(30, len(embeddings) - 1)).fit_transform(embeddings)


def _umap(embeddings: np.ndarray, seed: int = 0) -> np.ndarray:
    try:
        import umap

        return umap.UMAP(n_components=2, random_state=seed).fit_transform(embeddings)
    except ImportError:
        return _tsne(embeddings, seed)


# ---------------------------------------------------------------------------
# Plot helpers
# ---------------------------------------------------------------------------

_CMAP = plt.cm.get_cmap("tab10")


def _scatter(
    ax: plt.Axes,
    coords: np.ndarray,
    group_labels: list[str],
    title: str,
    max_groups: int = 20,
) -> None:
    unique = list(dict.fromkeys(group_labels))[:max_groups]
    lookup = {name: i for i, name in enumerate(unique)}
    colors = [_CMAP(lookup[l] % 10) if l in lookup else (0.7, 0.7, 0.7, 0.4)
              for l in group_labels]
    ax.scatter(coords[:, 0], coords[:, 1], c=colors, s=4, alpha=0.6, linewidths=0)
    handles = [
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=_CMAP(i % 10),
                   markersize=6, label=name)
        for i, name in enumerate(unique)
    ]
    ax.legend(handles=handles, fontsize=7, frameon=False, markerscale=1.2)
    ax.set_title(title, fontsize=11)
    ax.set_xticks([])
    ax.set_yticks([])


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def embedding_panel(
    embeddings: np.ndarray,
    class_labels: list[str],
    recording_ids: list[str],
    out: str | Path,
    method: str = "tsne",
    seed: int = 0,
) -> Path:
    """Render side-by-side scatter plots coloured by class and by recording.

    Parameters
    ----------
    embeddings:
        ``(N, D)`` array of L2-normalised embedding vectors.
    class_labels:
        Length-N list of class-name strings (e.g. ``"Cargo"``).
    recording_ids:
        Length-N list of recording identifiers.
    out:
        Output PNG path.
    method:
        ``"tsne"`` or ``"umap"``.
    """
    project = _umap if method == "umap" else _tsne
    coords = project(embeddings, seed=seed)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    _scatter(axes[0], coords, class_labels, f"coloured by class  ({method.upper()})")
    _scatter(axes[1], coords, recording_ids, "coloured by recording")

    fig.suptitle(
        f"{method.upper()} of encoder embeddings  (N={len(embeddings)})", fontsize=12
    )
    fig.tight_layout()
    target = resolve(out)
    target.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(target, dpi=130)
    plt.close(fig)
    return target


def per_class_uniformity(
    embeddings: np.ndarray,
    class_labels: list[str],
) -> dict[str, float]:
    """Mean cosine similarity within each class (lower = more spread out)."""
    import torch
    import torch.nn.functional as F

    classes = sorted(set(class_labels))
    label_arr = np.array(class_labels)
    result = {}
    emb_t = torch.from_numpy(embeddings).float()
    emb_t = F.normalize(emb_t, dim=1)

    for cls in classes:
        mask = label_arr == cls
        sub = emb_t[mask]
        if sub.shape[0] < 2:
            result[cls] = float("nan")
            continue
        sim = (sub @ sub.T).fill_diagonal_(0)
        n = sub.shape[0]
        result[cls] = float(sim.sum() / (n * (n - 1)))
    return result

