from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from sklearn.manifold import TSNE

from phase.data.manifest import resolve


def plot_embeddings_dual(
    embeddings: np.ndarray,
    classes: list[str],
    recordings: list[str],
    out: str | Path,
    title: str = "Embedding Space: Class vs Recording Identity",
    max_samples: int = 1500,
    seed: int = 0,
) -> Path:
    if embeddings.shape[0] > max_samples:
        rng = np.random.default_rng(seed)
        idx = rng.choice(embeddings.shape[0], max_samples, replace=False)
        embeddings = embeddings[idx]
        classes = [classes[i] for i in idx]
        recordings = [recordings[i] for i in idx]

    tsne = TSNE(n_components=2, random_state=seed, perplexity=30, n_iter=1000)
    coords = tsne.fit_transform(embeddings)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Coloured by class
    unique_classes = sorted(set(classes))
    cmap_class = plt.cm.get_cmap("tab10", len(unique_classes))
    for i, cls in enumerate(unique_classes):
        mask = [c == cls for c in classes]
        axes[0].scatter(
            coords[mask, 0],
            coords[mask, 1],
            label=cls,
            alpha=0.7,
            s=16,
            c=[cmap_class(i)],
        )
    axes[0].set_title("Coloured by Ship Class")
    axes[0].legend(frameon=True, fontsize=9)
    axes[0].set_xticks([])
    axes[0].set_yticks([])

    # Coloured by recording
    unique_recs = sorted(set(recordings))
    cmap_recs = plt.cm.get_cmap("viridis", len(unique_recs))
    rec_to_idx = {r: i for i, r in enumerate(unique_recs)}
    rec_colors = [cmap_recs(rec_to_idx[r]) for r in recordings]
    axes[1].scatter(
        coords[:, 0],
        coords[:, 1],
        c=rec_colors,
        alpha=0.6,
        s=16,
    )
    axes[1].set_title(f"Coloured by Recording ID ({len(unique_recs)} recordings)")
    axes[1].set_xticks([])
    axes[1].set_yticks([])

    fig.suptitle(title, fontsize=12)
    fig.tight_layout()

    target = resolve(out)
    target.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(target, dpi=150)
    plt.close(fig)
    return target

