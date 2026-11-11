"""Label-efficiency curve figures (Phase 8).

Renders the headline figure: accuracy / macro-F1 vs label fraction,
one curve per method (generic MoCo, physics MoCo, physics+DEMON, baselines).
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from phase.data.manifest import resolve

# Consistent colour palette across all figures
PALETTE = {
    "moco_physics_demon": "#2E7089",   # PHASE (ours) – prominent blue
    "moco_physics": "#3D6E53",         # Physics MoCo – green
    "moco_generic": "#B85F18",         # Generic MoCo – orange
    "supervised_100": "#63757F",       # Supervised 100% – grey
    "imagenet_transfer": "#A8362D",    # ImageNet transfer – red
    "random_init": "#888888",          # Random init – light grey
    "audioset_transfer": "#6B4C8A",   # AudioSet – purple
}

DISPLAY_NAMES = {
    "moco_physics_demon": "PHASE (physics + DEMON)",
    "moco_physics": "MoCo-physics",
    "moco_generic": "MoCo-generic",
    "supervised_100": "Supervised 100%",
    "imagenet_transfer": "ImageNet transfer",
    "random_init": "Random init",
    "audioset_transfer": "AudioSet transfer",
}

LABEL_FRACTIONS = [0.01, 0.05, 0.10, 0.25, 0.50, 1.00]


def label_efficiency_curve(
    results: dict[str, dict[str, Any]],
    out: str | Path,
    metric: str = "accuracy",
    fractions: list[float] = LABEL_FRACTIONS,
) -> Path:
    """Plot mean ± std for each method across label fractions.

    Parameters
    ----------
    results:
        ``{method_name: label_efficiency_result_dict}`` where each value
        is the dict returned by ``phase.eval.label_efficiency.run_label_efficiency``.
    out:
        Output PNG path.
    metric:
        ``"accuracy"`` or ``"macro_f1"``.
    """
    fig, ax = plt.subplots(figsize=(9, 5.5))

    x_ticks = [f"{f:.0%}" for f in fractions]

    for method, result in results.items():
        means, stds = [], []
        summary = result.get("summary", {})
        for fraction in fractions:
            key = f"{fraction:.2f}"
            if key in summary and metric in summary[key]:
                means.append(summary[key][metric]["mean"])
                stds.append(summary[key][metric]["std"])
            else:
                means.append(float("nan"))
                stds.append(0.0)

        means_arr = np.array(means)
        stds_arr = np.array(stds)
        valid = np.isfinite(means_arr)

        colour = PALETTE.get(method, "#333333")
        label = DISPLAY_NAMES.get(method, method)
        x = np.arange(len(fractions))[valid]
        ax.plot(x, means_arr[valid], "o-", color=colour, label=label, linewidth=2.0,
                markersize=5)
        ax.fill_between(
            x, means_arr[valid] - stds_arr[valid], means_arr[valid] + stds_arr[valid],
            alpha=0.15, color=colour,
        )

    ax.set_xticks(range(len(fractions)))
    ax.set_xticklabels(x_ticks)
    ax.set_xlabel("Fraction of labelled training recordings", fontsize=12)
    ax.set_ylabel(metric.replace("_", " ").title(), fontsize=12)
    ax.set_title("Label-efficiency curves — DeepShip", fontsize=13)
    ax.legend(fontsize=9, frameon=True, framealpha=0.9)
    ax.grid(alpha=0.25, linewidth=0.6)
    ax.set_ylim(0, 1.02)

    fig.tight_layout()
    target = resolve(out)
    target.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(target, dpi=150)
    plt.close(fig)
    return target


def training_monitor_curves(
    history: list[dict[str, Any]],
    out: str | Path,
    name: str = "pretrain",
) -> Path:
    """Plot alignment, uniformity, rank_fraction, and k-NN accuracy over epochs."""
    epochs = [r["epoch"] for r in history]
    metrics = {
        "alignment": ("Alignment  ↓", "#B85F18"),
        "uniformity": ("Uniformity  ↓", "#2E7089"),
        "rank_fraction": ("Rank fraction  ↑", "#3D6E53"),
        "knn_accuracy": ("k-NN accuracy  ↑", "#A8362D"),
    }
    available = [m for m in metrics if any(m in r for r in history)]

    n = len(available)
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 4))
    axes = np.atleast_1d(axes)

    for ax, metric in zip(axes, available):
        values = [r.get(metric, float("nan")) for r in history]
        label, colour = metrics[metric]
        ax.plot(epochs, values, linewidth=1.6, color=colour)
        ax.set_xlabel("epoch")
        ax.set_title(label, fontsize=10)
        ax.grid(alpha=0.25, linewidth=0.6)

    fig.suptitle(f"MoCo training monitors — {name}", fontsize=12)
    fig.tight_layout()
    target = resolve(out)
    target.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(target, dpi=130)
    plt.close(fig)
    return target


def confusion_matrix_figure(
    matrix: np.ndarray,
    classes: list[str],
    out: str | Path,
    title: str = "Confusion matrix",
) -> Path:
    """Render a normalised confusion matrix as a figure."""
    norm = matrix.astype(float) / matrix.sum(axis=1, keepdims=True).clip(1)
    fig, ax = plt.subplots(figsize=(max(5, len(classes)), max(4, len(classes))))
    im = ax.imshow(norm, vmin=0, vmax=1, cmap="Blues")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    ax.set_xticks(range(len(classes)))
    ax.set_yticks(range(len(classes)))
    ax.set_xticklabels(classes, rotation=45, ha="right", fontsize=9)
    ax.set_yticklabels(classes, fontsize=9)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)

    for i in range(len(classes)):
        for j in range(len(classes)):
            ax.text(
                j, i, f"{norm[i, j]:.2f}", ha="center", va="center",
                fontsize=8, color="white" if norm[i, j] > 0.5 else "black",
            )

    fig.tight_layout()
    target = resolve(out)
    target.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(target, dpi=130)
    plt.close(fig)
    return target


def cross_dataset_bar(
    results: dict[str, dict[str, Any]],
    out: str | Path,
    metric: str = "accuracy",
) -> Path:
    """Bar chart comparing source-test vs target-domain scores per method."""
    methods = list(results)
    source_means = []
    target_means = []
    source_stds = []
    target_stds = []

    for method in methods:
        s = results[method].get("source_summary", {})
        t = results[method].get("target_summary", {})
        source_means.append(s.get(metric, {}).get("mean", float("nan")))
        source_stds.append(s.get(metric, {}).get("std", 0.0))
        target_means.append(t.get(metric, {}).get("mean", float("nan")))
        target_stds.append(t.get(metric, {}).get("std", 0.0))

    x = np.arange(len(methods))
    width = 0.35
    fig, ax = plt.subplots(figsize=(max(6, 2 * len(methods)), 5))

    ax.bar(x - width / 2, source_means, width, yerr=source_stds,
           label="Source (DeepShip)", color="#2E7089", capsize=4, alpha=0.85)
    ax.bar(x + width / 2, target_means, width, yerr=target_stds,
           label="Target (ShipsEar)", color="#B85F18", capsize=4, alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels([DISPLAY_NAMES.get(m, m) for m in methods], rotation=20, ha="right")
    ax.set_ylabel(metric.replace("_", " ").title())
    ax.set_title("Cross-dataset transfer — source vs target accuracy")
    ax.legend()
    ax.set_ylim(0, 1.05)
    ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    target = resolve(out)
    target.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(target, dpi=130)
    plt.close(fig)
    return target
