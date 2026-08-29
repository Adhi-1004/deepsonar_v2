from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from phase.data.manifest import resolve

COLOURS = {
    "none": "#63757F",
    "lloyd": "#B85F18",
    "tl": "#2E7089",
    "lloyd+tl": "#A8362D",
    "lloyd+tl+multipath": "#3D6E53",
}


def physics_validation(payload: dict[str, Any], out: str | Path) -> Path:
    rows = payload["rows"]
    variants = list(COLOURS)
    ranges = sorted({r["range_km"] for r in rows})
    depths = sorted({r["depth_km"] for r in rows})
    bands = [k for k in rows[0] if k.startswith("lsd_") and k != "lsd_db"]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))

    for variant in variants:
        by_range = [
            payload["by_range"].get(str((variant, r)), {}).get("lsd_db", {}).get("mean", np.nan)
            for r in ranges
        ]
        axes[0].plot(ranges, by_range, "o-", color=COLOURS[variant], label=variant, linewidth=1.6)

        by_depth = [
            payload["by_depth"].get(str((variant, d)), {}).get("lsd_db", {}).get("mean", np.nan)
            for d in depths
        ]
        axes[1].plot(depths, by_depth, "o-", color=COLOURS[variant], linewidth=1.6)

        values = [payload["overall"][variant][b]["mean"] for b in bands]
        axes[2].plot(range(len(bands)), values, "o-", color=COLOURS[variant], linewidth=1.6)

    axes[0].set_xlabel("range, km")
    axes[0].set_ylabel("log-spectral distance to BELLHOP, dB")
    axes[0].set_title("by range")
    axes[0].legend(fontsize=8, frameon=False)

    axes[1].set_xlabel("receiver depth, km")
    axes[1].set_title("by receiver depth")

    axes[2].set_xticks(range(len(bands)))
    axes[2].set_xticklabels([b.replace("lsd_", "") for b in bands], fontsize=8)
    axes[2].set_title("by frequency band")

    for axis in axes:
        axis.grid(alpha=0.25, linewidth=0.6)

    fig.suptitle(
        f"Analytic channel model against BELLHOP ray theory — {payload['n_evaluated']} matched "
        "pairs, 36 geometries.  Lower is better; the untouched original is the baseline.",
        fontsize=11,
    )
    fig.tight_layout()

    target = resolve(out)
    target.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(target, dpi=130)
    plt.close(fig)
    return target
