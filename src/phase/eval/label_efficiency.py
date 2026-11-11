"""Label-efficiency curve evaluation (Phase 7).

Sweeps over label fractions [0.01, 0.05, 0.10, 0.25, 0.50, 1.00] and
runs a linear probe at each fraction for each seed. Outputs one JSON row
per (fraction, seed) pair, plus an aggregate summary table.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from phase.config import EvalConfig, OptimConfig
from phase.eval.linear_probe import train_linear_probe
from phase.eval.metrics import aggregate
from phase.data.manifest import resolve


def run_label_efficiency(
    embeddings: np.ndarray,
    meta: dict[str, Any],
    config: EvalConfig,
    device: str = "cpu",
    log: Any = None,
) -> dict[str, Any]:
    """Run label-efficiency sweep over fractions and seeds.

    Returns
    -------
    dict with keys:
        ``fractions`` – list of floats
        ``seeds`` – list of ints
        ``rows`` – list of per-(fraction, seed) dicts
        ``summary`` – dict[fraction_str -> aggregate stats]
    """
    rows: list[dict[str, Any]] = []

    for fraction in config.label_fractions:
        fraction_rows: list[dict[str, float]] = []
        for seed in config.seeds:
            result = train_linear_probe(
                embeddings,
                meta,
                config.optim,
                label_fraction=fraction,
                seed=seed,
                device=device,
                log=log,
            )
            # Prefer held-out test scores; fall back to best val
            scores = result.get("test", result["best_val"])
            row = {
                "fraction": fraction,
                "seed": seed,
                **{k: v for k, v in scores.items() if isinstance(v, float)},
            }
            rows.append(row)
            fraction_rows.append({k: v for k, v in scores.items() if isinstance(v, float)})

            print(
                f"  fraction={fraction:.0%}  seed={seed}"
                + "".join(f"  {k}={v:.4f}" for k, v in row.items()
                          if k not in {"fraction", "seed"})
            )

        agg = aggregate(fraction_rows)
        print(
            f"  fraction={fraction:.0%}  MEAN:"
            + "".join(f"  {k}={v['mean']:.4f}±{v['std']:.4f}" for k, v in agg.items())
        )

    # Build per-fraction summary
    summary: dict[str, Any] = {}
    for fraction in config.label_fractions:
        key = f"{fraction:.2f}"
        fraction_rows_filtered = [
            {k: v for k, v in r.items() if isinstance(v, float) and k not in {"fraction", "seed"}}
            for r in rows if r["fraction"] == fraction
        ]
        summary[key] = aggregate(fraction_rows_filtered)

    return {
        "protocol": "label_efficiency",
        "fractions": config.label_fractions,
        "seeds": config.seeds,
        "rows": rows,
        "summary": summary,
    }


def write(result: dict[str, Any], path: str | Path) -> Path:
    out = resolve(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return out


def render_table(result: dict[str, Any]) -> str:
    """Console-printable table of mean ± std per fraction."""
    fractions = result["fractions"]
    summary = result["summary"]
    metrics = sorted(next(iter(summary.values())))

    header = f"{'fraction':>10}" + "".join(f"  {m:>22}" for m in metrics)
    divider = "-" * len(header)
    lines = [header, divider]
    for fraction in fractions:
        key = f"{fraction:.2f}"
        stats = summary[key]
        row = f"{fraction:>10.0%}"
        for metric in metrics:
            if metric in stats:
                row += f"  {stats[metric]['mean']:>10.4f} ± {stats[metric]['std']:>8.4f}"
            else:
                row += f"  {'':>22}"
        lines.append(row)
    return "\n".join(lines)

