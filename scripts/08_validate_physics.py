from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict

from phase.config import REPO_ROOT, FeatureConfig, load_config
from phase.data.manifest import resolve
from phase.eval import physics_validation as pv
from phase.seed import set_seed


def stratified(pairs, per_position: int, seed: int):
    buckets = defaultdict(list)
    for pair in pairs:
        buckets[(pair.range_m, pair.receiver_depth_m)].append(pair)

    rng = random.Random(seed)
    chosen = []
    for key in sorted(buckets):
        members = buckets[key]
        rng.shuffle(members)
        chosen.extend(members[:per_position])
    return chosen


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Validate the analytic ocean channel against BELLHOP renderings"
    )
    parser.add_argument("--root", default="data/raw/ds3500")
    parser.add_argument("--per-position", type=int, default=6)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", default="results/tables/physics_validation.json")
    args = parser.parse_args(argv)

    set_seed(args.seed)
    demon_config = load_config("configs/features/demon.yaml", FeatureConfig).params

    pairs = pv.matched_pairs(args.root)
    sample = stratified(pairs, args.per_position, args.seed)
    print(f"{len(pairs)} matched pairs; evaluating {len(sample)} across 36 geometries")

    rows = pv.validate(sample, demon_config, seed=args.seed)

    overall = pv.aggregate(rows)
    by_range = pv.aggregate(rows, by="range_km")
    by_depth = pv.aggregate(rows, by="depth_km")

    print("\nlog-spectral distance to the BELLHOP rendering, lower is better")
    print(
        f"{'variant':22s} {'LSD dB':>10s} {'10-100':>9s} {'100-500':>9s} {'500-2k':>9s} {'2k-8k':>9s} {'DEMON cos':>10s}"
    )
    for variant in pv.VARIANTS:
        stats = overall.get(variant)
        if not stats:
            continue
        row = [
            stats["lsd_db"]["mean"],
            stats["lsd_10-100Hz"]["mean"],
            stats["lsd_100-500Hz"]["mean"],
            stats["lsd_500-2000Hz"]["mean"],
            stats["lsd_2000-8000Hz"]["mean"],
            stats["demon_cosine"]["mean"],
        ]
        print(f"{variant:22s} " + " ".join(f"{v:9.3f}" for v in row[:-1]) + f" {row[-1]:10.4f}")

    print("\nLSD by range, km")
    ranges = sorted({r["range_km"] for r in rows})
    print(f"{'variant':22s} " + " ".join(f"{r:>8.0f}" for r in ranges))
    for variant in pv.VARIANTS:
        cells = []
        for r in ranges:
            stats = by_range.get(str((variant, r)))
            cells.append(stats["lsd_db"]["mean"] if stats else float("nan"))
        print(f"{variant:22s} " + " ".join(f"{c:8.3f}" for c in cells))

    payload = {
        "n_pairs_total": len(pairs),
        "n_evaluated": len(sample),
        "per_position": args.per_position,
        "seed": args.seed,
        "overall": overall,
        "by_range": by_range,
        "by_depth": by_depth,
        "rows": rows,
    }
    out = resolve(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\nwrote {out.relative_to(REPO_ROOT).as_posix()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
