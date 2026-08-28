from __future__ import annotations

import argparse
import json

from phase.config import REPO_ROOT, DatasetConfig, FeatureConfig, load_config
from phase.data import cache, segment
from phase.data.manifest import resolve
from phase.seed import set_seed

CONFIG_DIR = REPO_ROOT / "configs" / "data"

SOURCES = {
    "deepship": {"root": "data/clean/deepship", "suffix": ".flac"},
    "esc50": {
        "root": "data/raw/esc50/environmental-sound-classification-50/audio/audio/44100",
        "suffix": None,
    },
    "shipsear_ds3500": {"root": "data/raw/ds3500/ShipsEar/shipsear_5s_16k", "suffix": None},
}


def main(argv: list[str] | None = None) -> int:
    available = {p.stem: p for p in sorted(CONFIG_DIR.glob("*.yaml"))}
    parser = argparse.ArgumentParser(
        description="Build a window index and optionally cache features"
    )
    parser.add_argument("--dataset", required=True, choices=sorted(available))
    parser.add_argument("--features", default=None)
    parser.add_argument("--fold", default="fold_0")
    parser.add_argument("--out", default="data/cache")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    set_seed(args.seed)
    config = load_config(available[args.dataset], DatasetConfig)
    source = SOURCES[args.dataset]

    manifest = json.loads(
        resolve(f"data/manifests/{args.dataset}_manifest.json").read_text(encoding="utf-8")
    )

    split_path = resolve(f"data/splits/{args.dataset}_splits.json")
    splits = json.loads(split_path.read_text(encoding="utf-8")) if split_path.is_file() else None
    if splits is None:
        print(f"note: {args.dataset} has no split file; windows will carry no split label")

    index = segment.build(
        manifest,
        splits,
        seconds=config.segment_seconds,
        hop=config.hop_seconds,
        sample_rate=config.sample_rate,
        suffix=source["suffix"],
        fold=args.fold,
    )
    print(segment.render(index))

    if args.dry_run:
        return 0

    out = segment.write(index, f"{args.out}/{args.dataset}_{args.fold}_windows.json")
    print(f"wrote {out.relative_to(REPO_ROOT).as_posix()}")

    if args.features:
        feature_config = load_config(f"configs/features/{args.features}.yaml", FeatureConfig)
        written = cache.build(
            index,
            resolve(source["root"]),
            feature_config,
            f"{args.out}/{args.dataset}_{args.fold}_{args.features}",
        )
        for line in written:
            print(line)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
