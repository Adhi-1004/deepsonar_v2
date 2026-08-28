from __future__ import annotations

import argparse
import sys

from phase.config import REPO_ROOT
from phase.data import splits
from phase.seed import set_seed

MANIFEST_DIR = REPO_ROOT / "data" / "manifests"


def main(argv: list[str] | None = None) -> int:
    available = {
        p.stem.replace("_manifest", ""): p for p in sorted(MANIFEST_DIR.glob("*_manifest.json"))
    }
    parser = argparse.ArgumentParser(description="Build immutable leakage-safe splits")
    parser.add_argument("--dataset", choices=sorted(available), action="append", default=[])
    parser.add_argument("--out", default="data/splits")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)

    set_seed(args.seed)
    selected = args.dataset or sorted(available)
    failures = 0
    for key in selected:
        manifest = splits.load_manifest(available[key])
        print(f"==> {key}")
        try:
            built = splits.build(manifest, n_folds=args.folds, seed=args.seed)
            out = splits.write(built, f"{args.out}/{key}_splits.json", overwrite=args.overwrite)
        except splits.SplitError as exc:
            print(f"SKIPPED: {exc}", file=sys.stderr)
            print()
            continue
        print(splits.summarise(built))
        print(f"wrote {out.relative_to(REPO_ROOT).as_posix()}")
        print()
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
