from __future__ import annotations

import argparse
import sys

from phase.config import REPO_ROOT, DatasetConfig, load_config
from phase.data import manifest

CONFIG_DIR = REPO_ROOT / "configs" / "data"

ROOT_OVERRIDES = {
    "deepship": "data/raw/deepship/deepship-raw",
    "esc50": "data/raw/esc50/environmental-sound-classification-50/audio/audio/44100",
    "shipsear_ds3500": "data/raw/ds3500/ShipsEar/shipsear_5s_16k",
}


def main(argv: list[str] | None = None) -> int:
    available = {p.stem: p for p in sorted(CONFIG_DIR.glob("*.yaml"))}
    parser = argparse.ArgumentParser(description="Build one manifest per dataset")
    parser.add_argument("--dataset", choices=sorted(available), action="append", default=[])
    parser.add_argument("--out", default="data/manifests")
    args = parser.parse_args(argv)

    selected = args.dataset or sorted(available)
    failures = 0
    for key in selected:
        config = load_config(available[key], DatasetConfig)
        root = ROOT_OVERRIDES.get(key)
        print(f"==> {key}")
        try:
            built = manifest.build(config, root=root)
        except SystemExit as exc:
            failures += 1
            print(f"FAILED: {exc}", file=sys.stderr)
            continue
        out = manifest.write(built, f"{args.out}/{key}_manifest.json")
        print(manifest.render(built))
        print(f"wrote {out.relative_to(REPO_ROOT).as_posix()}")
        print()
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
