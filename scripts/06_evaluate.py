from __future__ import annotations

import argparse
import json

import torch

from phase.config import REPO_ROOT, EvalConfig, WandbConfig, load_config
from phase.data import cache
from phase.data.manifest import resolve
from phase.eval.finetune import train_supervised
from phase.eval.metrics import aggregate
from phase.seed import set_seed
from phase.tracking import finish_run, init_run


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Train and evaluate a supervised baseline")
    parser.add_argument("--config", default="configs/eval/finetune.yaml")
    parser.add_argument("--cache", required=True)
    parser.add_argument("--seeds", type=int, nargs="*", default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--wandb-mode", default="offline")
    parser.add_argument("--out", default="results/tables")
    args = parser.parse_args(argv)

    config = load_config(args.config, EvalConfig)
    optim = config.optim
    if args.epochs:
        overrides = {
            **optim.__dict__,
            "epochs": args.epochs,
            "warmup_epochs": min(optim.warmup_epochs, max(0, args.epochs - 1)),
        }
        optim = type(optim)(**overrides)

    features, meta = cache.load(args.cache)
    print(f"features {features.shape} from {args.cache}")

    seeds = args.seeds if args.seeds is not None else config.seeds
    runs = []
    for seed in seeds:
        set_seed(seed)
        run = init_run(
            config,
            WandbConfig(project=config.wandb.project, mode=args.wandb_mode, tags=["dry_run"]),
            job_type="supervised",
            name=f"{config.name}-seed{seed}",
        )
        result = train_supervised(
            features, meta, optim, device=args.device, log=lambda r: run.log(r)
        )
        finish_run(run)

        scores = result.get("test", result["best_val"])
        print(
            f"seed {seed}: " + " ".join(f"{k}={v:.4f}" for k, v in scores.items() if k != "epoch")
        )
        runs.append({k: v for k, v in scores.items() if isinstance(v, float)})

    summary = aggregate(runs)
    print()
    for key, stats in summary.items():
        print(f"{key:18s} {stats['mean']:.4f} +/- {stats['std']:.4f}  (n={stats['n']})")

    out = resolve(f"{args.out}/{config.name}_{len(seeds)}seeds.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps({"seeds": seeds, "runs": runs, "summary": summary}, indent=2), encoding="utf-8"
    )
    print(f"wrote {out.relative_to(REPO_ROOT).as_posix()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
