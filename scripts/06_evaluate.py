from __future__ import annotations

import argparse
import json

import torch

from phase.config import REPO_ROOT, EvalConfig, WandbConfig, load_config
from phase.data import cache
from phase.data.manifest import resolve
from phase.eval.finetune import train_supervised
from phase.eval.label_efficiency import run_label_efficiency
from phase.eval.leakage_control import run_leakage_control
from phase.eval.linear_probe import evaluate_linear_probe
from phase.eval.metrics import aggregate
from phase.seed import set_seed
from phase.tracking import finish_run, init_run


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Evaluate downstream classification protocols")
    parser.add_argument("--config", default="configs/eval/linear_probe.yaml")
    parser.add_argument("--cache", required=True)
    parser.add_argument("--protocol", choices=["linear_probe", "label_efficiency", "finetune", "leakage_control"], default=None)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--imagenet", action="store_true")
    parser.add_argument("--seeds", type=int, nargs="*", default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--wandb-mode", default="offline")
    parser.add_argument("--out", default="results/tables")
    args = parser.parse_args(argv)

    config = load_config(args.config, EvalConfig)
    protocol = args.protocol or config.protocol
    optim = config.optim
    if args.epochs or args.batch_size:
        overrides = dict(optim.__dict__)
        if args.epochs:
            overrides["epochs"] = args.epochs
            overrides["warmup_epochs"] = min(optim.warmup_epochs, max(0, args.epochs - 1))
        if args.batch_size:
            overrides["batch_size"] = args.batch_size
        optim = type(optim)(**overrides)

    features, meta = cache.load(args.cache)
    print(f"features {features.shape} from {args.cache} | protocol: {protocol}")

    seeds = args.seeds if args.seeds is not None else config.seeds
    out_dir = resolve(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    if protocol == "label_efficiency":
        print(f"running label efficiency across fractions {config.label_fractions}...")
        results = run_label_efficiency(
            features,
            meta,
            optim,
            checkpoint_path=args.checkpoint,
            pretrained_imagenet=args.imagenet,
            fractions=config.label_fractions,
            seeds=seeds,
            device=args.device,
        )
        model_tag = "imagenet" if args.imagenet else ("pretrained" if args.checkpoint else "scratch")
        out_path = out_dir / f"label_efficiency_{model_tag}_{len(seeds)}seeds.json"
        out_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
        print(f"wrote {out_path.relative_to(REPO_ROOT).as_posix()}")
        return 0

    if protocol == "leakage_control":
        print("running recording-ID shuffle leakage control...")
        results = run_leakage_control(
            features,
            meta,
            optim,
            checkpoint_path=args.checkpoint,
            seeds=seeds,
            device=args.device,
        )
        print(f"shuffle control mean acc: {results['mean_accuracy']:.4f} +/- {results['std_accuracy']:.4f} (chance: {results['chance_level']:.4f})")
        out_path = out_dir / f"leakage_control_{len(seeds)}seeds.json"
        out_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
        print(f"wrote {out_path.relative_to(REPO_ROOT).as_posix()}")
        return 0

    runs = []
    for seed in seeds:
        set_seed(seed)
        run = init_run(
            config,
            WandbConfig(project=config.wandb.project, mode=args.wandb_mode, tags=[protocol]),
            job_type="eval",
            name=f"{config.name}-{protocol}-seed{seed}",
        )
        if protocol == "linear_probe":
            result = evaluate_linear_probe(
                features,
                meta,
                optim,
                checkpoint_path=args.checkpoint,
                pretrained_imagenet=args.imagenet,
                device=args.device,
            )
        else:
            result = train_supervised(
                features, meta, optim, device=args.device, log=lambda r: run.log(r)
            )
        finish_run(run)

        scores = {k: v for k, v in result.items() if isinstance(v, float)}
        print(f"seed {seed}: " + " ".join(f"{k}={v:.4f}" for k, v in scores.items()))
        runs.append(scores)

    summary = aggregate(runs)
    print()
    for key, stats in summary.items():
        print(f"{key:18s} {stats['mean']:.4f} +/- {stats['std']:.4f}  (n={stats['n']})")

    out = out_dir / f"{config.name}_{protocol}_{len(seeds)}seeds.json"
    out.write_text(
        json.dumps({"seeds": seeds, "runs": runs, "summary": summary}, indent=2), encoding="utf-8"
    )
    print(f"wrote {out.relative_to(REPO_ROOT).as_posix()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
