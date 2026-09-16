from __future__ import annotations

import argparse

import torch

from phase.augment import NoiseBank
from phase.config import REPO_ROOT, PretrainConfig, WandbConfig, load_config
from phase.seed import set_seed
from phase.ssl.train_moco import Trainer
from phase.tracking import finish_run, init_run


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="MoCo-v2 pretraining")
    parser.add_argument("--config", required=True)
    parser.add_argument("--windows", default="data/cache/deepship_fold_0_windows.json")
    parser.add_argument("--root", default="data/clean/deepship")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--noise-root", default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--wandb-mode", default="offline")
    args = parser.parse_args(argv)

    config = load_config(args.config, PretrainConfig)
    if args.batch_size:
        config = load_config(args.config, PretrainConfig, optim={"batch_size": args.batch_size})

    set_seed(config.seed)
    bank = NoiseBank(args.noise_root) if args.noise_root else None

    trainer = Trainer(
        config,
        args.windows,
        args.root,
        device=args.device,
        workers=args.workers,
        noise_bank=bank,
    )
    print(
        f"{config.name} | {len(trainer.loader.dataset)} train windows | augment {config.augment.name}"
    )
    print(
        f"queue {config.moco.queue_size} | batch {config.optim.batch_size} | device {args.device}"
    )

    if args.resume:
        checkpoint = REPO_ROOT / config.checkpoint_dir / "last.pt"
        if checkpoint.is_file():
            print(f"resumed from epoch {trainer.resume(checkpoint)}")

    run = init_run(
        config,
        WandbConfig(
            project=config.wandb.project, mode=args.wandb_mode, tags=list(config.wandb.tags)
        ),
        job_type="pretrain",
    )
    try:
        history = trainer.train(epochs=args.epochs, log=lambda record: run.log(record))
    finally:
        finish_run(run)

    if history:
        last = history[-1]
        print()
        print(
            f"epoch {last['epoch']}  loss {last['loss']:.4f}  "
            f"alignment {last['alignment']:.4f}  uniformity {last['uniformity']:.4f}  "
            f"rank {last['rank_fraction']:.3f}  knn {last.get('knn_accuracy', float('nan')):.4f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
