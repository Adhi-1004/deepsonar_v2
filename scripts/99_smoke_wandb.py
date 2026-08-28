from __future__ import annotations

import argparse

from phase.config import PretrainConfig, WandbConfig, load_config
from phase.seed import set_seed
from phase.tracking import finish_run, git_sha, init_run


def main() -> None:
    parser = argparse.ArgumentParser(description="Log a throwaway run to confirm W&B is wired up")
    parser.add_argument("--config", default="configs/pretrain/moco_physics_demon.yaml")
    parser.add_argument("--mode", default="offline", choices=["online", "offline", "disabled"])
    parser.add_argument("--project", default="phase")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    set_seed(args.seed)
    config = load_config(args.config, PretrainConfig)
    wandb_config = WandbConfig(project=args.project, mode=args.mode, tags=["smoke"])

    run = init_run(config, wandb_config, job_type="smoke", name="phase-0-smoke")
    for epoch in range(3):
        run.log({"epoch": epoch, "smoke/value": 1.0 / (epoch + 1)})
    finish_run(run)

    print(f"logged smoke run for {config.name} at {git_sha(short=True)} in {args.mode} mode")


if __name__ == "__main__":
    main()
