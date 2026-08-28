from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any

from phase.config import REPO_ROOT, WandbConfig, to_dict


def git_sha(short: bool = False) -> str:
    args = ["git", "rev-parse"] + (["--short"] if short else []) + ["HEAD"]
    try:
        out = subprocess.run(
            args, cwd=REPO_ROOT, capture_output=True, text=True, check=True, timeout=10
        )
    except (OSError, subprocess.SubprocessError):
        return "unknown"
    return out.stdout.strip() or "unknown"


def git_dirty() -> bool:
    try:
        out = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            check=True,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        return False
    return bool(out.stdout.strip())


def init_run(config: Any, wandb_config: WandbConfig, job_type: str, name: str | None = None):
    import wandb

    payload = to_dict(config)
    payload["git_sha"] = git_sha()
    payload["git_dirty"] = git_dirty()

    run = wandb.init(
        project=wandb_config.project,
        entity=wandb_config.entity,
        mode=wandb_config.mode,
        group=wandb_config.group,
        tags=list(wandb_config.tags),
        job_type=job_type,
        name=name or getattr(config, "name", None),
        config=payload,
        dir=str(Path(REPO_ROOT / "runs")),
    )
    return run


def finish_run(run: Any) -> None:
    if run is not None:
        run.finish()
