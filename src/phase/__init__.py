from __future__ import annotations

import importlib
from typing import Any

__version__ = "0.1.0"

_LAZY = {
    "AugmentConfig": "phase.config",
    "ConfigError": "phase.config",
    "DatasetConfig": "phase.config",
    "EvalConfig": "phase.config",
    "FeatureConfig": "phase.config",
    "PretrainConfig": "phase.config",
    "dump_config": "phase.config",
    "load_config": "phase.config",
    "load_yaml": "phase.config",
    "to_dict": "phase.config",
    "set_seed": "phase.seed",
    "init_run": "phase.tracking",
    "finish_run": "phase.tracking",
    "git_sha": "phase.tracking",
}

__all__ = sorted(_LAZY)


def __getattr__(name: str) -> Any:
    if name not in _LAZY:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    return getattr(importlib.import_module(_LAZY[name]), name)


def __dir__() -> list[str]:
    return sorted(set(__all__) | set(globals()))
