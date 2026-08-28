from __future__ import annotations

import dataclasses
import types
import typing
from pathlib import Path
from typing import Any, TypeVar, Union, get_args, get_origin

import yaml

T = TypeVar("T")

INCLUDE_KEY = "_from_"
REPO_ROOT = Path(__file__).resolve().parents[2]


class ConfigError(ValueError):
    pass


def _is_dataclass_type(tp: Any) -> bool:
    return isinstance(tp, type) and dataclasses.is_dataclass(tp)


def _unwrap_optional(tp: Any) -> tuple[Any, bool]:
    origin = get_origin(tp)
    if origin is Union or origin is types.UnionType:
        args = [a for a in get_args(tp) if a is not type(None)]
        if len(args) == 1:
            return args[0], True
        raise ConfigError(f"unsupported union type {tp}")
    return tp, False


def _coerce(value: Any, tp: Any, path: str) -> Any:
    tp, optional = _unwrap_optional(tp)
    if value is None:
        if optional:
            return None
        raise ConfigError(f"{path}: null is not allowed for {tp}")

    if tp is Any:
        return value

    origin = get_origin(tp)
    if origin in (list, tuple):
        if not isinstance(value, (list, tuple)):
            raise ConfigError(f"{path}: expected a list, got {type(value).__name__}")
        (item_tp,) = get_args(tp)[:1] or (Any,)
        return [_coerce(v, item_tp, f"{path}[{i}]") for i, v in enumerate(value)]
    if origin is dict:
        if not isinstance(value, dict):
            raise ConfigError(f"{path}: expected a mapping, got {type(value).__name__}")
        key_tp, val_tp = get_args(tp) or (Any, Any)
        return {
            _coerce(k, key_tp, path): _coerce(v, val_tp, f"{path}.{k}") for k, v in value.items()
        }
    if origin is typing.Literal:
        allowed = get_args(tp)
        if value not in allowed:
            raise ConfigError(f"{path}: {value!r} is not one of {list(allowed)}")
        return value

    if _is_dataclass_type(tp):
        if not isinstance(value, dict):
            raise ConfigError(f"{path}: expected a mapping for {tp.__name__}")
        return from_dict(tp, value, path)

    if tp is Path:
        return Path(value)
    if tp is bool:
        if not isinstance(value, bool):
            raise ConfigError(f"{path}: expected a bool, got {value!r}")
        return value
    if tp is int:
        if isinstance(value, bool) or not isinstance(value, int):
            raise ConfigError(f"{path}: expected an int, got {value!r}")
        return value
    if tp is float:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ConfigError(f"{path}: expected a float, got {value!r}")
        return float(value)
    if tp is str:
        if not isinstance(value, str):
            raise ConfigError(f"{path}: expected a str, got {value!r}")
        return value
    return value


def from_dict(cls: type[T], data: dict[str, Any], path: str = "") -> T:
    if not _is_dataclass_type(cls):
        raise ConfigError(f"{cls} is not a dataclass")
    if not isinstance(data, dict):
        raise ConfigError(f"{path or cls.__name__}: expected a mapping")

    hints = typing.get_type_hints(cls)
    fields = {f.name: f for f in dataclasses.fields(cls)}
    unknown = set(data) - set(fields)
    if unknown:
        raise ConfigError(f"{path or cls.__name__}: unknown keys {sorted(unknown)}")

    kwargs: dict[str, Any] = {}
    for name, field in fields.items():
        prefix = f"{path}.{name}" if path else name
        if name in data:
            kwargs[name] = _coerce(data[name], hints[name], prefix)
        elif field.default is dataclasses.MISSING and field.default_factory is dataclasses.MISSING:
            raise ConfigError(f"{prefix}: missing required key")
    return cls(**kwargs)


def to_dict(obj: Any) -> Any:
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        return {f.name: to_dict(getattr(obj, f.name)) for f in dataclasses.fields(obj)}
    if isinstance(obj, (list, tuple)):
        return [to_dict(v) for v in obj]
    if isinstance(obj, dict):
        return {k: to_dict(v) for k, v in obj.items()}
    if isinstance(obj, Path):
        return str(obj).replace("\\", "/")
    return obj


def _merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def _resolve_includes(node: Any, origin: Path) -> Any:
    if isinstance(node, list):
        return [_resolve_includes(v, origin) for v in node]
    if not isinstance(node, dict):
        return node

    resolved = {k: _resolve_includes(v, origin) for k, v in node.items() if k != INCLUDE_KEY}
    if INCLUDE_KEY not in node:
        return resolved

    target = Path(node[INCLUDE_KEY])
    for candidate in (origin.parent / target, REPO_ROOT / target, target):
        if candidate.is_file():
            return _merge(load_yaml(candidate), resolved)
    raise ConfigError(f"{origin}: cannot resolve include {target}")


def load_yaml(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    if not path.is_file():
        path = REPO_ROOT / path
    if not path.is_file():
        raise ConfigError(f"config file not found: {path}")
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ConfigError(f"{path}: top level must be a mapping")
    return _resolve_includes(data, path.resolve())


def load_config(path: str | Path, cls: type[T], **overrides: Any) -> T:
    data = load_yaml(path)
    if overrides:
        data = _merge(data, overrides)
    return from_dict(cls, data)


def dump_config(obj: Any, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(to_dict(obj), handle, sort_keys=False)
    return path


FORBIDDEN_AUGMENTATIONS = {
    "pitch_shift",
    "time_warp",
    "freq_mask_tonal",
    "spec_augment_freq",
}


@dataclasses.dataclass(frozen=True)
class DatasetConfig:
    name: str
    root: Path
    sample_rate: int
    classes: list[str]
    split_safe: bool
    segment_seconds: float
    hop_seconds: float
    recording_id: str
    source: str = ""
    source_url: str = ""
    manifest: Path = Path("data/manifests")
    splits: Path = Path("data/splits")
    cache: Path = Path("data/cache")

    def __post_init__(self) -> None:
        if self.sample_rate <= 0:
            raise ConfigError("dataset.sample_rate must be positive")
        if not self.classes and self.split_safe:
            raise ConfigError("dataset.classes must be non-empty for a labelled dataset")
        if self.segment_seconds <= 0 or self.hop_seconds <= 0:
            raise ConfigError("dataset segment and hop must be positive")
        if self.hop_seconds > self.segment_seconds:
            raise ConfigError("dataset.hop_seconds must not exceed segment_seconds")


@dataclasses.dataclass(frozen=True)
class LogMelConfig:
    n_fft: int = 2048
    hop_length: int = 512
    n_mels: int = 128
    fmin: float = 10.0
    fmax: float = 8000.0
    power: float = 2.0
    log_offset: float = 1e-6


@dataclasses.dataclass(frozen=True)
class CQTConfig:
    bins_per_octave: int = 30
    n_bins: int = 180
    fmin: float = 10.0
    hop_length: int = 512


@dataclasses.dataclass(frozen=True)
class DemonConfig:
    band_low: float = 1000.0
    band_high: float = 30000.0
    envelope_lowpass: float = 500.0
    n_fft: int = 4096
    max_modulation_hz: float = 100.0


@dataclasses.dataclass(frozen=True)
class LofarConfig:
    frame_ms: float = 50.0
    shift_ms: float = 25.0
    window: str = "hann"
    tpsw_length: int = 51
    tpsw_gap: int = 8


@dataclasses.dataclass(frozen=True)
class FeatureConfig:
    name: typing.Literal["logmel", "cqt", "demon", "lofar"]
    normalize: typing.Literal["per_sample_zscore", "none"] = "per_sample_zscore"
    highpass_hz: float = 0.0
    logmel: LogMelConfig | None = None
    cqt: CQTConfig | None = None
    demon: DemonConfig | None = None
    lofar: LofarConfig | None = None

    def __post_init__(self) -> None:
        if getattr(self, self.name) is None:
            raise ConfigError(f"features.{self.name} block is required when name is {self.name!r}")

    @property
    def params(self) -> Any:
        return getattr(self, self.name)


@dataclasses.dataclass(frozen=True)
class AugmentOp:
    name: str
    prob: float = 1.0
    params: dict[str, Any] = dataclasses.field(default_factory=dict)

    def __post_init__(self) -> None:
        if not 0.0 <= self.prob <= 1.0:
            raise ConfigError(f"augment op {self.name}: prob must be in [0, 1]")
        if self.name in FORBIDDEN_AUGMENTATIONS:
            raise ConfigError(
                f"augment op {self.name} is identity-destroying and cannot be used as a positive"
            )


@dataclasses.dataclass(frozen=True)
class AugmentConfig:
    name: str
    family: typing.Literal["generic", "physics"]
    ops: list[AugmentOp]

    def __post_init__(self) -> None:
        if not self.ops:
            raise ConfigError("augment.ops must not be empty")


@dataclasses.dataclass(frozen=True)
class MocoConfig:
    backbone: typing.Literal["resnet18", "resnet34"] = "resnet18"
    dim: int = 128
    queue_size: int = 8192
    momentum: float = 0.999
    temperature: float = 0.2
    mlp_hidden: int = 512
    norm: typing.Literal["splitbn", "layernorm", "batchnorm"] = "splitbn"

    def __post_init__(self) -> None:
        if not 0.0 < self.momentum < 1.0:
            raise ConfigError("moco.momentum must be in (0, 1)")
        if self.temperature <= 0:
            raise ConfigError("moco.temperature must be positive")
        if self.queue_size <= 0:
            raise ConfigError("moco.queue_size must be positive")


@dataclasses.dataclass(frozen=True)
class OptimConfig:
    optimizer: typing.Literal["sgd", "adamw"] = "sgd"
    lr: float = 0.03
    momentum: float = 0.9
    weight_decay: float = 1e-4
    epochs: int = 200
    warmup_epochs: int = 5
    schedule: typing.Literal["cosine", "step", "constant"] = "cosine"
    batch_size: int = 128

    def __post_init__(self) -> None:
        if self.batch_size <= 0 or self.epochs <= 0:
            raise ConfigError("optim.batch_size and optim.epochs must be positive")
        if self.warmup_epochs >= self.epochs:
            raise ConfigError("optim.warmup_epochs must be smaller than optim.epochs")


@dataclasses.dataclass(frozen=True)
class DemonLossConfig:
    enabled: bool = False
    weight: float = 0.0
    band_low: float = 1000.0
    band_high: float = 30000.0

    def __post_init__(self) -> None:
        if self.enabled and self.weight <= 0:
            raise ConfigError("demon_loss.weight must be positive when the term is enabled")


@dataclasses.dataclass(frozen=True)
class WandbConfig:
    project: str = "phase"
    entity: str | None = None
    mode: typing.Literal["online", "offline", "disabled"] = "online"
    group: str | None = None
    tags: list[str] = dataclasses.field(default_factory=list)


@dataclasses.dataclass(frozen=True)
class PretrainConfig:
    name: str
    seed: int
    data: DatasetConfig
    features: FeatureConfig
    augment: AugmentConfig
    moco: MocoConfig = dataclasses.field(default_factory=MocoConfig)
    optim: OptimConfig = dataclasses.field(default_factory=OptimConfig)
    demon_loss: DemonLossConfig = dataclasses.field(default_factory=DemonLossConfig)
    wandb: WandbConfig = dataclasses.field(default_factory=WandbConfig)
    checkpoint_dir: Path = Path("checkpoints")
    checkpoint_every: int = 1
    resume: Path | None = None

    def __post_init__(self) -> None:
        if self.moco.queue_size % self.optim.batch_size:
            raise ConfigError("moco.queue_size must be a multiple of optim.batch_size")


@dataclasses.dataclass(frozen=True)
class EvalConfig:
    name: str
    protocol: typing.Literal["linear_probe", "finetune", "label_efficiency", "cross_dataset"]
    seeds: list[int]
    data: DatasetConfig
    features: FeatureConfig
    optim: OptimConfig = dataclasses.field(default_factory=OptimConfig)
    wandb: WandbConfig = dataclasses.field(default_factory=WandbConfig)
    checkpoint: Path | None = None
    label_fractions: list[float] = dataclasses.field(default_factory=lambda: [1.0])
    leakage_control: bool = True
    transfer_data: DatasetConfig | None = None

    def __post_init__(self) -> None:
        if len(self.seeds) < 3:
            raise ConfigError("every reported number needs at least three seeds")
        if any(not 0.0 < f <= 1.0 for f in self.label_fractions):
            raise ConfigError("eval.label_fractions must lie in (0, 1]")
        if self.protocol == "cross_dataset" and self.transfer_data is None:
            raise ConfigError("cross_dataset evaluation needs a transfer_data block")
