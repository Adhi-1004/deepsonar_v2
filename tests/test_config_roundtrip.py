from pathlib import Path

import pytest

from phase.config import (
    AugmentConfig,
    ConfigError,
    DatasetConfig,
    EvalConfig,
    FeatureConfig,
    PretrainConfig,
    dump_config,
    from_dict,
    load_config,
    load_yaml,
    to_dict,
)

ROOT = Path(__file__).resolve().parents[1]

CASES = [
    ("configs/data", DatasetConfig),
    ("configs/features", FeatureConfig),
    ("configs/augment", AugmentConfig),
    ("configs/augment/physics_ablation", AugmentConfig),
    ("configs/pretrain", PretrainConfig),
    ("configs/eval", EvalConfig),
]

ALL_CONFIGS = [
    pytest.param(path, cls, id=str(path.relative_to(ROOT)).replace("\\", "/"))
    for directory, cls in CASES
    for path in sorted((ROOT / directory).glob("*.yaml"))
]


@pytest.mark.parametrize("path, cls", ALL_CONFIGS)
def test_every_config_loads(path, cls):
    assert load_config(path, cls) is not None


@pytest.mark.parametrize("path, cls", ALL_CONFIGS)
def test_roundtrip_is_stable(path, cls, tmp_path):
    original = load_config(path, cls)
    dumped = dump_config(original, tmp_path / "roundtrip.yaml")
    reloaded = load_config(dumped, cls)
    assert to_dict(reloaded) == to_dict(original)


def test_includes_are_merged_and_overridden():
    merged = load_yaml(ROOT / "configs/pretrain/moco_physics_demon.yaml")
    assert merged["name"] == "moco_physics_demon"
    assert merged["augment"]["name"] == "physics"
    assert merged["demon_loss"]["enabled"] is True
    assert merged["optim"]["epochs"] == 200


def test_unknown_keys_are_rejected():
    with pytest.raises(ConfigError):
        from_dict(FeatureConfig, {"name": "logmel", "logmel": {}, "nmels": 128})


def test_forbidden_augmentation_is_rejected():
    with pytest.raises(ConfigError):
        from_dict(
            AugmentConfig,
            {"name": "bad", "family": "physics", "ops": [{"name": "pitch_shift", "prob": 0.5}]},
        )


def test_fewer_than_three_seeds_is_rejected():
    data = load_yaml(ROOT / "configs/eval/linear_probe.yaml")
    data["seeds"] = [0]
    with pytest.raises(ConfigError):
        from_dict(EvalConfig, data)


def test_queue_size_must_be_multiple_of_batch_size():
    data = load_yaml(ROOT / "configs/pretrain/moco_generic.yaml")
    data["moco"]["queue_size"] = 8191
    with pytest.raises(ConfigError):
        from_dict(PretrainConfig, data)


def test_ds3500_is_marked_split_unsafe():
    assert (
        load_config(ROOT / "configs/data/shipsear_ds3500.yaml", DatasetConfig).split_safe is False
    )
