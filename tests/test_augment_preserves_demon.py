import numpy as np
import pytest
import torch
from scipy.signal import butter, resample_poly, sosfiltfilt

from phase.augment import Pipeline, build_op
from phase.config import AugmentConfig, AugmentOp, DemonConfig, FeatureConfig, load_config
from phase.features import demon

SAMPLE_RATE = 32000
SECONDS = 8.0
SHAFT_HZ = 6.2
BLADES = 4
BLADE_HZ = SHAFT_HZ * BLADES

TOLERANCE_HZ = 1.5


def cavitation_signal(seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    t = np.arange(int(SAMPLE_RATE * SECONDS)) / SAMPLE_RATE

    carrier = rng.normal(0.0, 1.0, t.size)
    sos = butter(4, [2000, 12000], btype="bandpass", fs=SAMPLE_RATE, output="sos")
    carrier = sosfiltfilt(sos, carrier)

    envelope = 1 + 0.6 * np.sin(2 * np.pi * SHAFT_HZ * t) + 0.4 * np.sin(2 * np.pi * BLADE_HZ * t)
    signal = carrier * envelope
    return (signal / np.abs(signal).max() * 0.5).astype(np.float32)


def demon_config() -> DemonConfig:
    return load_config("configs/features/demon.yaml", FeatureConfig).params


def detected_peaks(waveform: np.ndarray) -> np.ndarray:
    config = demon_config()
    spectrum = demon.compute(waveform, SAMPLE_RATE, config)
    return demon.peak_frequencies(spectrum, SAMPLE_RATE, config, top=6)


def nearest(peaks: np.ndarray, target: float) -> float:
    if peaks.size == 0:
        return float("inf")
    return float(np.min(np.abs(peaks - target)))


def physics_ops() -> list[AugmentOp]:
    config = load_config("configs/augment/physics.yaml", AugmentConfig)
    return [op for op in config.ops if op.name not in {"random_crop", "time_mask"}]


def test_the_source_signal_has_the_modulation_we_injected():
    peaks = detected_peaks(cavitation_signal())
    assert nearest(peaks, SHAFT_HZ) <= TOLERANCE_HZ, f"shaft rate not found in {peaks}"
    assert nearest(peaks, BLADE_HZ) <= TOLERANCE_HZ, f"blade rate not found in {peaks}"


@pytest.mark.parametrize("op", physics_ops(), ids=lambda op: op.name)
def test_each_augmentation_preserves_demon_peaks(op):
    forced = AugmentOp(name=op.name, prob=1.0, params=op.params)
    augmentation = build_op(forced)

    clean = cavitation_signal()
    generator = torch.Generator().manual_seed(0)
    batch = torch.from_numpy(clean).unsqueeze(0)
    augmented = augmentation(batch, SAMPLE_RATE, generator)[0].numpy()

    assert np.isfinite(augmented).all(), f"{op.name} produced non-finite samples"

    peaks = detected_peaks(augmented)
    assert nearest(peaks, SHAFT_HZ) <= TOLERANCE_HZ, (
        f"{op.name} moved or destroyed the {SHAFT_HZ} Hz shaft line; found {peaks}"
    )
    assert nearest(peaks, BLADE_HZ) <= TOLERANCE_HZ, (
        f"{op.name} moved or destroyed the {BLADE_HZ} Hz blade line; found {peaks}"
    )


def test_the_whole_physics_pipeline_preserves_demon_peaks():
    config = load_config("configs/augment/physics.yaml", AugmentConfig)
    pipeline = Pipeline(config)
    generator = torch.Generator().manual_seed(0)

    batch = torch.from_numpy(cavitation_signal()).unsqueeze(0)
    first, second = pipeline.views(batch, SAMPLE_RATE, generator)

    for label, view in (("view 1", first), ("view 2", second)):
        peaks = detected_peaks(view[0].numpy())
        assert nearest(peaks, SHAFT_HZ) <= TOLERANCE_HZ, f"{label} lost the shaft line: {peaks}"
        assert nearest(peaks, BLADE_HZ) <= TOLERANCE_HZ, f"{label} lost the blade line: {peaks}"


def test_a_forbidden_transform_fails_the_same_check():
    clean = cavitation_signal()
    shifted = resample_poly(clean, 3, 4).astype(np.float32)

    peaks = detected_peaks(shifted)
    moved = nearest(peaks, SHAFT_HZ) > TOLERANCE_HZ or nearest(peaks, BLADE_HZ) > TOLERANCE_HZ
    assert moved, (
        "a heavy pitch shift left the DEMON peaks intact, so this test cannot "
        f"distinguish valid positives from invalid ones; found {peaks}"
    )


def test_forbidden_augmentations_cannot_be_configured():
    from phase.config import ConfigError

    with pytest.raises(ConfigError):
        AugmentOp(name="pitch_shift", prob=1.0, params={})
