import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
MANIFEST_DIR = ROOT / "data" / "manifests"

REQUIRED_COLUMNS = {
    "path",
    "dataset",
    "class",
    "recording_id",
    "duration",
    "sample_rate",
    "channels",
}

MANIFESTS = sorted(MANIFEST_DIR.glob("*_manifest.json"))

pytestmark = pytest.mark.skipif(
    not MANIFESTS, reason="no manifests yet; manifests are built in Phase 2"
)


@pytest.mark.parametrize("path", MANIFESTS, ids=lambda p: p.name)
def test_manifest_has_required_columns(path):
    rows = json.loads(path.read_text(encoding="utf-8"))["rows"]
    assert rows
    for row in rows:
        assert REQUIRED_COLUMNS <= set(row)


@pytest.mark.parametrize("path", MANIFESTS, ids=lambda p: p.name)
def test_manifest_paths_are_unique_and_durations_positive(path):
    rows = json.loads(path.read_text(encoding="utf-8"))["rows"]
    paths = [row["path"] for row in rows]
    assert len(paths) == len(set(paths))
    assert all(row["duration"] > 0 for row in rows)
    assert all(row["sample_rate"] > 0 for row in rows)


@pytest.mark.parametrize("path", MANIFESTS, ids=lambda p: p.name)
def test_recording_id_present_unless_dataset_is_split_unsafe(path):
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not payload["split_safe"]:
        pytest.skip(f"{payload['dataset']} carries no recording identity")
    assert all(row["recording_id"] for row in payload["rows"])
