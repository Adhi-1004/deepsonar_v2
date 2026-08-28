import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SPLIT_DIR = ROOT / "data" / "splits"

SPLIT_FILES = sorted(SPLIT_DIR.glob("*.json"))

pytestmark = pytest.mark.skipif(
    not SPLIT_FILES, reason="no split files yet; splits are built in Phase 2"
)


@pytest.mark.parametrize("path", SPLIT_FILES, ids=lambda p: p.name)
def test_recording_ids_do_not_straddle_splits(path):
    payload = json.loads(path.read_text(encoding="utf-8"))
    for fold_name, fold in payload["folds"].items():
        groups = {name: {row["recording_id"] for row in rows} for name, rows in fold.items()}
        for a, b in (("train", "val"), ("train", "test"), ("val", "test")):
            overlap = groups[a] & groups[b]
            assert not overlap, (
                f"{path.name}:{fold_name} {a}/{b} share recordings {sorted(overlap)}"
            )


@pytest.mark.parametrize("path", SPLIT_FILES, ids=lambda p: p.name)
def test_every_split_covers_every_class(path):
    payload = json.loads(path.read_text(encoding="utf-8"))
    classes = set(payload["classes"])
    for fold_name, fold in payload["folds"].items():
        for name, rows in fold.items():
            present = {row["class"] for row in rows}
            assert present == classes, (
                f"{path.name}:{fold_name}:{name} is missing {classes - present}"
            )


@pytest.mark.parametrize("path", SPLIT_FILES, ids=lambda p: p.name)
def test_split_is_marked_split_safe(path):
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["split_safe"] is True


WINDOW_FILES = sorted((ROOT / "data" / "cache").glob("*_windows.json"))


@pytest.mark.skipif(not WINDOW_FILES, reason="no window index yet; segmentation is Phase 3")
@pytest.mark.parametrize("path", WINDOW_FILES, ids=lambda p: p.name)
def test_windows_from_one_recording_never_straddle_a_split(path):
    payload = json.loads(path.read_text(encoding="utf-8"))
    groups: dict[str, set[str]] = {}
    for window in payload["windows"]:
        groups.setdefault(window["recording_id"], set()).add(window["split"])
    straddling = {k: sorted(v) for k, v in groups.items() if len(v) > 1}
    assert not straddling, f"{path.name}: recordings span splits {list(straddling)[:5]}"


@pytest.mark.skipif(not WINDOW_FILES, reason="no window index yet; segmentation is Phase 3")
@pytest.mark.parametrize("path", WINDOW_FILES, ids=lambda p: p.name)
def test_overlapping_windows_from_one_file_stay_in_one_split(path):
    payload = json.loads(path.read_text(encoding="utf-8"))
    files: dict[str, set[str]] = {}
    for window in payload["windows"]:
        files.setdefault(window["path"], set()).add(window["split"])
    straddling = {k: sorted(v) for k, v in files.items() if len(v) > 1}
    assert not straddling, f"{path.name}: files span splits {list(straddling)[:5]}"


@pytest.mark.skipif(not WINDOW_FILES, reason="no window index yet; segmentation is Phase 3")
@pytest.mark.parametrize("path", WINDOW_FILES, ids=lambda p: p.name)
def test_windows_lie_inside_their_source_file(path):
    payload = json.loads(path.read_text(encoding="utf-8"))
    frame = payload["frame"]
    assert all(w["frames"] == frame for w in payload["windows"])
    assert all(w["start"] >= 0 for w in payload["windows"])
