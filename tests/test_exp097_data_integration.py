"""Integration test of exp097's pod data path on the real local competition data.

It runs the functions the pod runs (locate, verify, the YOLO dataset build,
densest_subset, build_repeated_subset) on the local copy of the competition
data, so the data path is exercised on real annotations rather than on a
synthetic stand-in.

Two substitutions keep the test local and download-free:
- Windows needs elevated rights to create symlinks, so `Path.symlink_to` writes an
  empty placeholder file. The pod runs Linux and makes the real symlinks.
- When shapely is not installed (it is not on this machine, and installing it
  would be a download), the build runs with polygon offset 0 through the same
  `prepare_yolo.build`. The test also asserts that `fetch_data.build_yolo` refuses
  loudly without shapely. That refusal is the pod's guard against the offset being
  skipped silently. The shapely offset itself already ran in the exp068 and exp073
  Kaggle kernels.
"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
COMPETITION = ROOT / "data" / "competition" / "unpacked"
HAS_SHAPELY = importlib.util.find_spec("shapely") is not None

pytestmark = pytest.mark.skipif(not COMPETITION.exists(),
                                reason="local competition data is not present")


@pytest.fixture
def placeholder_symlinks(monkeypatch):
    def create(self, target, target_is_directory=False):
        assert Path(target).exists(), f"symlink target missing: {target}"
        self.write_bytes(b"")
    monkeypatch.setattr(Path, "symlink_to", create)


def test_locate_and_verify_real_competition_data():
    from experiments.exp_097_runpod_yolo.src import fetch_data
    annotations, train, test = fetch_data.locate(COMPETITION)
    report = fetch_data.verify(annotations, train, test, expect_train=707, expect_test=180)
    assert report["annotations_sha256"] == fetch_data.ANNOTATIONS_SHA256
    assert (report["train_photographs"], report["test_photographs"]) == (707, 180)


def test_verify_fails_loudly_on_wrong_counts():
    from experiments.exp_097_runpod_yolo.src import fetch_data
    annotations, train, test = fetch_data.locate(COMPETITION)
    with pytest.raises(RuntimeError):
        fetch_data.verify(annotations, train, test, expect_train=706, expect_test=180)


@pytest.mark.skipif(HAS_SHAPELY, reason="only meaningful where shapely is absent")
def test_build_refuses_without_shapely(tmp_path):
    from experiments.exp_097_runpod_yolo.src import fetch_data
    annotations, train, _ = fetch_data.locate(COMPETITION)
    with pytest.raises(RuntimeError, match="shapely is required"):
        fetch_data.build_yolo(annotations, train, tmp_path / "yolo_ds", 0.5)
    assert not (tmp_path / "yolo_ds" / "data.yaml").exists()


def test_build_yolo_dataset_and_preflight_subsets(tmp_path, placeholder_symlinks):
    from experiments.exp_097_runpod_yolo.src import fetch_data, preflight
    from shared.data_split import make_split

    annotations, train, _ = fetch_data.locate(COMPETITION)
    split = make_split(str(annotations), val_fraction=fetch_data.VALIDATION_FRACTION)
    assert split.summary() == ("train: 601 observations / 974 records | "
                               "val: 106 observations / 180 records")

    output = tmp_path / "yolo_ds"
    if HAS_SHAPELY:
        report = fetch_data.build_yolo(annotations, train, output, 0.5)
    else:
        from experiments.exp_002_yolo_seg.src.prepare_yolo import build
        build(str(annotations), str(train), str(output), fetch_data.VALIDATION_FRACTION, 0.0)
        report = {}
        for fold in ("train", "val"):
            labels = sorted((output / "labels" / fold).glob("*.txt"))
            report[f"{fold}_images"] = len(list((output / "images" / fold).glob("*.jpeg")))
            report[f"{fold}_labels"] = len(labels)
            report[f"{fold}_instances"] = sum(fetch_data.count_instances(p) for p in labels)

    assert report["train_images"] == report["train_labels"]
    assert 900 < report["train_images"] <= 974
    assert report["val_images"] == report["val_labels"]
    assert 150 < report["val_images"] <= 180
    assert report["train_instances"] > report["train_images"]
    data_yaml = (output / "data.yaml").read_text()
    assert "train: images/train" in data_yaml and "0: filament" in data_yaml

    base = fetch_data.densest_subset(output, tmp_path / "base", 16)
    order = json.loads((tmp_path / "base" / "order.json").read_text())
    counts = [row["instances"] for row in order]
    assert base["samples"] == 16 and counts == sorted(counts, reverse=True)
    assert base["max_instances"] == counts[0] >= base["min_instances"] == counts[-1]

    subset_yaml = preflight.build_repeated_subset(tmp_path / "base", tmp_path / "subset", 40, 4)
    assert len(list((tmp_path / "subset" / "labels" / "train").glob("*.txt"))) == 40
    assert len(list((tmp_path / "subset" / "images" / "train").glob("*.jpeg"))) == 40
    # One full validation batch (Ultralytics validates with twice the training batch).
    assert len(list((tmp_path / "subset" / "images" / "val").glob("*.jpeg"))) == 4
    assert len(list((tmp_path / "subset" / "labels" / "val").glob("*.txt"))) == 4
    assert "val: images/val" in subset_yaml.read_text()
