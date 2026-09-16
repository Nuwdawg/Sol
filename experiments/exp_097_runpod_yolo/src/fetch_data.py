"""Bring the competition data onto the pod and build the YOLO dataset, verifying both.

Data path: Kaggle -> pod, over the pod's own connection. Nothing touches the
user's PC. The archive is about 0.7 GB.

Checks that fail loudly rather than training on the wrong thing:
- the annotation file's SHA256 equals the one every Kaggle kernel in this project
  asserts;
- exactly 707 training and 180 test photographs are present;
- `shapely` imports, because `prepare_yolo.shrink` silently skips the half-pixel
  polygon offset when it is missing, and that offset is the target correction the
  0.39 pipeline was trained with;
- the canonical photograph-grouped split is rebuilt by the same `make_split` the
  validation scorer uses, with the same 0.15 validation fraction.

It also builds the preflight's worst-case subset: the most instance-dense training
samples, where mask-loss memory peaks.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import subprocess
import zipfile

from experiments.exp_097_runpod_yolo.src.events import emit
from experiments.exp_097_runpod_yolo.src.kaggle_sync import sha256

COMPETITION = "filament-segmentation-2026"
ANNOTATIONS_NAME = "MAGFiLO_1.0_Annotations_kaggle2026_train.json"
ANNOTATIONS_SHA256 = "5da9e92b5a1a1947fd5d57adb6688269625c48ec1ef884daf2a01618c9ed54a1"
VALIDATION_FRACTION = 0.15   # identical to prepare_yolo's default and records_for's split


def download(competition: str, destination: Path) -> Path:
    destination.mkdir(parents=True, exist_ok=True)
    archives = sorted(destination.glob("*.zip"))
    if archives:
        return archives[0]
    result = subprocess.run(["kaggle", "competitions", "download", competition,
                             "-p", str(destination), "-q"],
                            capture_output=True, text=True, timeout=3600)
    if result.returncode != 0:
        raise RuntimeError(f"competition download failed: {(result.stdout + result.stderr)[-400:]}")
    archives = sorted(destination.glob("*.zip"))
    if not archives:
        raise RuntimeError(f"no archive appeared in {destination}")
    return archives[0]


def extract(archive: Path, target: Path) -> Path:
    marker = target / ".extracted"
    if marker.exists():
        return target
    target.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive) as bundle:
        bundle.extractall(target)
    marker.write_text(archive.name)
    return target


def locate(root: Path) -> tuple[Path, Path, Path]:
    annotations = next(root.rglob(ANNOTATIONS_NAME), None)
    if annotations is None:
        raise RuntimeError(f"{ANNOTATIONS_NAME} not found under {root}")
    train = annotations.parent / "train_images"
    test = next((p for p in root.rglob("test_images") if p.is_dir()), None)
    if not train.is_dir() or test is None:
        raise RuntimeError("train_images or test_images directory missing")
    return annotations, train, test


def verify(annotations: Path, train: Path, test: Path, expect_train: int,
           expect_test: int) -> dict:
    digest = sha256(annotations)
    if digest != ANNOTATIONS_SHA256:
        raise RuntimeError(f"annotation hash mismatch: {digest}")
    train_count = len(list(train.glob("*.jpeg")))
    test_count = len(list(test.glob("*.jpeg")))
    if (train_count, test_count) != (expect_train, expect_test):
        raise RuntimeError(f"photograph counts {train_count}/{test_count}, "
                           f"expected {expect_train}/{expect_test}")
    return dict(annotations_sha256=digest, train_photographs=train_count,
                test_photographs=test_count)


def count_instances(label: Path) -> int:
    return sum(1 for line in label.read_text().splitlines() if line.strip())


def build_yolo(annotations: Path, train: Path, output: Path, polygon_offset: float) -> dict:
    try:
        import shapely  # noqa: F401
    except ImportError as error:
        raise RuntimeError("shapely is required; without it the polygon offset is silently "
                           "skipped") from error
    from experiments.exp_002_yolo_seg.src.prepare_yolo import build
    if not (output / "data.yaml").exists():
        build(str(annotations), str(train), str(output), VALIDATION_FRACTION, polygon_offset)
    report = dict(data_yaml=str(output / "data.yaml"), polygon_offset=polygon_offset,
                  validation_fraction=VALIDATION_FRACTION)
    for split in ("train", "val"):
        labels = sorted((output / "labels" / split).glob("*.txt"))
        report[f"{split}_images"] = len(list((output / "images" / split).glob("*.jpeg")))
        report[f"{split}_labels"] = len(labels)
        report[f"{split}_instances"] = sum(count_instances(label) for label in labels)
    if report["train_images"] != report["train_labels"] or report["train_images"] == 0:
        raise RuntimeError(f"YOLO dataset is inconsistent: {report}")
    return report


def densest_subset(dataset: Path, target: Path, count: int) -> dict:
    """The `count` training samples with the most instances, for worst-case memory."""
    labels = sorted((dataset / "labels" / "train").glob("*.txt"),
                    key=lambda path: (-count_instances(path), path.name))[:count]
    if target.exists():
        shutil.rmtree(target)
    (target / "images").mkdir(parents=True)
    (target / "labels").mkdir(parents=True)
    order = []
    for label in labels:
        image = dataset / "images" / "train" / f"{label.stem}.jpeg"
        (target / "images" / image.name).symlink_to(image.resolve())
        shutil.copy2(label, target / "labels" / label.name)
        order.append(dict(stem=label.stem, instances=count_instances(label)))
    (target / "order.json").write_text(json.dumps(order, indent=2))
    return dict(path=str(target), samples=len(order),
                max_instances=order[0]["instances"] if order else 0,
                min_instances=order[-1]["instances"] if order else 0)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workdir", type=Path, required=True)
    parser.add_argument("--competition", default=COMPETITION)
    parser.add_argument("--polygon-offset", type=float, default=0.5)
    parser.add_argument("--expect-train", type=int, default=707)
    parser.add_argument("--expect-test", type=int, default=180)
    parser.add_argument("--subset-count", type=int, default=64)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--log", type=Path, default=None)
    args = parser.parse_args()

    archive = download(args.competition, args.workdir / "download")
    root = extract(archive, args.workdir / "competition")
    annotations, train, test = locate(root)
    report = dict(archive=archive.name, archive_bytes=archive.stat().st_size,
                  **verify(annotations, train, test, args.expect_train, args.expect_test))
    report["yolo"] = build_yolo(annotations, train, args.workdir / "yolo_ds", args.polygon_offset)
    report["preflight_subset"] = densest_subset(args.workdir / "yolo_ds",
                                                args.workdir / "preflight_base",
                                                args.subset_count)
    report["paths"] = dict(annotations=str(annotations), train_images=str(train),
                           test_images=str(test))
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(json.dumps(report, indent=2))
    emit("DATA", report, args.log)


if __name__ == "__main__":
    main()
