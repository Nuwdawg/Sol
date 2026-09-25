"""Build the YOLO dataset with preprocessing and one annotator per training photograph.

Two changes from `exp_002_yolo_seg/prepare_yolo.build`, which every detector in this
repository has been trained with.

**1. One pass per training photograph.** The existing builder emits one sample per
COCO image_id, symlinking the shared photograph under each record's name. Measured
on this release: 974 training image_ids cover **601 unique photographs**, a 1.62x
duplication, with 124 photographs present three times and 125 twice. In 215 of the
249 multi-pass photographs the annotators disagree on how many filaments exist -
one is labelled 8, 14 and 21 times by three people. The detector is therefore
trained to predict three different instance counts for one identical image, which
attacks recognition directly. Reche & Cid report the same finding on this dataset.

`--dedup` chooses which pass survives, and the choice is measured rather than
assumed:

- `none`   - current behaviour, every pass kept (the control)
- `median` - the pass whose instance count is the median of its photograph's passes
- `most`   - the most liberal annotator
- `fewest` - the most conservative annotator

**2. Preprocessing.** Limb-darkening flattened, histogram-matched to a frozen
reference built from training photographs only, off-disk zeroed. Measured on twelve
photographs, this flips filament-versus-disk contrast from -3 to -20 counts (the
limb annulus is darker than the filaments) to +18 to +69. See `preprocess.py`.

**The validation fold is never deduplicated.** `make_split` and the frozen
`BASELINE_PQ = 0.4558638687` depend on all 180 validation records existing, and
every comparison in this repository is against that number. Training labels change;
the yardstick does not.

Polygon handling is delegated to the original module unchanged - `merge_multi_segment`,
`shrink` and `write_label` - because exp002 measured that an unguarded half-pixel
buffer splits 42 of 8199 instances, and that correction is not being revisited here.
An image whose annotations yield no polygon is skipped entirely, image included,
exactly as the original does.
"""
from __future__ import annotations

import argparse
import json
import shutil
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

from experiments.exp_002_yolo_seg.src.prepare_yolo import (
    merge_multi_segment, shrink, write_label)
from experiments.exp_128_detector_retrain.src.preprocess import (
    build_reference, histogram_of, prepare, preprocess, save_reference)
from shared.data_split import assert_disjoint, make_split


def choose_pass(image_ids: list[int], counts: dict[int, int], rule: str) -> list[int]:
    """Which annotator pass survives for one photograph."""
    if rule == "none" or len(image_ids) == 1:
        return list(image_ids)
    ordered = sorted(image_ids, key=lambda i: (counts.get(i, 0), i))
    if rule == "most":
        return [ordered[-1]]
    if rule == "fewest":
        return [ordered[0]]
    if rule == "median":
        return [ordered[len(ordered) // 2]]
    raise ValueError(f"unknown dedup rule: {rule}")


def polygons_for(annotations: list[dict], polygon_offset: float) -> list[np.ndarray]:
    """The original builder's conversion, reused verbatim rather than reimplemented."""
    polygons: list[np.ndarray] = []
    for annotation in annotations:
        segmentation = annotation.get("segmentation")
        # An RLE-encoded annotation has no polygon to convert; the dataset is
        # polygon-based, so this is a guard rather than an expected path.
        if not segmentation or isinstance(segmentation, dict):
            continue
        rings = [s for s in segmentation if len(s) >= 6]
        if not rings:
            continue
        polygons.extend(shrink(merge_multi_segment(rings), polygon_offset))
    return polygons


def build(annotations: Path, images_dir: Path, output: Path, val_fraction: float,
          polygon_offset: float, dedup: str, preprocess_images: bool,
          limit: int = 0, reference_sample: int = 150) -> dict:
    coco = json.loads(Path(annotations).read_text(encoding="utf-8"))
    meta = {record["id"]: record for record in coco["images"]}
    by_image: dict[int, list[dict]] = defaultdict(list)
    for annotation in coco["annotations"]:
        by_image[annotation["image_id"]].append(annotation)
    counts = {image_id: len(rows) for image_id, rows in by_image.items()}

    split = make_split(str(annotations), val_fraction=val_fraction)
    assert_disjoint(split)
    print(split.summary(), flush=True)
    fold_of = {i: "train" for i in split.train_image_ids}
    fold_of.update({i: "val" for i in split.val_image_ids})

    # Deduplicate the training fold only. Validation keeps every record.
    by_file: dict[str, list[int]] = defaultdict(list)
    for image_id in split.train_image_ids:
        by_file[meta[image_id]["file_name"]].append(image_id)
    keep = set(split.val_image_ids)
    for image_ids in by_file.values():
        keep.update(choose_pass(image_ids, counts, dedup))
    kept_train = len(keep) - len(split.val_image_ids)
    print(f"dedup={dedup}: kept {kept_train} of {len(split.train_image_ids)} training "
          f"records over {len(by_file)} photographs, dropped "
          f"{len(split.train_image_ids) - kept_train}", flush=True)

    root = Path(output)
    for fold in ("train", "val"):
        (root / "images" / fold).mkdir(parents=True, exist_ok=True)
        (root / "labels" / fold).mkdir(parents=True, exist_ok=True)

    ordered = [i for i in meta if fold_of.get(i) and i in keep]
    if limit:
        ordered = ordered[:limit]

    reference = None
    if preprocess_images:
        # Built from TRAINING photographs only, then frozen for every fold.
        # Sampled rather than exhaustive: the mean histogram converges long before
        # 601 photographs, and each one costs a disk fit plus a radial profile.
        # The smoke test measured ~3 s per photograph, so an exhaustive pass would
        # add roughly 25 minutes of pod time for no measurable change in the CDF.
        histograms, seen = [], set()
        for image_id in ordered:
            if fold_of[image_id] != "train":
                continue
            if reference_sample and len(histograms) >= reference_sample:
                break
            name = meta[image_id]["file_name"]
            if name in seen:
                continue
            seen.add(name)
            raw = cv2.imread(str(Path(images_dir) / name), cv2.IMREAD_GRAYSCALE)
            if raw is None:
                continue
            levels, inner = prepare(raw)
            if inner.sum() and levels[inner > 0].std() >= 1:
                histograms.append(histogram_of(levels, inner))
        if not histograms:
            raise SystemExit("no usable training photographs for the reference")
        reference = build_reference(histograms)
        save_reference(reference, root / "reference_cdf.json")
        reference_photographs = len(histograms)
        print(f"reference CDF built from {reference_photographs} photographs "
              f"(sample cap {reference_sample or 'none'})", flush=True)
    else:
        reference_photographs = 0

    written = {"train": 0, "val": 0}
    instances = {"train": 0, "val": 0}
    empty = 0
    cache: dict[str, np.ndarray] = {}
    for image_id in ordered:
        record = meta[image_id]
        fold = fold_of[image_id]
        width = record.get("width") or 2048
        height = record.get("height") or 2048

        # Convert first: an image whose annotations yield no polygon is skipped
        # entirely, image included, exactly as the original builder does.
        polygons = polygons_for(by_image.get(image_id, []), polygon_offset)
        if not polygons:
            empty += 1
            continue

        name = record["file_name"]
        source = Path(images_dir) / name
        if not source.is_file():
            raise SystemExit(f"missing image referenced by annotations: {source}")
        destination = root / "images" / fold / f"{image_id}.jpeg"
        if preprocess_images:
            if name not in cache:
                raw = cv2.imread(str(source), cv2.IMREAD_GRAYSCALE)
                if raw is None:
                    raise SystemExit(f"unreadable image: {source}")
                cache[name] = preprocess(raw, reference)
                if len(cache) > 16:
                    cache.pop(next(iter(cache)))
            cv2.imwrite(str(destination), cache[name])
        else:
            shutil.copy2(source, destination)

        count = write_label(root / "labels" / fold / f"{image_id}.txt",
                            polygons, width, height)
        written[fold] += 1
        instances[fold] += count

    yaml = root / "data.yaml"
    yaml.write_text("\n".join([
        f"path: {root.resolve()}", "train: images/train", "val: images/val",
        "names:", "  0: filament", ""]), encoding="utf-8")
    report = dict(data_yaml=str(yaml), dedup=dedup, preprocessed=preprocess_images,
                  reference_photographs=reference_photographs,
                  reference_sample=reference_sample,
                  polygon_offset=polygon_offset,
                  train_images=written["train"], val_images=written["val"],
                  train_instances=instances["train"], val_instances=instances["val"],
                  photographs=len(by_file), skipped_empty=empty,
                  dropped_records=len(split.train_image_ids) - kept_train)
    (root / "build_report.json").write_text(json.dumps(report, indent=2))
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--images", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--val-fraction", type=float, default=0.15)
    parser.add_argument("--polygon-offset", type=float, default=0.5)
    parser.add_argument("--dedup", choices=("none", "median", "most", "fewest"),
                        default="median")
    parser.add_argument("--no-preprocess", action="store_true")
    parser.add_argument("--limit", type=int, default=0,
                        help="build only the first N records, for a smoke test")
    parser.add_argument("--reference-sample", type=int, default=150,
                        help="photographs used to build the frozen reference CDF; "
                             "0 uses every training photograph")
    args = parser.parse_args()
    report = build(args.annotations, args.images, args.output, args.val_fraction,
                   args.polygon_offset, args.dedup, not args.no_preprocess, args.limit,
                   args.reference_sample)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
