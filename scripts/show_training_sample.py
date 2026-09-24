"""Render what the model is actually trained on, as images a person can look at.

Three things differ between "the competition data" and "the training data", and
each one matters when judging a prediction:

1. A photograph is annotated by several people. The metric scores a prediction
   against **one** annotator's record at a time, so the same pixels are both
   right and wrong depending on the record. The last figure puts every
   annotator's polygons for one photograph side by side.
2. The training labels are not the annotators' polygons. `prepare_yolo.shrink`
   pulls every ring inward by half a pixel before writing the YOLO label, which
   is the correction exp_006 measured. The third panel draws that label.
3. Filament boundaries fade over several pixels. At 2048 the disagreement lives
   in a band a few pixels wide, invisible in a thumbnail, so each sample also
   gets a full-resolution crop.

Usage:
    python scripts/show_training_sample.py
    python scripts/show_training_sample.py --count 6 --output data/training_sample
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path

import cv2
import numpy as np

from experiments.exp_002_yolo_seg.src.prepare_yolo import (
    DEFAULT_POLYGON_OFFSET, merge_multi_segment, shrink)
from shared.data_split import make_split

ROOT = Path(__file__).resolve().parents[1]
ANNOTATIONS = ROOT / "data" / "competition" / "unpacked" / "MAGFiLO_1.0_Kaggle_2026" / \
    "train" / "MAGFiLO_1.0_Annotations_kaggle2026_train.json"
IMAGES = ANNOTATIONS.parent / "train_images"
NATIVE = 2048
COLOURS = [(60, 220, 60), (60, 160, 255), (255, 160, 60), (220, 60, 220),
           (60, 230, 230), (200, 200, 60)]


def rings_of(annotation: dict) -> list[list[float]]:
    segmentation = annotation.get("segmentation")
    if not segmentation or isinstance(segmentation, dict):
        return []
    return [ring for ring in segmentation if len(ring) >= 6]


def as_contours(rings: list[list[float]]) -> list[np.ndarray]:
    return [np.asarray(ring, np.float64).reshape(-1, 2).round().astype(np.int32)
            for ring in rings]


def label(image: np.ndarray, text: str) -> np.ndarray:
    strip = np.zeros((34, image.shape[1], 3), np.uint8)
    cv2.putText(strip, text, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (255, 255, 255), 1,
                cv2.LINE_AA)
    return np.vstack([strip, image])


def outline(canvas: np.ndarray, contours: list[np.ndarray], colour, thickness=3) -> None:
    cv2.polylines(canvas, contours, True, colour, thickness, cv2.LINE_AA)


def filled(canvas: np.ndarray, contours: list[np.ndarray], colour, alpha=0.35) -> None:
    overlay = canvas.copy()
    cv2.fillPoly(overlay, contours, colour)
    cv2.addWeighted(overlay, alpha, canvas, 1 - alpha, 0, canvas)


def crop_around(photograph: np.ndarray, contours: list[np.ndarray], size: int) -> np.ndarray:
    """A full-resolution window on the largest filament, where the boundary is visible."""
    largest = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(largest)
    cx, cy = x + w // 2, y + h // 2
    half = size // 2
    cx = int(np.clip(cx, half, NATIVE - half))
    cy = int(np.clip(cy, half, NATIVE - half))
    return photograph[cy - half:cy + half, cx - half:cx + half].copy()


def render_record(photograph: np.ndarray, annotations: list[dict], offset: float,
                  crop_size: int) -> tuple[np.ndarray, np.ndarray, dict]:
    raw = photograph.copy()
    human = photograph.copy()
    training = photograph.copy()
    drawn, shrunk_total, areas = [], 0, []
    for index, annotation in enumerate(annotations):
        rings = rings_of(annotation)
        if not rings:
            continue
        contours = as_contours(rings)
        drawn.extend(contours)
        areas.append(sum(abs(cv2.contourArea(c)) for c in contours))
        filled(human, contours, COLOURS[index % len(COLOURS)])
        outline(human, contours, COLOURS[index % len(COLOURS)], 2)
        merged = merge_multi_segment(rings)
        for polygon in shrink(merged, offset):
            contour = polygon.round().astype(np.int32).reshape(-1, 2)
            outline(training, [contour], (60, 220, 60), 2)
            shrunk_total += 1
    try:
        import shapely  # noqa: F401
        note = f"training label, rings pulled in {offset} px"
    except ImportError:
        # shrink() needs shapely and silently returns the ring unchanged without it,
        # so the panel must not claim an offset that was never applied.
        note = f"shrink SKIPPED (no shapely here); the pod applies {offset} px"
    panels = [label(raw, "photograph, native 2048"),
              label(human, f"annotator polygons: {len(drawn)} instances"),
              label(training, f"{note}: {shrunk_total} polygons")]
    crop = crop_around(photograph, drawn, crop_size) if drawn else photograph[:crop_size, :crop_size]
    crop_human = crop.copy()
    shift = None
    if drawn:
        largest = max(drawn, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest)
        half = crop_size // 2
        cx = int(np.clip(x + w // 2, half, NATIVE - half))
        cy = int(np.clip(y + h // 2, half, NATIVE - half))
        shift = (cx - half, cy - half)
        moved = [c - np.asarray(shift, np.int32) for c in drawn]
        outline(crop_human, moved, (60, 160, 255), 1)
    detail = np.hstack([label(crop, f"crop {crop_size} px, unmarked"),
                        label(crop_human, "same crop with the annotator's boundary")])
    summary = dict(instances=len(drawn), shrunk_polygons=shrunk_total,
                   mean_instance_pixels=round(float(np.mean(areas)), 1) if areas else 0.0,
                   crop_origin=shift)
    return np.hstack(panels), detail, summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--annotations", type=Path, default=ANNOTATIONS)
    parser.add_argument("--images", type=Path, default=IMAGES)
    parser.add_argument("--output", type=Path, default=ROOT / "data" / "training_sample")
    parser.add_argument("--count", type=int, default=4, help="photographs of varying density")
    parser.add_argument("--offset", type=float, default=DEFAULT_POLYGON_OFFSET)
    parser.add_argument("--crop", type=int, default=520)
    parser.add_argument("--width", type=int, default=1500, help="width of each saved figure")
    args = parser.parse_args()

    coco = json.loads(args.annotations.read_text(encoding="utf-8"))
    meta = {record["id"]: record for record in coco["images"]}
    by_record = defaultdict(list)
    for annotation in coco["annotations"]:
        by_record[annotation["image_id"]].append(annotation)
    by_photograph = defaultdict(list)
    for image_id, record in meta.items():
        by_photograph[record["file_name"]].append(image_id)

    split = make_split(str(args.annotations))
    fold_of = {image_id: "train" for image_id in split.train_image_ids}
    fold_of.update({image_id: "val" for image_id in split.val_image_ids})

    counted = sorted(((len(by_record[i]), i) for i in meta), reverse=True)
    picks = []
    if counted:
        picks.append(("densest", counted[0][1]))
        picks.append(("median", counted[len(counted) // 2][1]))
        non_empty = [pair for pair in counted if pair[0] > 0]
        picks.append(("sparsest", non_empty[-1][1]))
        for _, image_id in counted:
            if len(by_photograph[meta[image_id]["file_name"]]) >= 3:
                picks.append(("multi-annotator", image_id))
                break
    picks = picks[:max(args.count, 3)]

    args.output.mkdir(parents=True, exist_ok=True)
    report = []
    for kind, image_id in picks:
        record = meta[image_id]
        path = args.images / record["file_name"]
        grey = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if grey is None:
            raise SystemExit(f"cannot read {path}")
        photograph = np.repeat(grey[..., None], 3, axis=2)
        figure, detail, summary = render_record(photograph, by_record[image_id], args.offset,
                                                args.crop)
        scale = args.width * 3 / figure.shape[1]
        figure = cv2.resize(figure, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        stem = f"{kind}_{Path(record['file_name']).stem}_{image_id}"
        cv2.imwrite(str(args.output / f"{stem}.jpg"), figure, [cv2.IMWRITE_JPEG_QUALITY, 92])
        cv2.imwrite(str(args.output / f"{stem}_detail.jpg"), detail,
                    [cv2.IMWRITE_JPEG_QUALITY, 95])
        report.append(dict(kind=kind, image_id=image_id, photograph=record["file_name"],
                           fold=fold_of.get(image_id, "unused"),
                           annotator_records_for_this_photograph=len(
                               by_photograph[record["file_name"]]),
                           **summary))
        print(f"{kind:16s} {record['file_name']:28s} fold={report[-1]['fold']:5s} "
              f"instances={summary['instances']:3d} "
              f"records_for_photo={report[-1]['annotator_records_for_this_photograph']}")

    # The disagreement figure: one photograph, one panel per annotator record.
    name, ids = max(by_photograph.items(), key=lambda item: len(item[1]))
    grey = cv2.imread(str(args.images / name), cv2.IMREAD_GRAYSCALE)
    panels = []
    for position, image_id in enumerate(sorted(ids)):
        canvas = np.repeat(grey[..., None], 3, axis=2)
        contours = [c for annotation in by_record[image_id] for c in as_contours(rings_of(annotation))]
        filled(canvas, contours, COLOURS[position % len(COLOURS)])
        outline(canvas, contours, COLOURS[position % len(COLOURS)], 2)
        panels.append(label(canvas, f"record {image_id}: {len(contours)} instances "
                                    f"({fold_of.get(image_id, 'unused')})"))
    disagreement = np.hstack(panels)
    scale = args.width * min(len(panels), 4) / disagreement.shape[1]
    disagreement = cv2.resize(disagreement, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    cv2.imwrite(str(args.output / f"annotators_{Path(name).stem}.jpg"), disagreement,
                [cv2.IMWRITE_JPEG_QUALITY, 92])
    print(f"\nannotator comparison: {name}, {len(ids)} records, "
          f"{[len(by_record[i]) for i in sorted(ids)]} instances each")

    (args.output / "sample.json").write_text(json.dumps(dict(
        annotations=str(args.annotations), images=str(args.images), offset=args.offset,
        photographs=len(by_photograph), records=len(meta),
        instances=sum(len(v) for v in by_record.values()), samples=report), indent=2))
    print(f"\nwrote {len(list(args.output.glob('*.jpg')))} figures to {args.output}")


if __name__ == "__main__":
    main()
