"""Emit a test submission from an exp097 checkpoint, under a rule exp098 validated.

The rule is not re-tuned here. exp098 swept confidence, erosion and minimum area
on the canonical validation split and reported one winning rule with its exact
pooled PQ. This script replays that rule on validation first and refuses to touch
a test photograph unless it reproduces that PQ, so the deployed rule cannot
silently differ from the validated one (CLAUDE.md section 5).

Two emission modes, both identical to the code paths exp098 scored:

- `single`: one view at native 2048, confidence floor 0.05, NMS IoU 0.60, retina
  masks.
- `d4`: the 0.39 pipeline. Eight dihedral views at floor 0.25 through
  `dihedral.predict_views`, clustered and pixel-voted by `cached_tta.fuse` at
  (0.5, 0.5).

Painting is `sweep.evaluate`'s: drop candidates below the confidence, erode with
an elliptical 3x3 kernel `-grow` times, then `paint_panoptic` with the minimum
area. Painting assigns pixels to higher-scoring masks first, so filtering before
painting and filtering after it give the same instances, which is why one rule
reproduces the sweep exactly.

The CSV is written and checked on Kaggle: header, one row per instance, positive
areas, pixel-disjointness within each photograph, and the SHA256 is printed.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
from pycocotools import mask as mu

from experiments.exp_007_tta.src.dihedral import predict_views
from experiments.exp_033_inference.src.cached_tta import fuse
from experiments.exp_033_inference.src.sweep import encode, pq_rle, records_for
from shared.utils import aggregate_pq, paint_panoptic

NATIVE = 2048
KERNEL = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))


def emit(entries: list[dict], conf: float, grow: int, min_area: int) -> list[dict]:
    """sweep.evaluate's emission, kept identical so the validated rule deploys unchanged."""
    candidates = []
    for entry in entries:
        if entry["score"] < conf:
            continue
        mask = mu.decode(entry["rle"])
        if grow < 0:
            mask = cv2.erode(mask, KERNEL, iterations=-grow)
        elif grow != 0:
            raise ValueError(f"only unaltered and eroded masks are screened: {grow}")
        candidates.append((entry["score"], mask))
    return [encode(mask) for _, mask, _ in paint_panoptic(candidates, min_area=min_area)]


def single_view(model, image_path: Path, iou: float, floor: float, max_det: int) -> list[dict]:
    grey = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if grey is None or grey.shape != (NATIVE, NATIVE):
        raise SystemExit(f"cannot read {image_path}")
    image = np.repeat(grey[..., None], 3, axis=2)
    result = model.predict(image, imgsz=NATIVE, conf=floor, iou=iou, max_det=max_det,
                           retina_masks=True, verbose=False)[0]
    if result.masks is None or len(result.masks.data) == 0:
        return []
    entries = []
    for mask, score in zip(result.masks.data.cpu().numpy(), result.boxes.conf.cpu().numpy()):
        binary = (mask > 0.5).astype(np.uint8)
        if binary.shape != (NATIVE, NATIVE):
            binary = cv2.resize(binary, (NATIVE, NATIVE), interpolation=cv2.INTER_NEAREST)
        if binary.sum() >= 40:
            entries.append(dict(score=float(score), rle=encode(binary)))
    return entries


def d4_pool(model, image_path: Path, floor: float, max_det: int) -> list[dict]:
    views = predict_views(model, image_path, NATIVE, floor, max_det)
    entries = [(score, dict(size=rle["size"],
                            counts=rle["counts"].decode("ascii")
                            if isinstance(rle["counts"], bytes) else rle["counts"]))
               for score, rle in views]
    return fuse(entries, 0.5, 0.5)


def pool_for(model, mode: str, image_path: Path, args) -> list[dict]:
    if mode == "d4":
        return d4_pool(model, image_path, args.floor_d4, args.max_det)
    return single_view(model, image_path, args.iou, args.floor_single, args.max_det)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", required=True)
    parser.add_argument("--annotations", required=True)
    parser.add_argument("--train-images", type=Path, required=True)
    parser.add_argument("--test-images", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mode", choices=("single", "d4"), required=True)
    parser.add_argument("--conf", type=float, required=True)
    parser.add_argument("--grow", type=int, required=True)
    parser.add_argument("--min-area", type=int, required=True)
    parser.add_argument("--expected-pq", type=float, required=True)
    parser.add_argument("--replay-pool", type=Path, default=None,
                        help="exp098's saved validation pool; recomputed when absent")
    parser.add_argument("--tolerance", type=float, default=5e-4)
    parser.add_argument("--iou", type=float, default=0.60)
    parser.add_argument("--floor-single", type=float, default=0.05)
    parser.add_argument("--floor-d4", type=float, default=0.25)
    parser.add_argument("--max-det", type=int, default=100)
    args = parser.parse_args()
    cv2.setNumThreads(2)
    args.output.mkdir(parents=True, exist_ok=True)

    from ultralytics import YOLO
    model = YOLO(args.weights)
    records, split = records_for(args.annotations)

    # 1. Replay the validated rule. exp098's pool is the pool it scored, so reusing
    #    it proves the rule rather than the inference settings; recomputing is the
    #    fallback and costs a full validation pass.
    if args.replay_pool and args.replay_pool.is_file():
        pool = json.loads(args.replay_pool.read_text())
        source = f"exp098 pool {args.replay_pool.name}"
    else:
        pool = {name: pool_for(model, args.mode, args.train_images / name, args)
                for name in records}
        source = "recomputed"
    missing = [name for name in records if name not in pool]
    if missing:
        raise SystemExit(f"replay pool is missing {len(missing)} photographs: {missing[:3]}")
    rows = []
    for name, passes in records.items():
        predicted = emit(pool[name], args.conf, args.grow, args.min_area)
        rows.extend(pq_rle(predicted, truth) for _, truth in passes)
    replay = aggregate_pq(rows)
    print("VALIDATION_REPLAY " + json.dumps(dict(replay, source=source)), flush=True)
    if abs(replay["pq"] - args.expected_pq) > args.tolerance:
        raise SystemExit(f"replay {replay['pq']} differs from exp098's {args.expected_pq}")

    # 2. Emit the test set only after the replay agreed.
    test = sorted(args.test_images.glob("*.jpeg"))
    if len(test) != 180:
        raise SystemExit(f"expected 180 test photographs, found {len(test)}")
    destination = args.output / "submission.csv"
    written, contributing = 0, set()
    with destination.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["filament_id", "segmentation_rle"])
        for position, path in enumerate(test, start=1):
            predicted = emit(pool_for(model, args.mode, path, args), args.conf, args.grow,
                             args.min_area)
            claimed = np.zeros((NATIVE, NATIVE), np.uint8)
            for index, rle in enumerate(predicted, start=1):
                mask = mu.decode(dict(size=rle["size"], counts=rle["counts"].encode("ascii")))
                if not mask.sum():
                    raise SystemExit(f"empty instance emitted for {path.stem}")
                if (claimed & mask).any():
                    raise SystemExit(f"overlapping instances emitted for {path.stem}")
                claimed |= mask
                writer.writerow([f"{path.stem}_{index}", rle["counts"]])
                written += 1
                contributing.add(path.stem)
            if position % 20 == 0:
                print(f"PROGRESS {position}/180 photographs, {written} rows", flush=True)

    summary = dict(status="complete", mode=args.mode, checkpoint=Path(args.weights).name,
                   rule=dict(conf=args.conf, grow=args.grow, min_area=args.min_area),
                   expected_pq=args.expected_pq, validation_pq=replay["pq"],
                   replay_source=source, sq=replay["sq"], rq=replay["rq"],
                   tp=replay["tp"], fp=replay["fp"], fn=replay["fn"],
                   split=split.summary(), rows=written,
                   contributing_images=len(contributing), test_photographs=len(test),
                   sha256=hashlib.sha256(destination.read_bytes()).hexdigest().upper())
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2))
    print("SUMMARY " + json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
