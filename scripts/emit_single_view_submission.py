"""Emit a single-view exp002 test submission under a rule chosen by exp095.

The identical code path that exp095 scored (`predict` then `paint` from
`scripts.probe_yolo_inference_config`) is replayed on the canonical validation
split first. It must reproduce the PQ exp095 reported for this rule before any
test photograph is touched, so the deployed rule cannot silently differ from the
validated one. The CSV is written and checked on Kaggle: schema, all 180 test
identifiers, positive areas and pixel-disjointness. Only a one-line summary is
printed.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
from pycocotools import mask as mu

from experiments.exp_033_inference.src.sweep import pq_rle, records_for
from scripts.probe_yolo_inference_config import paint, predict, read_image
from shared.utils import aggregate_pq

NATIVE = 2048


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", required=True)
    parser.add_argument("--annotations", required=True)
    parser.add_argument("--train-images", type=Path, required=True)
    parser.add_argument("--test-images", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--iou", type=float, required=True)
    parser.add_argument("--retina", type=int, required=True)
    parser.add_argument("--conf", type=float, required=True)
    parser.add_argument("--erode", type=int, required=True)
    parser.add_argument("--min-area", type=int, required=True)
    parser.add_argument("--order", choices=("score", "public"), required=True)
    parser.add_argument("--expected-pq", type=float, required=True)
    parser.add_argument("--floor", type=float, default=0.05)
    parser.add_argument("--max-det", type=int, default=100)
    args = parser.parse_args()

    from ultralytics import YOLO
    model = YOLO(args.weights)
    rule = dict(conf=args.conf, erode=bool(args.erode), min_area=args.min_area, order=args.order)

    records, _ = records_for(args.annotations)
    metrics = []
    for name, passes in records.items():
        entries = predict(model, read_image(args.train_images / name), args.iou,
                          bool(args.retina), args.floor, args.max_det)
        predicted = paint(entries, **rule)
        metrics.extend(pq_rle(predicted, truth) for _, truth in passes)
    replay = aggregate_pq(metrics)
    print("VALIDATION_REPLAY", json.dumps(replay), flush=True)
    assert abs(replay["pq"] - args.expected_pq) < 5e-4, (replay["pq"], args.expected_pq)

    test = sorted(args.test_images.glob("*.jpeg"))
    assert len(test) == 180, len(test)
    args.output.mkdir(parents=True, exist_ok=True)
    destination = args.output / "submission.csv"
    rows, images = 0, set()
    with destination.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["filament_id", "segmentation_rle"])
        for path in test:
            entries = predict(model, read_image(path), args.iou, bool(args.retina),
                              args.floor, args.max_det)
            predicted = paint(entries, **rule)
            if predicted:
                claimed = np.zeros((NATIVE, NATIVE), np.uint8)
                for rle in predicted:
                    mask = mu.decode(rle)
                    assert mask.sum() > 0 and not (claimed & mask).any(), path.stem
                    claimed |= mask
            for index, rle in enumerate(predicted, start=1):
                writer.writerow([f"{path.stem}_{index}", rle["counts"]])
                rows += 1
                images.add(path.stem)
    summary = dict(status="complete", rule=dict(iou=args.iou, retina=bool(args.retina), **rule),
                   expected_pq=args.expected_pq, validation_pq=replay["pq"],
                   tp=replay["tp"], fp=replay["fp"], fn=replay["fn"],
                   rows=rows, contributing_images=len(images),
                   sha256=hashlib.sha256(destination.read_bytes()).hexdigest().upper())
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2))
    print("SUMMARY", json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
