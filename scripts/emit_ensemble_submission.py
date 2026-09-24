"""Emit a test submission from the exp102 two-architecture ensemble.

The rule is not re-tuned here. exp102 fused exp002's D4 instance pool with
exp097's and swept emission, reporting PQ 0.468146 at confidence 0.35, erosion
1 px and minimum area 100. This script replays that fusion and that rule on the
validation split first, and refuses to touch a test photograph unless it
reproduces 0.468146 (CLAUDE.md section 5).

Two conventions matter and are easy to confuse, so both are named explicitly:

- **D4 fusion within one model** clusters the eight dihedral views of that model.
  `cached_tta.fuse` divides cluster support by a hardcoded eight, which is correct
  for exactly this step. Both models' pools are built this way.
- **Cross-model fusion** clusters the two models' already-fused *instances*.
  `fuse_sources(..., sources=2)` scores an instance both architectures proposed at
  its full value and one proposed by a single architecture at 0.75. Passing raw
  views here, or passing instances with sources=8, both silently produce a
  different ensemble than the one exp102 scored.

exp002's contribution is precomputed on both splits: its validation pool is already
fused, and its test views arrive raw and are fused here with the identical call.
exp097 has no cached test pool, so its eight views are predicted on the pod.
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
from experiments.exp_097_runpod_yolo.src.evaluate_ensemble import fuse_sources
from shared.utils import aggregate_pq, paint_panoptic

NATIVE = 2048
KERNEL = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
MODELS = 2


def as_text(rle: dict) -> dict:
    counts = rle["counts"]
    return dict(size=rle["size"],
                counts=counts.decode("ascii") if isinstance(counts, bytes) else counts)


def emit(entries: list[dict], conf: float, grow: int, min_area: int) -> list[dict]:
    """sweep.evaluate's emission, kept identical so the validated rule deploys unchanged."""
    candidates = []
    for entry in entries:
        if entry["score"] < conf:
            continue
        mask = mu.decode(as_text(entry["rle"]))
        if grow < 0:
            mask = cv2.erode(mask, KERNEL, iterations=-grow)
        elif grow != 0:
            raise ValueError(f"only unaltered and eroded masks are screened: {grow}")
        candidates.append((entry["score"], mask))
    return [encode(mask) for _, mask, _ in paint_panoptic(candidates, min_area=min_area)]


def instances_from_views(views) -> list[dict]:
    """One model's eight dihedral views -> its fused D4 instances (denominator eight)."""
    entries = [(float(score), as_text(rle)) for score, rle in views]
    return fuse(entries, 0.5, 0.5)


def cross_fuse(pools: list[list[dict]], link: float, vote: float) -> list[dict]:
    """exp102's cross-model step: cluster instances, denominator = number of models."""
    entries = [(float(entry["score"]), as_text(entry["rle"]))
               for pool in pools for entry in pool]
    return fuse_sources(entries, link, vote, MODELS)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", required=True)
    parser.add_argument("--annotations", required=True)
    parser.add_argument("--test-images", type=Path, required=True)
    parser.add_argument("--exp002-val-pool", type=Path, required=True)
    parser.add_argument("--exp097-val-pool", type=Path, required=True)
    parser.add_argument("--exp002-test-views", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--conf", type=float, required=True)
    parser.add_argument("--grow", type=int, required=True)
    parser.add_argument("--min-area", type=int, required=True)
    parser.add_argument("--expected-pq", type=float, required=True)
    parser.add_argument("--tolerance", type=float, default=5e-4)
    parser.add_argument("--link", type=float, default=0.5)
    parser.add_argument("--vote", type=float, default=0.5)
    parser.add_argument("--floor", type=float, default=0.25)
    parser.add_argument("--max-det", type=int, default=100)
    args = parser.parse_args()
    cv2.setNumThreads(2)
    args.output.mkdir(parents=True, exist_ok=True)
    records, split = records_for(args.annotations)

    # 1. Replay exp102's fusion and rule on validation, from both cached pools.
    exp002 = {Path(k).name: v for k, v in json.loads(args.exp002_val_pool.read_text()).items()}
    exp097 = {Path(k).name: v for k, v in json.loads(args.exp097_val_pool.read_text()).items()}
    missing = [name for name in records if name not in exp002 or name not in exp097]
    if missing:
        raise SystemExit(f"validation pools miss {len(missing)} photographs: {missing[:3]}")
    rows = []
    for name, passes in records.items():
        fused = cross_fuse([exp002[name], exp097[name]], args.link, args.vote)
        predicted = emit(fused, args.conf, args.grow, args.min_area)
        rows.extend(pq_rle(predicted, truth) for _, truth in passes)
    replay = aggregate_pq(rows)
    print("VALIDATION_REPLAY " + json.dumps(replay), flush=True)
    if abs(replay["pq"] - args.expected_pq) > args.tolerance:
        raise SystemExit(f"replay {replay['pq']} differs from exp102's {args.expected_pq}")

    # 2. exp002's test contribution: raw views in, fused instances out.
    views_by_photograph: dict[str, list] = {}
    for path in args.exp002_test_views:
        for name, views in json.loads(path.read_text()).items():
            views_by_photograph.setdefault(Path(name).name, []).extend(views)
    print(f"EXP002_TEST_VIEWS {len(views_by_photograph)} photographs", flush=True)

    test = sorted(args.test_images.glob("*.jpeg"))
    if len(test) != 180:
        raise SystemExit(f"expected 180 test photographs, found {len(test)}")
    absent = [p.name for p in test if p.name not in views_by_photograph]
    if absent:
        raise SystemExit(f"exp002 test views miss {len(absent)} photographs: {absent[:3]}")

    # 3. exp097 has no cached test pool, so predict its eight views on the pod.
    from ultralytics import YOLO
    model = YOLO(args.weights)

    destination = args.output / "submission.csv"
    written, contributing = 0, set()
    with destination.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["filament_id", "segmentation_rle"])
        for position, path in enumerate(test, start=1):
            pool_002 = instances_from_views(views_by_photograph[path.name])
            pool_097 = instances_from_views(
                predict_views(model, path, NATIVE, args.floor, args.max_det))
            fused = cross_fuse([pool_002, pool_097], args.link, args.vote)
            predicted = emit(fused, args.conf, args.grow, args.min_area)
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

    summary = dict(status="complete", ensemble=["exp002", "exp097"],
                   checkpoint=Path(args.weights).name,
                   rule=dict(conf=args.conf, grow=args.grow, min_area=args.min_area,
                             link=args.link, vote=args.vote, sources=MODELS),
                   expected_pq=args.expected_pq, validation_pq=replay["pq"],
                   sq=replay["sq"], rq=replay["rq"], tp=replay["tp"], fp=replay["fp"],
                   fn=replay["fn"], split=split.summary(), rows=written,
                   contributing_images=len(contributing), test_photographs=len(test),
                   sha256=hashlib.sha256(destination.read_bytes()).hexdigest().upper())
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2))
    print("SUMMARY " + json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
