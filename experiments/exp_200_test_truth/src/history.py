"""exp200e: score past submissions under per-record and pooled one-to-one truth, beside their public score.

matching.py found that pooled truth (every annotator's filaments of a photograph in one image) with
one-to-one matching puts exp139, exp169 and exp097 at 0.400 / 0.414 / 0.387 on validation, against
public 0.41 / 0.41 / 0.39, where per-record scoring says 0.471 / 0.489 / 0.467. Three points can fit
by accident. This adds every other submission whose validation predictions can be rebuilt from the
stored pools: the YOLO11m D4 operating points of exp037-exp065, and exp097 stage2.

    python -m experiments.exp_200_test_truth.src.history --annotations ... --predictions ...
        --yolo11m-pool ...
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from pycocotools import mask as mu

from experiments.exp_033_inference.src.sweep import records_for
from experiments.exp_159_yolo_votes.src.votes import load_pool
from experiments.exp_200_test_truth.src.hypotheses import as_bytes, match, total
from experiments.exp_200_test_truth.src.matching import one_to_one
from shared.utils import paint_panoptic

KERNEL = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))


def emit(pool: dict, names, conf: float, erode: int, min_area: int) -> dict:
    """exp033's emission: score floor, optional erosion, painted by score with an area floor."""
    out = {}
    for name in names:
        candidates = []
        for e in pool.get(name, []):
            if e["score"] < conf:
                continue
            mask = mu.decode(as_bytes(e["rle"]))
            if erode:
                mask = cv2.erode(mask, KERNEL, iterations=erode)
            candidates.append((e["score"], mask))
        out[name] = [mu.encode(np.asfortranarray(m)) for _, m, _ in paint_panoptic(candidates, min_area=min_area)]
    return out


def score(preds: dict, records: dict) -> tuple[dict, dict]:
    per_record, pooled = [], []
    for name in sorted(records):
        mine = [as_bytes(x) for x in preds[name]]
        passes = [[as_bytes(t) for t in truth] for _, truth in sorted(records[name])]
        per_record.extend(match(mine, truth) for truth in passes)
        pooled.append(one_to_one(mine, [t for truth in passes for t in truth]))
    return total(per_record), total(pooled)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True, help="hypotheses.py's prediction cache")
    parser.add_argument("--yolo11m-pool", type=Path, required=True, help="exp002 D4 validation pool")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    records, _ = records_for(args.annotations)
    names = sorted(records)
    cached = json.loads(args.predictions.read_text())
    y11 = load_pool(args.yolo11m_pool)
    # (submission, public, recorded validation PQ or None, predictions)
    rows = [
        ("exp169 joint rule", 0.41, 0.48894859995, cached["exp169"]),
        ("exp139 Mask2Former e40", 0.41, 0.4705249990, cached["exp139"]),
        ("exp097 YOLOv8l D4 c.35 a300", 0.39, 0.466806, cached["exp097"]),
        ("exp062 YOLO11m D4 c.35 e1 a300", 0.39, 0.4558638687, emit(y11, names, 0.35, 1, 300)),
        ("exp065 YOLO11m D4 c.35 e1 a100", 0.38, None, emit(y11, names, 0.35, 1, 100)),
        ("exp057 YOLO11m D4 c.385 e1 a100", 0.38, 0.45864, emit(y11, names, 0.385, 1, 100)),
        ("exp064 YOLO11m D4 c.30 e1 a300", 0.38, None, emit(y11, names, 0.30, 1, 300)),
    ]
    report = []
    print(f"{'submission':48s} {'public':>6s} {'record':>8s} {'pooled 1:1':>10s}  recorded")
    for label, public, recorded, preds in rows:
        rec, pooled = score({n: preds[n] for n in names}, records)
        report.append(dict(submission=label, public=public, recorded=recorded, record=rec, pooled_1to1=pooled))
        print(f"{label:48s} {public if public else '':>6} {rec['pq']:8.4f} {pooled['pq']:10.4f}  {recorded or ''}")
    if args.output:
        args.output.write_text(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
