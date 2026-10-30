"""exp200i: would YOLO detections that exp169 lacks pay under pooled truth?

exp169's instances all come from Mask2Former. A YOLO detection that no exp169 instance covers was a
net loss under per-record scoring: a filament one annotator drew is a false positive against the
others. Under pooled one-to-one truth it is a true positive if any annotator drew it, and an
addition pays when its chance of matching times its expected IoU exceeds PQ / 2 (about 0.21).

For each YOLO score band this reports how many uncovered YOLO detections there are on validation
and how often they match a still-unmatched truth (pooled) or the truth of each record.

    python -m experiments.exp_200_test_truth.src.additions --annotations ... --predictions ...
        --yolo yolov8l=... --yolo yolo11m=...
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from pycocotools import mask as mu

from experiments.exp_033_inference.src.sweep import records_for
from experiments.exp_159_yolo_votes.src.votes import load_pool
from experiments.exp_200_test_truth.src.hypotheses import as_bytes
from experiments.exp_200_test_truth.src.tune import greedy

BANDS = ((0.9, 1.01), (0.8, 0.9), (0.7, 0.8), (0.6, 0.7), (0.5, 0.6), (0.4, 0.5), (0.25, 0.4))
MAX_COVERED = 0.10


def add_uncovered(final: list[dict], entries: list[dict], low: float, high: float = 1.01,
                  min_area: int = 100) -> list[dict]:
    """YOLO detections scored in [low, high) that final instances cover by less than 10%, painted by
    descending score into pixels nobody claimed, kept when at least `min_area` pixels remain."""
    claimed = None
    if final:
        claimed = mu.decode(mu.merge(final, intersect=False)) if len(final) > 1 else mu.decode(final[0])
    claimed = claimed.copy() if claimed is not None else np.zeros((2048, 2048), np.uint8)
    final_union = claimed.copy()
    added = []
    for e in sorted(entries, key=lambda e: -e["score"]):
        if not low <= e["score"] < high:
            continue
        mask = mu.decode(as_bytes(e["rle"]))
        if (mask & final_union).sum() >= MAX_COVERED * max(int(mask.sum()), 1):
            continue
        mask = mask & (1 - claimed)
        if int(mask.sum()) < min_area:
            continue
        claimed |= mask
        added.append(mu.encode(np.asfortranarray(mask)))
    return added


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--yolo", action="append", default=[])
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    records, _ = records_for(args.annotations)
    preds = json.loads(args.predictions.read_text())["exp169"]
    pools = {t: load_pool(Path(p)) for t, p in (s.split("=", 1) for s in args.yolo)}
    stats = {(t, b): dict(n=0, pooled_hits=0, record_hits=0, records=0, iou=0.0) for t in pools for b in BANDS}
    for name in sorted(records):
        final = [as_bytes(r) for r in preds[name]]
        passes = [[as_bytes(t) for t in truth] for _, truth in sorted(records[name])]
        flat = [t for truth in passes for t in truth]
        matched = set()
        if final and flat:
            iou = np.asarray(mu.iou(final, flat, [0] * len(flat))).reshape(len(final), len(flat))
            pairs = sorted(((iou[i, j], i, j) for i, j in zip(*np.nonzero(iou > 0.5))), reverse=True)
            used = set()
            for _, i, j in pairs:
                if i not in used and j not in matched:
                    used.add(i); matched.add(j)
        union = mu.merge(final, intersect=False) if len(final) > 1 else (final[0] if final else None)
        for tag, pool in pools.items():
            for e in pool.get(name, []):
                rle = as_bytes(e["rle"])
                area = mu.area(rle)
                if area < 100:
                    continue
                covered = mu.area(mu.merge([rle, union], intersect=True)) if union is not None else 0
                if covered >= MAX_COVERED * area:
                    continue
                band = next((b for b in BANDS if b[0] <= e["score"] < b[1]), None)
                if band is None:
                    continue
                s = stats[(tag, band)]
                s["n"] += 1
                free = [j for j in range(len(flat)) if j not in matched]
                if free:
                    ious = np.asarray(mu.iou([rle], [flat[j] for j in free], [0] * len(free))).ravel()
                    if ious.max() > 0.5:
                        s["pooled_hits"] += 1
                        s["iou"] += float(ious.max())
                for truth in passes:
                    s["records"] += 1
                    if truth and np.asarray(mu.iou([rle], truth, [0] * len(truth))).max() > 0.5:
                        s["record_hits"] += 1
    rows = []
    print(f"{'pool':8s} {'band':>11s} {'uncovered':>9s} {'pooled match':>12s} {'record match':>12s} {'mean IoU':>8s}")
    for (tag, band), s in stats.items():
        if not s["n"]:
            continue
        row = dict(pool=tag, band=band, n=s["n"], pooled_rate=s["pooled_hits"] / s["n"],
                   record_rate=s["record_hits"] / max(s["records"], 1),
                   mean_iou=s["iou"] / s["pooled_hits"] if s["pooled_hits"] else 0.0)
        rows.append(row)
        print(f"{tag:8s} {band[0]:.2f}-{band[1]:.2f} {s['n']:9d} {row['pooled_rate']:12.3f} {row['record_rate']:12.3f} "
              f"{row['mean_iou']:8.3f}")
    if args.output:
        args.output.write_text(json.dumps(rows, indent=1))


if __name__ == "__main__":
    main()
