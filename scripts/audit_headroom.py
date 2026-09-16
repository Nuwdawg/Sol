"""Measure where the canonical PQ gap actually is: boundaries or discovery.

The retained D4 emission is held fixed. This script never changes a prediction.
It decomposes the exact 0.4558638687 baseline into segmentation quality and
recognition quality, then prices three oracles:

1. Boundary oracle: raise every matched IoU to a target and rescore.
2. Threshold-crossing oracle: count unmatched truth records that already have a
   predicted mask above a given IoU, so a better boundary alone would match.
3. Selection oracle: the best achievable IoU per truth record from the full weak
   D4 candidate pool, which bounds any pure re-ranking or proposal-quality head.
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path

import cv2
import numpy as np
from pycocotools import mask as mu

from experiments.exp_033_inference.src.sweep import records_for
from shared.utils import aggregate_pq

BASELINE_PQ = 0.4558638687292126


def emitted(rich: dict) -> dict:
    return {name: [row["rle"] for row in rows] for name, rows in rich.items()}


def matched_pairs(predicted: list, truth: list):
    """Greedy strict >0.5 matching, returning the pairs and the full IoU table."""
    if not predicted or not truth:
        return [], np.zeros((len(predicted), len(truth)))
    ious = np.asarray(mu.iou(predicted, truth, [0] * len(truth)))
    used_p, used_g, pairs = set(), set(), []
    for p, g in sorted(np.argwhere(ious > .5), key=lambda pair: -ious[pair[0], pair[1]]):
        if p not in used_p and g not in used_g:
            used_p.add(int(p)); used_g.add(int(g)); pairs.append((int(p), int(g), float(ious[p, g])))
    return pairs, ious


def pq_from(tp: int, fp: int, fn: int, iou_sum: float) -> dict:
    sq = iou_sum / tp if tp else 0.
    rq = tp / (tp + .5 * fp + .5 * fn) if (tp + fp + fn) else 0.
    return dict(pq=sq * rq, sq=sq, rq=rq, tp=tp, fp=fp, fn=fn)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rich", type=Path, required=True)
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    cv2.setNumThreads(1)

    rich = json.loads(args.rich.read_text())
    pool = json.loads(args.candidates.read_text())
    records, split = records_for(args.annotations)
    assert set(rich) == set(records), "retained candidates must cover the split"

    predictions = emitted(rich)
    metrics, matched_ious, miss_best, pool_best, fp_best = [], [], [], [], []
    per_photo = {}
    for name, passes in records.items():
        predicted = predictions[name]
        pool_rles = [row["rle"] for row in pool.get(name, [])]
        for _, truth in passes:
            pairs, ious = matched_pairs(predicted, truth)
            tp = len(pairs)
            iou_sum = sum(pair[2] for pair in pairs)
            metrics.append(dict(tp=tp, fp=len(predicted) - tp, fn=len(truth) - tp,
                                iou_sum=iou_sum))
            matched_ious.extend(pair[2] for pair in pairs)
            taken_g = {pair[1] for pair in pairs}
            taken_p = {pair[0] for pair in pairs}
            for g in range(len(truth)):
                if g in taken_g:
                    continue
                miss_best.append(float(ious[:, g].max()) if len(predicted) else 0.)
            for p in range(len(predicted)):
                if p in taken_p:
                    continue
                fp_best.append(float(ious[p, :].max()) if len(truth) else 0.)
            if pool_rles and truth:
                table = np.asarray(mu.iou(pool_rles, truth, [0] * len(truth)))
                pool_best.extend(float(v) for v in table.max(axis=0))
            else:
                pool_best.extend(0. for _ in truth)
        per_photo[name] = dict(predicted=len(predicted), passes=len(passes),
                               pool=len(pool_rles))

    baseline = aggregate_pq(metrics)
    assert abs(baseline["pq"] - BASELINE_PQ) < 1e-9, baseline
    tp, fp, fn = baseline["tp"], baseline["fp"], baseline["fn"]
    iou_sum = tp * baseline["sq"]

    # 1. Boundary oracle: the same matches, better masks.
    boundary = [dict(target=t, **pq_from(tp, fp, fn, tp * max(t, baseline["sq"])))
                for t in (.70, .75, .80, .85, .90, 1.0)]

    # 2. Threshold-crossing oracle: misses that a better boundary would match.
    miss = np.asarray(miss_best)
    crossing = []
    for floor in (.5, .4, .3, .2, .1, .0):
        recoverable = int(((miss > floor) & (miss <= .5)).sum())
        # Each recovery converts one FN and one FP into a TP at the assumed IoU.
        for assumed in (.55, .65, .75):
            crossing.append(dict(floor=floor, recoverable=recoverable, assumed=assumed,
                                 **pq_from(tp + recoverable,
                                           max(0, fp - recoverable),
                                           fn - recoverable,
                                           iou_sum + recoverable * assumed)))

    # 3. Selection oracle over the full weak pool.
    pool_array = np.asarray(pool_best)
    selection = dict(
        truth_records=int(pool_array.size),
        above_half=int((pool_array > .5).sum()),
        mean_best=float(pool_array.mean()),
        mean_best_above_half=float(pool_array[pool_array > .5].mean()) if (pool_array > .5).any() else 0.,
    )
    # Perfect selection with today's false-positive burden unchanged.
    oracle_tp = selection["above_half"]
    oracle_iou = float(pool_array[pool_array > .5].sum())
    selection["perfect_keep_fp"] = pq_from(oracle_tp, fp, int(pool_array.size) - oracle_tp,
                                           oracle_iou)
    selection["perfect_no_fp"] = pq_from(oracle_tp, 0, int(pool_array.size) - oracle_tp,
                                          oracle_iou)

    def histogram(values, edges=(0., .1, .2, .3, .4, .5, .6, .7, .8, .9, 1.01)):
        counts = Counter()
        for value in values:
            for low, high in zip(edges, edges[1:]):
                if low <= value < high:
                    counts[f"{low:.1f}-{high:.1f}"] += 1
                    break
        return dict(sorted(counts.items()))

    report = dict(
        status="complete",
        split=split.summary(),
        baseline=baseline,
        matched_iou=dict(count=len(matched_ious),
                         mean=float(np.mean(matched_ious)),
                         median=float(np.median(matched_ious)),
                         p10=float(np.percentile(matched_ious, 10)),
                         p90=float(np.percentile(matched_ious, 90)),
                         histogram=histogram(matched_ious)),
        missed_truth=dict(count=int(miss.size), zero=int((miss == 0).sum()),
                          histogram=histogram(miss)),
        false_positive=dict(count=len(fp_best), zero=int((np.asarray(fp_best) == 0).sum()),
                            histogram=histogram(fp_best)),
        boundary_oracle=boundary,
        crossing_oracle=crossing,
        selection_oracle=selection,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
