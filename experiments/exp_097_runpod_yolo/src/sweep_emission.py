"""Sweep emission rules the frozen scorer refuses to try: dilation, and larger areas.

Why reopen a settled question
-----------------------------
exp_005 measured erosion against dilation on the exp002 model and found every
dilation lost, monotonically: grow +3 scored 0.2016 where grow -1 scored 0.4404.
`sweep.evaluate_thresholds` encodes that result as a guard - it raises on any
positive `grow` - and the guard has been correct for every model since.

It is no longer obviously correct. That measurement was made on a yolo11m-seg
trained at 1280, whose masks were systematically too large, which is why trimming
a pixel helped. exp097's YOLOv8l-seg trained at 2048 is a different model with a
different bias: exp098 found its optimum at grow 0 for every checkpoint and in
both single-view and D4 modes, with higher SQ than the old model achieved under
erosion. The optimum moved from -1 to 0 when training resolution doubled. Whether
it has moved past 0 is an open question that no measurement here has asked,
because the guard prevents asking it.

This module asks it, and keeps the old result honest two ways: it reproduces
`evaluate_thresholds` exactly wherever the two overlap (grow 0 and -1), and it
reports the full grid so a win at +1 can be read against its neighbours rather
than in isolation. A peak on the grid boundary is reported as such.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from pycocotools import mask as mu

from experiments.exp_033_inference.src.sweep import encode, evaluate_thresholds, pq_rle, records_for
from experiments.exp_097_runpod_yolo.src.events import emit
from shared.utils import aggregate_pq, paint_panoptic

BASELINE_PQ = 0.4558638687292126
KERNEL = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))


def reshape(mask: np.ndarray, grow: int) -> np.ndarray:
    """Erode for negative `grow`, dilate for positive, leave alone at zero."""
    if grow < 0:
        return cv2.erode(mask, KERNEL, iterations=-grow)
    if grow > 0:
        return cv2.dilate(mask, KERNEL, iterations=grow)
    return mask


def evaluate_grid(pool: dict, records: dict, confs: tuple, grow: int, min_area: int) -> list[dict]:
    """`sweep.evaluate_thresholds` with the dilation guard lifted.

    Identical in every other respect: paint once at the lowest confidence, then
    take prefixes, because `paint_panoptic` gives pixels to higher-scoring masks
    first and a lower-scoring mask cannot change what a higher one kept.
    """
    confs = sorted(set(confs))
    totals = {conf: [] for conf in confs}
    for name, passes in records.items():
        candidates = []
        for entry in pool[name]:
            if entry["score"] < confs[0]:
                continue
            candidates.append((entry["score"], reshape(mu.decode(entry["rle"]), grow)))
        painted = [(score, encode(mask)) for score, mask, _ in
                   paint_panoptic(candidates, min_area=min_area)]
        for conf in confs:
            predicted = [rle for score, rle in painted if score >= conf]
            for _, truth in passes:
                totals[conf].append(pq_rle(predicted, truth))
    return [dict(aggregate_pq(totals[conf]), conf=conf, grow=grow, min_area=min_area)
            for conf in confs]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pool", required=True, help="label=path")
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--grows", default="-2,-1,0,1,2")
    parser.add_argument("--areas", default="100,300,500,800")
    parser.add_argument("--confs", default="0.15,0.20,0.25,0.30,0.35,0.40,0.45,0.50")
    parser.add_argument("--parity-pool", type=Path, default=None,
                        help="exp002's stored pool; its 0.35/-1/300 point must reproduce")
    args = parser.parse_args()
    cv2.setNumThreads(2)
    args.output.mkdir(parents=True, exist_ok=True)
    records, split = records_for(args.annotations)
    label, _, path = args.pool.partition("=")
    pool = json.loads(Path(path).read_text())
    pool = {Path(name).name: entries for name, entries in pool.items()}

    # Parity, two ways: the historical number, and agreement with the frozen
    # scorer on the settings both can evaluate.
    if args.parity_pool and args.parity_pool.is_file():
        baseline = json.loads(args.parity_pool.read_text())
        frozen = evaluate_thresholds(baseline, records, (0.35,), grow=-1, min_area=300)[0]
        mine = evaluate_grid(baseline, records, (0.35,), grow=-1, min_area=300)[0]
        if abs(frozen["pq"] - BASELINE_PQ) > 1e-9 or abs(mine["pq"] - frozen["pq"]) > 1e-12:
            raise SystemExit(f"parity failed: frozen {frozen['pq']}, here {mine['pq']}")
        emit("PARITY", dict(baseline=frozen["pq"], reproduced=mine["pq"],
                            tp=mine["tp"], fp=mine["fp"], fn=mine["fn"]))

    grows = [int(value) for value in args.grows.split(",")]
    areas = [int(value) for value in args.areas.split(",")]
    confs = tuple(float(value) for value in args.confs.split(","))
    rows = []
    for grow in grows:
        for area in areas:
            found = evaluate_grid(pool, records, confs, grow, area)
            rows.extend(found)
            best = max(found, key=lambda row: row["pq"])
            emit("GRID", dict(label=label, grow=grow, min_area=area, best_pq=best["pq"],
                              conf=best["conf"], tp=best["tp"], fp=best["fp"], fn=best["fn"]))
            (args.output / "grid.json").write_text(json.dumps(
                dict(label=label, split=split.summary(), rows=rows), indent=2))

    best = max(rows, key=lambda row: row["pq"])
    on_edge = best["grow"] in (grows[0], grows[-1]) or best["min_area"] in (areas[0], areas[-1]) \
        or best["conf"] in (confs[0], confs[-1])
    emit("SUMMARY", dict(label=label, **best, delta_vs_baseline=best["pq"] - BASELINE_PQ,
                         on_grid_edge=on_edge, clears_040_bar=best["pq"] >= 0.472))


if __name__ == "__main__":
    main()
