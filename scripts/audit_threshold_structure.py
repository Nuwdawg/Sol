"""What kind of error puts an object at IoU 0.4-0.5, and what could move it across.

Why this audit and not another score
------------------------------------
Panoptic Quality is

    PQ = sum(IoU over matched pairs) / (TP + FP/2 + FN/2)

so an object sitting just below the 0.5 matching threshold costs a full unit of
denominator (half a false positive plus half a false negative) and contributes
nothing to the numerator. Push the same object to 0.51 and the denominator is
essentially unchanged while the numerator gains 0.51. Polishing an already
matched 0.55 pair to 0.60 gains 0.05. Threshold crossings are worth roughly ten
times a comparable geometric correction, so the question is not "how good are the
masks" but "what is keeping these specific objects below 0.5".

exp088 measured the populations: 66 truths whose best overlap is 0.4-0.5, 127
false positives overlapping truth at 0.3-0.5, and only 13 misses recoverable by
joining a *pair* of fragments. That last test looked at one repair in one
direction. This audit asks the structural question instead.

Matching is global, so per-truth maxima are misleading: one prediction can cover
two truths, two predictions can compete for one truth, and a repair can change
several assignments at once. The audit therefore builds the bipartite
prediction-truth overlap graph per record, takes its connected components, and
classifies each as 1:1, 1:N (one prediction over several truths, a merge), N:1
(several predictions over one truth, a fragmentation) or N:M.

Two oracle repairs bound what instance surgery could win, using ground truth only
to set the ceiling, never as an inference rule:

- **union**: replace the predictions overlapping one truth with their union;
- **split**: divide a prediction covering several truths by assigning each of its
  pixels to the truth it lies in.

Both are upper bounds. A real spine-graph stage decides these cases from geometry
- tangent continuity at junctions, curvature, width consistency - without seeing
truth. If the bounds are small, that stage is not worth building.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path

import cv2
import numpy as np
from pycocotools import mask as mu

from experiments.exp_033_inference.src.sweep import encode, pq_rle, records_for
from shared.utils import aggregate_pq, paint_panoptic

BASELINE_PQ = 0.4558638687292126
RULE = dict(conf=0.35, grow=-1, min_area=300)
KERNEL = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
TOUCH = 0.05          # an overlap below this is not evidence of a shared object


def emit(entries: list[dict], conf: float, grow: int, min_area: int) -> list[dict]:
    candidates = []
    for entry in entries:
        if entry["score"] < conf:
            continue
        mask = mu.decode(entry["rle"])
        if grow < 0:
            mask = cv2.erode(mask, KERNEL, iterations=-grow)
        candidates.append((entry["score"], mask))
    return [encode(mask) for _, mask, _ in paint_panoptic(candidates, min_area=min_area)]


def components(ious: np.ndarray, threshold: float) -> list[tuple[list[int], list[int]]]:
    """Connected components of the bipartite prediction-truth overlap graph."""
    predictions, truths = ious.shape
    seen_p, seen_g, found = set(), set(), []
    for start in range(predictions):
        if start in seen_p:
            continue
        stack_p, stack_g = [start], []
        seen_p.add(start)
        group_p, group_g = [start], []
        while stack_p or stack_g:
            if stack_p:
                node = stack_p.pop()
                for g in np.nonzero(ious[node] > threshold)[0]:
                    if int(g) not in seen_g:
                        seen_g.add(int(g)); group_g.append(int(g)); stack_g.append(int(g))
            else:
                node = stack_g.pop()
                for p in np.nonzero(ious[:, node] > threshold)[0]:
                    if int(p) not in seen_p:
                        seen_p.add(int(p)); group_p.append(int(p)); stack_p.append(int(p))
        found.append((sorted(group_p), sorted(group_g)))
    for g in range(truths):                      # truths nothing overlaps at all
        if g not in seen_g:
            found.append(([], [g]))
    return found


def classify(group: tuple[list[int], list[int]]) -> str:
    predictions, truths = len(group[0]), len(group[1])
    if predictions == 0:
        return "0:1 invisible"
    if truths == 0:
        return "1:0 spurious"
    if predictions == 1 and truths == 1:
        return "1:1"
    if predictions == 1:
        return "1:N merge"
    if truths == 1:
        return "N:1 fragmentation"
    return "N:M"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pool", type=Path,
                        default=Path("data/exp033_cached_tta/candidates_0.5_0.5.json"))
    parser.add_argument("--annotations", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=Path("data/exp105_threshold_structure"))
    parser.add_argument("--band", type=float, nargs=2, default=(0.4, 0.5))
    args = parser.parse_args()
    cv2.setNumThreads(2)
    args.output.mkdir(parents=True, exist_ok=True)
    annotations = args.annotations or next(Path(".").rglob("MAGFiLO_1.0_Annotations_kaggle2026_train.json"))
    records, split = records_for(annotations)
    pool = json.loads(args.pool.read_text())

    rows, kinds = [], defaultdict(int)
    band_low, band_high = args.band
    band_members, oracle = defaultdict(int), dict(union=[], split=[])
    union_rows, split_rows = [], []

    for photograph, passes in records.items():
        predicted = emit(pool[photograph], **RULE)
        for image_id, truths in passes:
            rows.append(pq_rle(predicted, truths))
            if not predicted or not truths:
                kinds["0:1 invisible" if truths else "1:0 spurious"] += max(len(truths), len(predicted))
                continue
            ious = np.asarray(mu.iou(predicted, truths, [0] * len(truths)))
            best_per_truth = ious.max(axis=0)

            for group in components(ious, TOUCH):
                kind = classify(group)
                kinds[kind] += 1
                members = group[1]
                for g in members:
                    if band_low <= best_per_truth[g] < band_high:
                        band_members[kind] += 1

            # Oracle 1: union every prediction that overlaps a given truth.
            for g, truth in enumerate(truths):
                overlapping = [p for p in range(len(predicted)) if ious[p, g] > TOUCH]
                if len(overlapping) < 2 or best_per_truth[g] >= 0.5:
                    continue
                merged = mu.merge([predicted[p] for p in overlapping])
                after = float(mu.iou([merged], [truth], [0])[0][0])
                union_rows.append(dict(before=round(float(best_per_truth[g]), 4),
                                       after=round(after, 4), parts=len(overlapping),
                                       crosses=bool(before_after := after >= 0.5 > best_per_truth[g])))
                if before_after:
                    oracle["union"].append(photograph)

            # Oracle 2: split a prediction that covers several truths, by truth assignment.
            for p in range(len(predicted)):
                covered = [g for g in range(len(truths)) if ious[p, g] > TOUCH]
                if len(covered) < 2:
                    continue
                mask = mu.decode(predicted[p])
                gains = 0
                for g in covered:
                    piece = mask & mu.decode(truths[g])
                    if piece.sum() < 40:
                        continue
                    after = float(mu.iou([encode(piece)], [truths[g]], [0])[0][0])
                    if after >= 0.5 > best_per_truth[g]:
                        gains += 1
                split_rows.append(dict(covers=len(covered), rescued=gains))
                if gains:
                    oracle["split"].append(photograph)

    baseline = aggregate_pq(rows)
    if abs(baseline["pq"] - BASELINE_PQ) > 1e-9:
        raise SystemExit(f"baseline not reproduced: {baseline}")

    union_crossings = sum(1 for row in union_rows if row["crosses"])
    split_crossings = sum(row["rescued"] for row in split_rows)
    report = dict(
        split=split.summary(), baseline=baseline, rule=RULE,
        component_kinds=dict(sorted(kinds.items())),
        near_threshold_band=[band_low, band_high],
        band_by_component=dict(sorted(band_members.items())),
        union_oracle=dict(candidates=len(union_rows), crossings=union_crossings,
                          mean_gain=round(float(np.mean([r["after"] - r["before"]
                                                         for r in union_rows])), 4) if union_rows else 0.0),
        split_oracle=dict(candidates=len(split_rows), crossings=split_crossings))
    (args.output / "report.json").write_text(json.dumps(report, indent=2))

    print(f"baseline reproduced: PQ {baseline['pq']:.10f} "
          f"TP/FP/FN {baseline['tp']}/{baseline['fp']}/{baseline['fn']}\n")
    print("overlap-graph components:")
    for kind, count in sorted(kinds.items()):
        print(f"  {kind:20s} {count:5d}   of which in the "
              f"{band_low}-{band_high} band: {band_members.get(kind, 0)}")
    print(f"\nunion oracle : {len(union_rows)} truths have 2+ overlapping predictions; "
          f"{union_crossings} cross 0.5 when unioned")
    print(f"split oracle : {len(split_rows)} predictions cover 2+ truths; "
          f"{split_crossings} truths cross 0.5 when split by assignment")
    print(f"\nwrote {args.output / 'report.json'}")


if __name__ == "__main__":
    main()
