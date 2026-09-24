"""exp118: how much recall sits in the candidate pool below the confidence floor?

The repair families are closed (exp114-117). The arithmetic in exp117 addendum 2
says recovering 45 of 465 misses at IoU 0.70 reaches 0.472 validation, which is
worth more than every repair result combined. This measures whether those
recoveries are already present in the pool and merely discarded by the confidence
floor.

Why the bar is lower here than for any repair gate
--------------------------------------------------
Admitting k true instances at mean IoU q, together with m false ones, improves PQ
exactly when

    k*q > 0.5 * PQ * (k + m)

because a recovered miss turns an FN (half a unit of denominator) into a TP (one
unit) while adding q to the numerator, and a false instance adds only half a unit
of denominator. At PQ 0.4559 and q = 0.70 that is m < 2.07k, so a verifier needs
about 33% precision to pay. Every repair gate needed roughly 80%. That difference
is the whole reason this family is worth testing.

What is measured
----------------
1. the score distribution of the pool, so the floors tested are real;
2. a plain floor sweep - what simply lowering the floor does, with no verifier.
   exp062 already found 0.35 optimal, so this is expected to fall, and it is the
   control;
3. the oracle admit - sub-threshold candidates are admitted only when they match a
   currently unmatched truth at IoU >= 0.5. That is the ceiling for "lower the
   floor and verify perfectly", and it is never deployable.

Every emission goes through the frozen path (erode, paint_panoptic, encode) so the
numbers are comparable with every other experiment in this log.
"""
from __future__ import annotations

import argparse
import json
import time
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
from pycocotools import mask as mu

from experiments.exp_033_inference.src.sweep import encode, pq_rle, records_for
from shared.utils import aggregate_pq, paint_panoptic

KERNEL = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
BASELINE_PQ = 0.4558638687292126
BASELINE_TP, BASELINE_FP, BASELINE_FN = 860, 397, 465
MATCH = 0.5


def paint(selected, min_area):
    """The frozen emission path, given already-eroded (score, mask) pairs."""
    return [encode(mask) for _, mask, _ in paint_panoptic(list(selected), min_area=min_area)]


def score_all(per_record):
    rows = [pq_rle(predicted, truths) for predicted, truths in per_record]
    aggregate = aggregate_pq(rows)
    aggregate["iou_sum"] = float(sum(row["iou_sum"] for row in rows))
    return aggregate


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pool", type=Path,
                        default=Path("data/exp033_cached_tta/candidates_0.5_0.5.json"))
    parser.add_argument("--output", type=Path, default=Path("data/exp118_recall_ceiling"))
    parser.add_argument("--grow", type=int, default=-1)
    parser.add_argument("--min-area", type=int, default=300)
    parser.add_argument("--base-conf", type=float, default=0.35)
    parser.add_argument("--floors", type=float, nargs="+",
                        default=[0.35, 0.30, 0.25, 0.20, 0.15, 0.10])
    args = parser.parse_args()
    cv2.setNumThreads(2)
    args.output.mkdir(parents=True, exist_ok=True)
    started = time.time()

    annotations = next(Path(".").rglob("MAGFiLO_1.0_Annotations_kaggle2026_train.json"))
    records, _ = records_for(annotations)
    pool = json.loads(args.pool.read_text())

    scores = np.array([entry["score"] for photograph in records for entry in pool[photograph]])
    print(f"pool: {len(scores)} candidates over {len(records)} photographs")
    print(f"  min {scores.min():.4f}  max {scores.max():.4f}")
    for edge in (0.10, 0.15, 0.20, 0.25, 0.30, 0.35):
        print(f"  score >= {edge:.2f}: {int((scores >= edge).sum()):5d}"
              f"   in [{edge:.2f}, {args.base_conf:.2f}): "
              f"{int(((scores >= edge) & (scores < args.base_conf)).sum()):5d}")
    print()

    # One decode pass per photograph; every floor and the oracle reuse it.
    emissions = {floor: defaultdict(list) for floor in args.floors}
    oracle = defaultdict(list)
    admitted_total = 0
    for photograph, passes in records.items():
        entries = pool[photograph]
        prepared = []
        for entry in entries:
            mask = mu.decode(entry["rle"])
            if args.grow < 0:
                mask = cv2.erode(mask, KERNEL, iterations=-args.grow)
            prepared.append((float(entry["score"]), mask))
        for floor in args.floors:
            chosen = [(s, m) for s, m in prepared if s >= floor]
            emitted = paint(chosen, args.min_area)
            for image_id, truths in passes:
                emissions[floor][image_id] = (emitted, truths)

        # ORACLE: which sub-threshold candidates would a perfect verifier admit?
        base = [(s, m) for s, m in prepared if s >= args.base_conf]
        base_rle = paint(base, args.min_area)
        unmatched = []
        for _, truths in passes:
            if not truths:
                continue
            best = (np.asarray(mu.iou(base_rle, truths, [0] * len(truths))).max(axis=0)
                    if base_rle else np.zeros(len(truths)))
            unmatched.extend(truths[g] for g in range(len(truths)) if best[g] < MATCH)
        extra = []
        if unmatched:
            for s, m in prepared:
                if s >= args.base_conf or int(m.sum()) < args.min_area:
                    continue
                rle = encode(m)
                if float(np.asarray(mu.iou([rle], unmatched, [0] * len(unmatched))).max()) >= MATCH:
                    extra.append((s, m))
        admitted_total += len(extra)
        oracle_rle = paint(base + extra, args.min_area) if extra else base_rle
        for image_id, truths in passes:
            oracle[image_id] = (oracle_rle, truths)

    print(f"{'floor':>6} {'PQ':>10} {'TP':>5} {'FP':>5} {'FN':>5} "
          f"{'dTP':>5} {'dFP':>6}  added precision")
    report = dict(experiment_id="exp118", baseline_pq=BASELINE_PQ, floors={}, oracle={})
    base_row = None
    for floor in args.floors:
        result = score_all(list(emissions[floor].values()))
        if base_row is None and abs(floor - args.base_conf) < 1e-9:
            base_row = result
            assert abs(result["pq"] - BASELINE_PQ) < 1e-9, result["pq"]
        delta_tp = result["tp"] - (base_row["tp"] if base_row else result["tp"])
        delta_fp = result["fp"] - (base_row["fp"] if base_row else result["fp"])
        added = delta_tp + delta_fp
        precision = f"{delta_tp / added:.3f}" if added > 0 else "     -"
        print(f"{floor:6.2f} {result['pq']:10.6f} {result['tp']:5d} {result['fp']:5d} "
              f"{result['fn']:5d} {delta_tp:+5d} {delta_fp:+6d}  {precision}")
        report["floors"][f"{floor:.2f}"] = {k: result[k] for k in
                                            ("pq", "sq", "rq", "tp", "fp", "fn", "iou_sum")}

    result = score_all(list(oracle.values()))
    gain = result["pq"] - BASELINE_PQ
    print(f"\nORACLE admit (never deployable): PQ {result['pq']:.6f}  {gain:+.6f}")
    print(f"  TP {result['tp']} FP {result['fp']} FN {result['fn']}  "
          f"candidates admitted {admitted_total}")
    report["oracle"] = dict(admitted=admitted_total, delta_pq=round(gain, 6),
                            **{k: result[k] for k in ("pq", "tp", "fp", "fn", "iou_sum")})

    recovered = result["tp"] - BASELINE_TP
    if recovered > 0:
        baseline_iou_sum = BASELINE_PQ * (BASELINE_TP + BASELINE_FP / 2 + BASELINE_FN / 2)
        quality = (result["iou_sum"] - baseline_iou_sum) / recovered
        affordable = recovered * (quality / (0.5 * BASELINE_PQ) - 1)
        print(f"  recovered {recovered} misses at mean IoU {quality:.3f}")
        print(f"  a verifier may admit up to {affordable:.0f} false positives "
              f"alongside them and still gain")
        report["oracle"]["recovered"] = recovered
        report["oracle"]["mean_iou"] = round(float(quality), 4)
        report["oracle"]["affordable_false_positives"] = int(affordable)

    (args.output / "report.json").write_text(json.dumps(report, indent=2))
    print(f"\nwrote {args.output / 'report.json'} in {time.time() - started:.0f}s")


if __name__ == "__main__":
    main()
