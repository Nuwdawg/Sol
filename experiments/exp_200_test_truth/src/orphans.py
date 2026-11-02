"""exp200x: filaments exp169 misses that two other architectures both find, added per record.

exp169 scores only exp139's full-view candidates, so a filament exp139 never proposes cannot be
kept. Swin-B (a Mask2Former with another backbone) and the two YOLO models are independent
detectors. Swin-B's uncovered detections alone did not pay per record (C3 at 0.90: -0.0009). This
asks whether agreement pays: a Swin-B full-view detection (class probability >= 0.5, as on test)
that exp169's instances cover by less than 10% is added only when a YOLO instance from either pool
overlaps it at IoU > 0.5 with score >= tau_y, and its own score is >= tau_b. Added instances are
painted by Swin-B score into unclaimed pixels (`additions.add_uncovered`). No constant is fitted
except the two thresholds, chosen by exp168's two folds (seed 2026).

    python -m experiments.exp_200_test_truth.src.orphans --annotations ... --predictions val_predictions.json
        --swinb SHARD0 ... SHARD4 --yolo yolo11m=... --yolo yolov8l=... --output orphans.json
"""
from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import numpy as np
from pycocotools import mask as mu

from experiments.exp_033_inference.src.sweep import records_for
from experiments.exp_159_yolo_votes.src.votes import load_pool
from experiments.exp_199_swinb_votes.src.votes import merge_shards
from experiments.exp_200_test_truth.src.additions import add_uncovered
from experiments.exp_200_test_truth.src.hypotheses import as_bytes, match, total

FLOOR = 0.5
SWINB_TAUS = (0.5, 0.6, 0.7, 0.8, 0.9)
YOLO_TAUS = (0.05, 0.2, 0.35, 0.5, 0.65)


def yolo_agreement(candidates: list[dict], pools: list[list[dict]]) -> np.ndarray:
    """Per candidate: the best score among YOLO instances of either pool at IoU > 0.5 with it."""
    best = np.zeros(len(candidates))
    if not candidates:
        return best
    rles = [as_bytes(c["rle"]) for c in candidates]
    for pool in pools:
        if not pool:
            continue
        others = [as_bytes(e["rle"]) for e in pool]
        iou = np.asarray(mu.iou(rles, others, [0] * len(others))).reshape(len(rles), len(others))
        scores = np.array([e["score"] for e in pool])
        best = np.maximum(best, np.where(iou.max(axis=1) > 0.5, scores[iou.argmax(axis=1)], 0.0))
    return best


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True, help="hypotheses.py's validation predictions")
    parser.add_argument("--key", default="exp169")
    parser.add_argument("--swinb", type=Path, nargs="+", required=True)
    parser.add_argument("--yolo", action="append", default=[])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    records, _ = records_for(args.annotations)
    names = sorted(records)
    passes = {n: [[as_bytes(t) for t in truth] for _, truth in sorted(records[n])] for n in names}
    rng = np.random.default_rng(2026)
    order = rng.permutation(len(names))
    folds = [set(names[i] for i in order[:len(names) // 2]), set(names[i] for i in order[len(names) // 2:])]
    base = {n: [as_bytes(r) for r in json.loads(args.predictions.read_text())[args.key][n]] for n in names}
    swinb, _ = merge_shards(args.swinb, names)
    pools = [load_pool(Path(p)) for _, p in (s.split("=", 1) for s in sorted(args.yolo))]
    cands, agree = {}, {}
    for n in names:
        cands[n] = [dict(score=c["score"], rle=c["rle"]) for c in swinb[n]["full"] if c["p"] >= FLOOR]
        agree[n] = yolo_agreement(cands[n], [p.get(n, []) for p in pools])

    def scored(preds):
        rows = {n: [match(preds[n], t) for t in passes[n]] for n in names}
        return (total([r for n in names for r in rows[n]])["pq"], [total([r for n in f for r in rows[n]])["pq"] for f in folds])

    reference, reference_halves = scored(base)
    print("EXP169", json.dumps(dict(pq=reference, halves=reference_halves)), flush=True)
    rows = []
    for tau_b, tau_y in itertools.product(SWINB_TAUS, YOLO_TAUS):
        preds, added = {}, 0
        for n in names:
            entries = [c for c, y in zip(cands[n], agree[n]) if y >= tau_y]
            extra = add_uncovered(base[n], entries, tau_b)
            added += len(extra)
            preds[n] = base[n] + extra
        whole, halves = scored(preds)
        rows.append(dict(tau_b=tau_b, tau_y=tau_y, added=added, pq=whole, halves=halves,
                         gain=whole - reference, gains=[h - r for h, r in zip(halves, reference_halves)]))
        print(json.dumps(dict(tau_b=tau_b, tau_y=tau_y, added=added, gain=round(whole - reference, 5),
                              gains=[round(h - r, 5) for h, r in zip(halves, reference_halves)])), flush=True)
    picks = []
    for chooser in (0, 1):
        best = max(rows, key=lambda r: r["gains"][chooser])
        picks.append(dict(chosen_on=chooser, rule=[best["tau_b"], best["tau_y"]], held_out_gain=best["gains"][1 - chooser]))
    print("CV", json.dumps(dict(picks=picks, mean_held_out_gain=float(np.mean([p["held_out_gain"] for p in picks])))), flush=True)
    args.output.write_text(json.dumps(dict(reference=reference, rows=rows, cv=picks), indent=1))


if __name__ == "__main__":
    main()
