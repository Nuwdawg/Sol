"""exp200aa: a SAM-style stability score for exp169's instances, judged per record.

Segment Anything ranks masks by stability: the IoU between the mask cut at logit +1 and at logit -1.
exp147's cache stores both cuts (rle_high, rle_low) for every exp139 full-view candidate, so
stability = area(high) / area(low) costs nothing and learns nothing. A filament whose extent the
model is sure of should match an annotator's more often than one whose edge is a soft ramp.
Rules: score' = score x stability^beta (beta 0, 0.5, 1, 2; beta 0 is exp169), threshold chosen by
exp168's two folds (seed 2026); paint by score' (or, `--small-first`, by area ascending).

    python -m experiments.exp_200_test_truth.src.stability --annotations ... --cache exp147_val_cache.json
        --ingredients val_ingredients.pkl --output stability.json
"""
from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import numpy as np
from pycocotools import mask as mu

from experiments.exp_033_inference.src.sweep import records_for
from experiments.exp_149_evidence_fusion.src import evidence as E
from experiments.exp_168_joint_tuning.src.joint import scores
from experiments.exp_200_test_truth.src.hypotheses import as_bytes, match, total
from experiments.exp_200_test_truth.src.paint_order import ORDERS, paint

BETAS = (0.0, 0.5, 1.0, 2.0)
THRESHOLDS = tuple(round(t, 3) for t in np.arange(0.40, 0.801, 0.02))


def stability(entry: dict) -> np.ndarray:
    out = []
    for c in entry["full"]:
        low, high = mu.area(E.as_rle(c["rle_low"])), mu.area(E.as_rle(c["rle_high"]))
        out.append(high / low if low else 0.0)
    return np.array(out)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True, help="exp147 validation cache (full view with rle_low/high)")
    parser.add_argument("--ingredients", type=Path, required=True)
    parser.add_argument("--small-first", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    records, _ = records_for(args.annotations)
    names = sorted(records)
    passes = {n: [[as_bytes(t) for t in truth] for _, truth in sorted(records[n])] for n in names}
    rng = np.random.default_rng(2026)
    order = rng.permutation(len(names))
    folds = [set(names[i] for i in order[:len(names) // 2]), set(names[i] for i in order[len(names) // 2:])]
    prep = pickle.loads(args.ingredients.read_bytes())
    cache = E.load_cache(args.cache)
    stab = {n: stability(cache[n]) for n in names}
    key = ORDERS["area_asc" if args.small_first else "score_desc"]
    rows = []
    for beta in BETAS:
        for t in THRESHOLDS:
            by = {}
            for n in names:
                s = scores(prep[n], 4.0, 0.3) * np.power(stab[n], beta)
                kept = [(float(v), c) for v, c in zip(s, prep[n]["crops"]) if v >= t]
                by[n] = [match(paint(kept, key), truth) for truth in passes[n]]
            whole = total([r for n in names for r in by[n]])["pq"]
            halves = [total([r for n in f for r in by[n]])["pq"] for f in folds]
            rows.append(dict(beta=beta, threshold=t, pq=whole, halves=halves))
        best = max((r for r in rows if r["beta"] == beta), key=lambda r: r["pq"])
        print(json.dumps(dict(beta=beta, threshold=best["threshold"], pq=round(best["pq"], 5),
                              halves=[round(h, 5) for h in best["halves"]])), flush=True)
    picks = []
    for chooser in (0, 1):
        best = max(rows, key=lambda r: r["halves"][chooser])
        picks.append(dict(chosen_on=chooser, rule=[best["beta"], best["threshold"]], held_out=best["halves"][1 - chooser]))
    print("CV", json.dumps(dict(picks=picks, mean_held_out=float(np.mean([p["held_out"] for p in picks])))), flush=True)
    args.output.write_text(json.dumps(dict(rows=rows, cv=picks), indent=1))


if __name__ == "__main__":
    main()
