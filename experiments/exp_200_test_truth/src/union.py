"""exp200m: add Swin-B's own confident detections that the rule's instances do not cover.

Under pooled truth a filament any annotator drew is a hit. exp169's instances all come from exp139.
Swin-B (exp190b epoch 32, validation 0.4698 alone) finds filaments exp139 misses: its full-view
candidates at score >= 0.80 that no exp169 instance covers match a still-unmatched pooled truth 40%
of the time at IoU 0.67, against a break-even near 31%.

Rule: after the base rule paints its instances (and exp199's guard), Swin-B full-view candidates with
score >= tau and less than 10% covered by the base instances are painted, in descending score, into
unclaimed pixels; a remnant under 100 px is dropped. `add_uncovered` (additions.py) does the painting,
so validation and test share it. Scored pooled and per record on exp168's two halves.

    python -m experiments.exp_200_test_truth.src.union --annotations ... --ingredients val_ingredients.pkl
        --cache exp147_val_cache.json --swinb SHARD0 ... SHARD4 --output union.json
"""
from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import numpy as np

from experiments.exp_033_inference.src.sweep import records_for
from experiments.exp_149_evidence_fusion.src import evidence as E
from experiments.exp_168_joint_tuning.src.joint import scores
from experiments.exp_199_swinb_votes.src.guard import guard
from experiments.exp_199_swinb_votes.src.votes import merge_shards
from experiments.exp_200_test_truth.src.additions import add_uncovered
from experiments.exp_200_test_truth.src.hypotheses import as_bytes, match, total
from experiments.exp_200_test_truth.src.matching import one_to_one

BASES = {"exp169": (4.0, 0.3, 0.68), "A": (4.0, 0.3, 0.64), "B": (0.0, 0.5, 0.78)}
TAUS = (None, 0.95, 0.90, 0.85, 0.80, 0.75)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--ingredients", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True, help="exp147 validation cache (exp139's full view)")
    parser.add_argument("--swinb", type=Path, nargs="+", required=True)
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
    swinb, _ = merge_shards(args.swinb, names)
    rows, base_folds = [], None
    for label, (w, a, t) in BASES.items():
        base = {}
        for n in names:
            entries = list(zip(map(float, scores(prep[n], w, a)), prep[n]["crops"]))
            final = [as_bytes(r) for _, r in E.paint([(s, c) for s, c in entries if s >= t], 100)]
            exp139 = [as_bytes(r) for _, r in E.paint([(c["score"], E.Crop(c["rle"])) for c in cache[n]["full"]
                                                       if c["score"] >= 0.85], 100)]
            base[n] = final + guard(exp139, final)
        for tau in TAUS:
            preds = base if tau is None else {
                n: base[n] + add_uncovered(base[n], [dict(score=c["score"], rle=c["rle"]) for c in swinb[n]["full"]], tau)
                for n in names}
            pooled = [total([one_to_one(preds[n], [x for truth in passes[n] for x in truth]) for n in f])["pq"]
                      for f in folds]
            everything = total([one_to_one(preds[n], [x for truth in passes[n] for x in truth]) for n in names])
            record = total([match(preds[n], truth) for n in names for truth in passes[n]])["pq"]
            row = dict(base=label, tau=tau, pooled=everything["pq"], pooled_folds=pooled, record=record,
                       instances=sum(len(v) for v in preds.values()))
            if label == "exp169" and tau is None:
                base_folds = pooled
            row["gain_folds"] = [pooled[i] - base_folds[i] for i in range(2)]
            rows.append(row)
            print(f"{label:6s} + guard + Swin-B >= {tau}: pooled {row['pooled']:.4f} gain A {row['gain_folds'][0]:+.4f} "
                  f"B {row['gain_folds'][1]:+.4f}  record {record:.4f}  instances {row['instances']}", flush=True)
    args.output.write_text(json.dumps(rows, indent=1))


if __name__ == "__main__":
    main()
