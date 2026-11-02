"""exp200ac: exp169 + guard with learning-free fixes stacked, judged per record.

Pieces (each measured alone first):
  remnant f: painting in descending score, drop an instance that keeps less than f of its own vote
             mask (remnants.py: f 0.7 gives +0.0010, both halves non-negative);
  rescue:    also keep Mask2Former instances scoring in [0.60, 0.68) whose Swin-B full-view partner
             scores >= 0.95 (swinb_gate.py: +0.0010, both halves positive);
  guard:     exp199's re-addition of large exp139 instances the vote deleted (the deployed file).
Scored per record on validation with exp168's two folds (seed 2026).

    python -m experiments.exp_200_test_truth.src.combo --annotations ... --cache ... --shift ...
        --ingredients val_ingredients.pkl --swinb SHARD0 ... SHARD4 --output combo.json
"""
from __future__ import annotations

import argparse
import itertools
import json
import pickle
from pathlib import Path

import numpy as np

from experiments.exp_033_inference.src.sweep import records_for
from experiments.exp_149_evidence_fusion.src import evidence as E
from experiments.exp_157_shift_views.src.views import merged_cache
from experiments.exp_168_joint_tuning.src.joint import scores
from experiments.exp_199_swinb_votes.src.guard import guard
from experiments.exp_199_swinb_votes.src.votes import merge_shards
from experiments.exp_200_test_truth.src.hypotheses import as_bytes, match, total
from experiments.exp_200_test_truth.src.paint_order import ORDERS
from experiments.exp_200_test_truth.src.remnants import paint_fraction
from experiments.exp_200_test_truth.src.swinb_gate import support

FRACTIONS = (0.0, 0.5, 0.6, 0.7, 0.8, 0.9)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--shift", type=Path, required=True)
    parser.add_argument("--ingredients", type=Path, required=True)
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
    cache = merged_cache(args.cache, args.shift)
    swinb, _ = merge_shards(args.swinb, names)
    s169 = {n: scores(prep[n], 4.0, 0.3) for n in names}
    s_b = {n: support(cache[n]["full"], [c for c in swinb[n]["full"] if c["p"] >= 0.5]) for n in names}
    exp139 = {n: [as_bytes(r) for _, r in E.paint([(c["score"], E.Crop(c["rle"])) for c in cache[n]["full"]
                                                  if c["score"] >= 0.85], 100)] for n in names}
    rows = []
    for fraction, rescue, guarded in itertools.product(FRACTIONS, (False, True), (False, True)):
        by = {}
        for n in names:
            kept = [(float(s), c) for s, b, c in zip(s169[n], s_b[n], prep[n]["crops"])
                    if s >= 0.68 or (rescue and 0.60 <= s < 0.68 and b >= 0.95)]
            final = [as_bytes(r) for r in paint_fraction(kept, ORDERS["score_desc"], fraction)]
            if guarded:
                final = final + guard(exp139[n], final)
            by[n] = [match(final, t) for t in passes[n]]
        whole = total([r for n in names for r in by[n]])["pq"]
        halves = [total([r for n in f for r in by[n]])["pq"] for f in folds]
        rows.append(dict(fraction=fraction, rescue=rescue, guard=guarded, pq=whole, halves=halves))
        print(json.dumps(dict(fraction=fraction, rescue=rescue, guard=guarded, pq=round(whole, 5),
                              halves=[round(h, 5) for h in halves])), flush=True)
    args.output.write_text(json.dumps(rows, indent=1))


if __name__ == "__main__":
    main()
