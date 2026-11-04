"""exp201d: add a second Swin-S's large, confident instances that the first pick leaves uncovered.

The review of exp201's file showed the retrained models finding long, clearly dark filaments
(4,000-14,500 px) that the first pick (remnant + rescue on exp139) does not hold. This tests the
idea on validation with exp190b Swin-S e50, the same recipe trained without the validation
photographs:
  final = first pick + every second-model full-view instance (painted at score >= c, area >= 100)
          with area >= A that the first pick's instances cover by less than 10%
          (exp199's guard rule, with the second model as the source).
Grid: c in {0.85, 0.90, 0.95}, A in {3000, 5000, 8000}. Chosen on one of exp168's photograph folds
and scored on the other. Bar (as for exp201b): +0.003 in-sample and cross-validated over 0.49095,
with no half lower.

    python -m experiments.exp_201_alldata.src.large_orphans --annotations ... --cache ... --shift ...
        --ingredients val_ingredients.pkl --swinb SHARD0 ... SHARD4 --other P1 ... P106 --output out.json
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
from experiments.exp_157_shift_views.src.views import merged_cache
from experiments.exp_168_joint_tuning.src.joint import scores
from experiments.exp_199_swinb_votes.src.guard import guard
from experiments.exp_199_swinb_votes.src.votes import merge_shards
from experiments.exp_200_test_truth.src.hypotheses import as_bytes, match
from experiments.exp_200_test_truth.src.paint_order import ORDERS
from experiments.exp_200_test_truth.src.remnants import paint_fraction
from experiments.exp_200_test_truth.src.swinb_gate import support
from experiments.exp_201_alldata.src.transfer import load_other, pq_of

CONFS, AREAS = (0.85, 0.90, 0.95), (3000, 5000, 8000)


def add_orphans(final: list[dict], other: list[dict], min_area: int) -> list[dict]:
    """exp199's guard rule with another model as the source: re-add large instances covered < 10%."""
    big = [r for r in other if int(mu.area(r)) >= min_area]
    return final + guard(big, final) if big else final


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--shift", type=Path, required=True)
    parser.add_argument("--ingredients", type=Path, required=True)
    parser.add_argument("--swinb", type=Path, nargs="+", required=True)
    parser.add_argument("--other", type=Path, nargs="+", required=True)
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
    other = load_other(args.other, names)
    swinb, _ = merge_shards(args.swinb, names)

    def sums(final, n):
        rows = [match(final, t) for t in passes[n]]
        return tuple(sum(r[k] for r in rows) for k in ("tp", "fp", "fn", "iou_sum"))

    table = {}
    for n in names:
        base = [as_bytes(r) for _, r in E.paint([(c["score"], E.Crop(c["rle"])) for c in cache[n]["full"]
                                                 if c["score"] >= 0.85], 100)]
        s = scores(prep[n], 4.0, 0.3)
        s_b = support(cache[n]["full"], [c for c in swinb[n]["full"] if c["p"] >= 0.5])
        kept = [(float(v), c) for v, b, c in zip(s, s_b, prep[n]["crops"]) if v >= 0.68 or (0.60 <= v < 0.68 and b >= 0.95)]
        combo = [as_bytes(r) for r in paint_fraction(kept, ORDERS["score_desc"], 0.7)]
        first = combo + guard(base, combo)
        table[n] = {"none": sums(first, n)}
        for c in CONFS:
            painted = [as_bytes(r) for _, r in E.paint([(x["score"], E.Crop(x["rle"])) for x in other[n]["full"]
                                                        if x["score"] >= c], 100)]
            for a in AREAS:
                table[n][(c, a)] = sums(add_orphans(first, painted, a), n)
    keys = ["none"] + [(c, a) for c in CONFS for a in AREAS]

    def pq(key, subset):
        return pq_of(tuple(sum(table[n][key][j] for n in subset) for j in range(4)))

    out = {"rows": {}}
    for key in keys:
        out["rows"][str(key)] = dict(pq=pq(key, names), halves=[pq(key, f) for f in folds])
        print("ROW", json.dumps(dict(key=str(key), pq=round(pq(key, names), 5),
                                     halves=[round(pq(key, f), 5) for f in folds])), flush=True)
    held = {}
    choices = []
    for k, f in enumerate(folds):
        best = max(keys, key=lambda kk: pq(kk, f))
        choices.append(str(best))
        for n in folds[1 - k]:
            held[n] = table[n][best]
    out["cv"] = pq_of(tuple(sum(held[n][j] for n in names) for j in range(4)))
    out["fold_choices"] = choices
    print("CV", round(out["cv"], 5), choices, flush=True)
    args.output.write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
