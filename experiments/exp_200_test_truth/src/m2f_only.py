"""exp200t: a Mask2Former-only rule (exp139's views plus Swin-B as a voter, no YOLO), per record.

The public board treats the two families of submission differently. Raw single-view Mask2Former
submissions (exp139 e40, e50, their soup, eight-view fusion) all fit public = 0.8615 x per-record
validation, and the YOLO-voted rules (exp162, exp162b, exp169) fit 0.837-0.849 x; one common
factor misses some of either group by a rounding step. So a rule without YOLO may reach the board
at a higher rate. This measures how high such a rule gets on validation: exp169's family
s^a x mean^(1-a), mean = (sum of exp139's 11 view scores + m x s_B) / (11 + m), no YOLO, where s_B is
the score of Swin-B's best full-view partner at IoU > 0.5 (class probability >= 0.5, what was computed
on test). exp185's `model_support` divides by eight views, which would dilute a full-view-only voter
eightfold, so s_B is computed directly (`swinb_gate.support`).
m = 0 is exp139's views alone (rule B's family). Every (m, a, threshold) is scored per record;
exp168's two folds (seed 2026) choose out of sample.

    python -m experiments.exp_200_test_truth.src.m2f_only --annotations ... --cache ... --shift ...
        --swinb SHARD0 ... SHARD4 --output m2f_only.json
"""
from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import numpy as np

from experiments.exp_033_inference.src.sweep import records_for
from experiments.exp_157_shift_views.src.views import merged_cache
from experiments.exp_168_joint_tuning.src.joint import prepared
from experiments.exp_199_swinb_votes.src.votes import merge_shards
from experiments.exp_200_test_truth.src.hypotheses import as_bytes, match, total
from experiments.exp_200_test_truth.src.swinb_gate import support
from experiments.exp_200_test_truth.src.tune import painted_prefix

FLOOR = 0.5
THRESHOLDS = tuple(round(t, 2) for t in np.arange(0.50, 0.921, 0.02))
GRID = tuple(itertools.product((0.0, 2.0, 4.0, 8.0, 16.0), (0.3, 0.5, 0.7)))


def scores(p: dict, s_b: np.ndarray, m: float, a: float) -> np.ndarray:
    mean = (p["total"] + m * s_b) / (p["views"] + m)
    return np.power(p["s"], a) * np.power(np.maximum(mean, 1e-12), 1.0 - a)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--shift", type=Path, required=True)
    parser.add_argument("--swinb", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    records, _ = records_for(args.annotations)
    names = sorted(records)
    passes = {n: [[as_bytes(t) for t in truth] for _, truth in sorted(records[n])] for n in names}
    rng = np.random.default_rng(2026)
    order = rng.permutation(len(names))
    folds = [set(names[i] for i in order[:len(names) // 2]), set(names[i] for i in order[len(names) // 2:])]
    cache = merged_cache(args.cache, args.shift)
    swinb, _ = merge_shards(args.swinb, names)
    prep = {n: prepared(cache[n], [], ("d4", "shift")) for n in names}
    s_b = {n: support(cache[n]["full"], [c for c in swinb[n]["full"] if c["p"] >= FLOOR]) for n in names}

    rows = []
    for m, a in GRID:
        sums = {t: {} for t in THRESHOLDS}
        for n in names:
            entries = [(s, c) for s, c in zip(map(float, scores(prep[n], s_b[n], m, a)), prep[n]["crops"]) if s >= THRESHOLDS[0]]
            kept_scores, kept = painted_prefix(entries, 100)
            for t in THRESHOLDS:
                pred = [as_bytes(r) for s, r in zip(kept_scores, kept) if s >= t]
                sums[t][n] = [match(pred, truth) for truth in passes[n]]
        for t in THRESHOLDS:
            whole = total([r for n in names for r in sums[t][n]])["pq"]
            halves = [total([r for n in f for r in sums[t][n]])["pq"] for f in folds]
            rows.append(dict(swinb_weight=m, exponent=a, threshold=t, pq=whole, halves=halves))
        best = max((r for r in rows if r["swinb_weight"] == m and r["exponent"] == a), key=lambda r: r["pq"])
        print(json.dumps(dict(swinb_weight=m, exponent=a, best_threshold=best["threshold"], pq=round(best["pq"], 5),
                              halves=[round(h, 5) for h in best["halves"]])), flush=True)
    picks = []
    for chooser in (0, 1):
        best = max(rows, key=lambda r: r["halves"][chooser])
        picks.append(dict(chosen_on=chooser, rule=[best["swinb_weight"], best["exponent"], best["threshold"]],
                          held_out=best["halves"][1 - chooser]))
    for m in (0.0,):
        sub = [r for r in rows if r["swinb_weight"] == m]
        for chooser in (0, 1):
            best = max(sub, key=lambda r: r["halves"][chooser])
            picks.append(dict(chosen_on=chooser, restricted_to_m=m, rule=[best["swinb_weight"], best["exponent"], best["threshold"]],
                              held_out=best["halves"][1 - chooser]))
    print("CV", json.dumps(picks), flush=True)
    args.output.write_text(json.dumps(dict(rows=rows, cv=picks), indent=1))


if __name__ == "__main__":
    main()
