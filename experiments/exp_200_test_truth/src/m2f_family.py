"""exp200v: the Mask2Former-only rule with exp184a's three Mask2Former models as further voters.

m2f_only.py's rule is exp139's 11 views plus Swin-B (full view) as a voter, no YOLO. exp184a cached
three more independently trained Mask2Former models (base e20, control e31, quality e31; validation
PQ 0.452-0.464) on validation and test, full and dihedral views. exp195a found them useless inside
exp169's YOLO rule; Swin-B was useless there too and helps once YOLO is gone. Here:
  mean = (exp139's 11 view scores + m_B x s_B + m_O x (sum of the three models' support)) / (11 + m_B + 3 m_O)
where s_B is Swin-B's best full-view partner score (m2f_only.py) and a model's support is its best
partner score averaged over its eight views (exp185's `model_support`, all eight views present).
Masks stay exp139's view vote. Scored per record; exp168's two folds (seed 2026) choose out of sample.

    python -m experiments.exp_200_test_truth.src.m2f_family --annotations ... --cache ... --shift ...
        --swinb SHARD0 ... SHARD4 --other base=... --other control=... --other quality=... --output m2f_family.json
"""
from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import numpy as np

from experiments.exp_033_inference.src.sweep import records_for
from experiments.exp_149_evidence_fusion.src import evidence as E
from experiments.exp_157_shift_views.src.views import merged_cache
from experiments.exp_168_joint_tuning.src.joint import prepared
from experiments.exp_185_m2f_votes.src.votes import model_support
from experiments.exp_199_swinb_votes.src.votes import merge_shards
from experiments.exp_200_test_truth.src.hypotheses import as_bytes, match, total
from experiments.exp_200_test_truth.src.m2f_only import FLOOR
from experiments.exp_200_test_truth.src.swinb_gate import support
from experiments.exp_200_test_truth.src.tune import painted_prefix

THRESHOLDS = tuple(round(t, 2) for t in np.arange(0.70, 0.921, 0.02))
GRID = tuple(itertools.product((8.0, 16.0), (0.0, 2.0, 4.0, 8.0), (0.3, 0.5)))


def family_scores(p: dict, s_b: np.ndarray, others: list[np.ndarray], m_b: float, m_o: float, a: float) -> np.ndarray:
    numerator = p["total"] + m_b * s_b + m_o * sum(others)
    denominator = p["views"] + m_b + m_o * len(others)
    return np.power(p["s"], a) * np.power(np.maximum(numerator / denominator, 1e-12), 1.0 - a)


def ingredients(cache: dict, swinb_full: dict, others: list[dict], names) -> dict:
    out = {}
    for n in names:
        p = prepared(cache[n], [], ("d4", "shift"))
        s_b = support(cache[n]["full"], [c for c in swinb_full[n] if c["p"] >= FLOOR])
        out[n] = (p, s_b, [model_support(cache[n], other[n])[0] for other in others])
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--shift", type=Path, required=True)
    parser.add_argument("--swinb", type=Path, nargs="+", required=True)
    parser.add_argument("--other", action="append", default=[], help="tag=exp184a validation cache")
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
    others = [E.load_cache(Path(p)) for _, p in (s.split("=", 1) for s in sorted(args.other))]
    if len(others) != 3 or any(sorted(o) != names for o in others):
        raise SystemExit("expected three caches covering the validation photographs")
    ing = ingredients(cache, {n: swinb[n]["full"] for n in names}, others, names)

    rows = []
    for m_b, m_o, a in GRID:
        sums = {t: {} for t in THRESHOLDS}
        for n in names:
            p, s_b, sup = ing[n]
            entries = [(s, c) for s, c in zip(map(float, family_scores(p, s_b, sup, m_b, m_o, a)), p["crops"])
                       if s >= THRESHOLDS[0]]
            kept_scores, kept = painted_prefix(entries, 100)
            for t in THRESHOLDS:
                pred = [as_bytes(r) for s, r in zip(kept_scores, kept) if s >= t]
                sums[t][n] = [match(pred, truth) for truth in passes[n]]
        for t in THRESHOLDS:
            rows.append(dict(m_b=m_b, m_o=m_o, a=a, threshold=t,
                             pq=total([r for n in names for r in sums[t][n]])["pq"],
                             halves=[total([r for n in f for r in sums[t][n]])["pq"] for f in folds]))
        best = max((r for r in rows if (r["m_b"], r["m_o"], r["a"]) == (m_b, m_o, a)), key=lambda r: r["pq"])
        print(json.dumps(dict(m_b=m_b, m_o=m_o, a=a, threshold=best["threshold"], pq=round(best["pq"], 5),
                              halves=[round(h, 5) for h in best["halves"]])), flush=True)
    picks = []
    for chooser in (0, 1):
        best = max(rows, key=lambda r: r["halves"][chooser])
        picks.append(dict(chosen_on=chooser, rule=[best["m_b"], best["m_o"], best["a"], best["threshold"]],
                          held_out=best["halves"][1 - chooser]))
    print("CV", json.dumps(dict(picks=picks, mean_held_out=float(np.mean([p["held_out"] for p in picks])))), flush=True)
    args.output.write_text(json.dumps(dict(rows=rows, cv=picks), indent=1))


if __name__ == "__main__":
    main()
