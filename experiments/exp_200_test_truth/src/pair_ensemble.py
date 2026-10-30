"""exp200s: a plain two-model ensemble of exp139 e40 and Swin-B e32 full-view instances, per record.

On the public board the raw single-view Mask2Former submissions (exp139 e40 0.41, e50 0.40, the
e40/e50 soup 0.40, eight-view fusion 0.40) all sit one rounding step above public = 0.84 x
validation, and the val-tuned YOLO vote rules (exp162, exp162b, exp169) one step below it: rule
tuning lifts validation more than the board. exp199 added Swin-B to exp169's tuned rule as a voter
and lost. This asks the model-level question instead: does averaging two independently trained
Mask2Former models, with no YOLO and one tuned constant, beat either alone?

Instances of the two full views are paired greedily by mask IoU > 0.5. A pair scores the mean of
its two scores, an unpaired instance half its score. The mask is exp139's, or Swin-B's when the
instance is Swin-B's alone. Keep score >= t, paint by score, area >= 100. Swin-B candidates use class
probability >= 0.5, as on test. t is chosen by exp168's two folds (seed 2026).

    python -m experiments.exp_200_test_truth.src.pair_ensemble --annotations ... --cache ... --shift ...
        --swinb SHARD0 ... SHARD4 --output pair_ensemble.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from pycocotools import mask as mu

from experiments.exp_033_inference.src.sweep import records_for
from experiments.exp_149_evidence_fusion.src import evidence as E
from experiments.exp_157_shift_views.src.views import merged_cache
from experiments.exp_199_swinb_votes.src.votes import merge_shards
from experiments.exp_200_test_truth.src.hypotheses import as_bytes, match, total
from experiments.exp_200_test_truth.src.tune import painted_prefix

FLOOR = 0.5
THRESHOLDS = tuple(round(t, 2) for t in np.arange(0.40, 0.951, 0.01))
EXP139, SWINB = 0.47052, 0.46975          # recorded single-view PQ: conf 0.85 / 0.90, area 100


def pairs(a: list[dict], b: list[dict]) -> list[tuple[float, str | dict]]:
    """[(fused score, mask)] for exp139 instances a and Swin-B instances b."""
    if not a or not b:
        return [(c["score"] / 2, c["rle"]) for c in a + b]
    iou = np.asarray(mu.iou([E.as_rle(c["rle"]) for c in a], [E.as_rle(c["rle"]) for c in b],
                            [0] * len(b))).reshape(len(a), len(b))
    order = sorted(((iou[i, j], i, j) for i, j in zip(*np.nonzero(iou > 0.5))), reverse=True)
    used_a, used_b, out = set(), set(), []
    for _, i, j in order:
        if i in used_a or j in used_b:
            continue
        used_a.add(i); used_b.add(j)
        out.append(((a[i]["score"] + b[j]["score"]) / 2, a[i]["rle"]))
    out += [(a[i]["score"] / 2, a[i]["rle"]) for i in range(len(a)) if i not in used_a]
    out += [(b[j]["score"] / 2, b[j]["rle"]) for j in range(len(b)) if j not in used_b]
    return out


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
    a = {n: cache[n]["full"] for n in names}
    b = {n: [c for c in swinb[n]["full"] if c["p"] >= FLOOR] for n in names}

    def scored(entries_by_photo, thresholds):
        """{threshold: (pq, [pq of each fold])}; one painting per photograph serves every threshold."""
        rows = {t: {} for t in thresholds}
        for n in names:
            kept_scores, kept = painted_prefix([(s, E.Crop(m)) for s, m in entries_by_photo[n]
                                                if s >= min(thresholds)], 100)
            for t in thresholds:
                pred = [as_bytes(r) for s, r in zip(kept_scores, kept) if s >= t]
                rows[t][n] = [match(pred, truth) for truth in passes[n]]
        return {t: (total([r for n in names for r in rows[t][n]])["pq"],
                    [total([r for n in f for r in rows[t][n]])["pq"] for f in folds]) for t in thresholds}

    single_a = scored({n: [(c["score"], c["rle"]) for c in a[n]] for n in names}, (0.85,))[0.85][0]
    single_b = scored({n: [(c["score"], c["rle"]) for c in b[n]] for n in names}, (0.90,))[0.90][0]
    if abs(single_a - EXP139) > 5e-5 or abs(single_b - SWINB) > 5e-5:
        raise SystemExit(f"single models not reproduced: {single_a:.5f} {single_b:.5f}")
    print("SINGLE", json.dumps(dict(exp139=single_a, swinb=single_b)), flush=True)
    fused = {n: pairs(a[n], b[n]) for n in names}
    rows = []
    for t, (whole, halves) in scored(fused, THRESHOLDS).items():
        rows.append(dict(threshold=t, pq=whole, halves=halves))
        print(json.dumps(dict(threshold=t, pq=round(whole, 5), halves=[round(h, 5) for h in halves])), flush=True)
    picks = []
    for chooser in (0, 1):
        best = max(rows, key=lambda r: r["halves"][chooser])
        picks.append(dict(chosen_on=chooser, threshold=best["threshold"], held_out=best["halves"][1 - chooser]))
    cv = (picks[0]["held_out"] + picks[1]["held_out"]) / 2
    print("CV", json.dumps(dict(picks=picks, mean_held_out=cv)), flush=True)
    args.output.write_text(json.dumps(dict(single=dict(exp139=single_a, swinb=single_b), rows=rows, cv=picks), indent=1))


if __name__ == "__main__":
    main()
