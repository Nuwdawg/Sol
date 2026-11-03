"""exp200ag: an instance-dependent threshold for the remnant + rescue rule (per record).

The deployed rule keeps an instance when its exp169 score s >= 0.68 (rescue band [0.60, 0.68) with
Swin-B support >= 0.95). Here the threshold moves with one property of the instance:
  t_i = t0 + beta x z_i,
z_i one standardised feature: log area, elongation, distance from the disk centre (in disk radii),
darkness against the local background, or Swin-B's continuous support. That is a two-parameter
rescoring; exp149's many-feature logistic and boosted combiners overfitted, so each feature is
tried alone and (t0, beta) is fitted on one of exp168's two photograph folds and scored on the
other. Painting order stays the exp169 score.

    python -m experiments.exp_200_test_truth.src.instance_adaptive --annotations ... --cache ... --shift ...
        --ingredients val_ingredients.pkl --swinb SHARD0 ... SHARD4 --images VAL_IMAGES --output out.json
"""
from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import cv2
import numpy as np

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

T0 = tuple(round(t, 2) for t in np.arange(0.60, 0.761, 0.02))
BETAS = (-0.09, -0.06, -0.03, 0.0, 0.03, 0.06, 0.09)
BAND, TAU, FRACTION = 0.08, 0.95, 0.7


def pq_of(sums) -> float:
    tp, fp, fn, iou = sums
    return (iou / tp) * (tp / (tp + 0.5 * fp + 0.5 * fn)) if tp else 0.0


def instance_features(gray: np.ndarray, crops, b: np.ndarray) -> dict:
    disk = gray > 0.25 * np.percentile(gray, 99)
    ys, xs = np.nonzero(disk)
    cy, cx, radius = ys.mean(), xs.mean(), np.sqrt(disk.sum() / np.pi)
    blur = cv2.GaussianBlur(gray, (0, 0), 24)
    out = dict(log_area=[], elongation=[], radial=[], darkness=[], swinb=list(map(float, b)))
    for crop in crops:
        m = crop.mask.astype(bool)
        h, w = m.shape
        py, px = np.nonzero(m)
        area = max(len(py), 1)
        out["log_area"].append(float(np.log(area)))
        if len(py) > 2:
            cov = np.cov(np.stack([py, px]).astype(np.float64))
            ev = np.sort(np.linalg.eigvalsh(cov))
            out["elongation"].append(float(np.log(np.sqrt(max(ev[1], 1e-6) / max(ev[0], 1e-6)))))
        else:
            out["elongation"].append(0.0)
        gy, gx = py.mean() + crop.y0 if len(py) else cy, px.mean() + crop.x0 if len(px) else cx
        out["radial"].append(float(np.hypot(gy - cy, gx - cx) / radius))
        patch = gray[crop.y0:crop.y0 + h, crop.x0:crop.x0 + w]
        back = blur[crop.y0:crop.y0 + h, crop.x0:crop.x0 + w]
        out["darkness"].append(float(np.mean((back[m] - patch[m]) / np.maximum(back[m], 1.0))) if len(py) else 0.0)
    return {k: np.asarray(v, np.float64) for k, v in out.items()}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--shift", type=Path, required=True)
    parser.add_argument("--ingredients", type=Path, required=True)
    parser.add_argument("--swinb", type=Path, nargs="+", required=True)
    parser.add_argument("--images", type=Path, required=True)
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
    exp139 = {n: [as_bytes(r) for _, r in E.paint([(c["score"], E.Crop(c["rle"])) for c in cache[n]["full"]
                                                  if c["score"] >= 0.85], 100)] for n in names}
    s_all, b_all, f_all = {}, {}, {}
    for n in names:
        s_all[n] = np.asarray(scores(prep[n], 4.0, 0.3), np.float64)
        b_all[n] = support(cache[n]["full"], [c for c in swinb[n]["full"] if c["p"] >= 0.5])
        gray = cv2.imread(str(args.images / n), cv2.IMREAD_GRAYSCALE).astype(np.float32)
        f_all[n] = instance_features(gray, prep[n]["crops"], b_all[n])
    keys = sorted(f_all[names[0]])
    # standardise each feature over the candidates that matter (s >= 0.45)
    z = {}
    for key in keys:
        pool = np.concatenate([f_all[n][key][s_all[n] >= 0.45] for n in names])
        mu_, sd = pool.mean(), pool.std() or 1.0
        z[key] = {n: (f_all[n][key] - mu_) / sd for n in names}
    memo = {n: {} for n in names}

    def sums_for(n, idx):
        if idx not in memo[n]:
            kept = [(float(s_all[n][i]), prep[n]["crops"][i]) for i in idx]
            final = [as_bytes(r) for r in paint_fraction(kept, ORDERS["score_desc"], FRACTION)]
            final = final + guard(exp139[n], final)
            rows = [match(final, truth) for truth in passes[n]]
            memo[n][idx] = tuple(sum(r[k] for r in rows) for k in ("tp", "fp", "fn", "iou_sum"))
        return memo[n][idx]

    def kept_idx(n, key, t0, beta):
        s, b = s_all[n], b_all[n]
        t = t0 + beta * (z[key][n] if key else 0.0)
        return tuple(i for i in range(len(s)) if s[i] >= t[i] or (t[i] - BAND <= s[i] < t[i] and b[i] >= TAU)) \
            if key else tuple(i for i in range(len(s)) if s[i] >= t0 or (t0 - BAND <= s[i] < t0 and b[i] >= TAU))

    table = {}

    def pq_rule(key, t0, beta, subset):
        k = (key, t0, beta)
        if k not in table:
            table[k] = {n: sums_for(n, kept_idx(n, key, t0, beta)) for n in names}
        return pq_of(tuple(sum(table[k][n][j] for n in subset) for j in range(4)))

    base = pq_rule(None, 0.68, 0.0, names)
    print("BASE", round(base, 5), flush=True)
    out = dict(base=base, rows={})
    for key in [None] + keys:
        betas = BETAS if key else (0.0,)
        grid = [(t0, beta) for t0 in T0 for beta in betas]
        held = {}
        params = []
        for k, f in enumerate(folds):
            t0, beta = max(grid, key=lambda p: pq_rule(key, p[0], p[1], f))
            params.append((t0, beta))
            for n in folds[1 - k]:
                held[n] = table[(key, t0, beta)][n]
        cv = pq_of(tuple(sum(held[n][j] for n in names) for j in range(4)))
        whole = max(grid, key=lambda p: pq_rule(key, p[0], p[1], names))
        row = dict(cv=cv, fold_params=params, in_sample=pq_rule(key, *whole, names), in_sample_params=whole,
                   halves_in_sample=[pq_rule(key, *whole, f) for f in folds])
        out["rows"][str(key)] = row
        print("INST", json.dumps(dict(feature=key, cv=round(cv, 5), in_sample=round(row["in_sample"], 5),
                                      whole=whole, fold_params=params,
                                      halves=[round(h, 5) for h in row["halves_in_sample"]])), flush=True)
    args.output.write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
