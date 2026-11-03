"""exp200af: does the best score threshold depend on the photograph? (per record, learning-light)

The remnant + rescue file keeps an instance when its exp169 score is >= 0.68 (or in the rescue
band [0.60, 0.68) with Swin-B support >= 0.95). One global threshold assumes every photograph
wants the same trade-off. Annotators may draw more faint filaments on sharp, active photographs
and fewer on blurred ones; if so the best threshold moves with something visible in the photo.

PQ = sum IoU / (TP + FP/2 + FN/2). Per photograph TP + FN is fixed (the truths) and TP + FP is
(predictions x records), so at the optimum PQ* the photograph's contribution is separable:
  sum IoU - (PQ*/2) x (TP + FP).
That gives (1) an in-sample oracle threshold per photograph (an upper bound, optimistic), and
(2) a one-feature rule t = t0 + beta x z(feature), fitted on one of exp168's two photograph folds
and scored on the other. The rescue band moves with the threshold: [t - 0.08, t).

    python -m experiments.exp_200_test_truth.src.adaptive --annotations ... --cache ... --shift ...
        --ingredients val_ingredients.pkl --swinb SHARD0 ... SHARD4 --images VAL_IMAGES --output adaptive.json
"""
from __future__ import annotations

import argparse
import itertools
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

GRID = tuple(round(t, 2) for t in np.arange(0.58, 0.821, 0.02))
BAND, TAU, FRACTION = 0.08, 0.95, 0.7
BETAS = (-0.08, -0.06, -0.04, -0.02, 0.0, 0.02, 0.04, 0.06, 0.08)


def pq_of(sums) -> float:
    tp, fp, fn, iou = sums
    return (iou / tp) * (tp / (tp + 0.5 * fp + 0.5 * fn)) if tp else 0.0


def photo_features(path: Path, s: np.ndarray, crops) -> dict:
    gray = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE).astype(np.float32)
    disk = gray > 0.25 * np.percentile(gray, 99)
    inside = gray[disk]
    lap = cv2.Laplacian(gray, cv2.CV_32F, ksize=3)[disk]
    blur = cv2.GaussianBlur(gray, (0, 0), 16)
    local = (gray - blur)[disk]
    kept = s >= 0.68
    return dict(
        n_kept=float(kept.sum()),
        n_band=float(((s >= 0.45) & (s < 0.68)).sum()),
        mean_top=float(s[s >= 0.45].mean()) if (s >= 0.45).any() else 0.0,
        kept_area=float(sum(int(c.mask.sum()) for c, k in zip(crops, kept) if k)),
        contrast=float(inside.std() / max(inside.mean(), 1.0)),
        sharpness=float(np.log(lap.var() + 1e-6)),
        dark_local=float(np.mean(local < -0.08 * inside.mean())),
        radius=float(np.sqrt(disk.sum() / np.pi)),
    )


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
    # table[n][t] = (tp, fp, fn, iou_sum) summed over the photograph's records
    table, feats = {}, {}
    for n in names:
        s = np.asarray(scores(prep[n], 4.0, 0.3), np.float64)
        b = support(cache[n]["full"], [c for c in swinb[n]["full"] if c["p"] >= 0.5])
        crops = prep[n]["crops"]
        memo, table[n] = {}, {}
        for t in GRID:
            idx = tuple(i for i in range(len(s)) if s[i] >= t or (t - BAND <= s[i] < t and b[i] >= TAU))
            if idx not in memo:
                kept = [(float(s[i]), crops[i]) for i in idx]
                final = [as_bytes(r) for r in paint_fraction(kept, ORDERS["score_desc"], FRACTION)]
                final = final + guard(exp139[n], final)
                rows = [match(final, truth) for truth in passes[n]]
                memo[idx] = tuple(sum(r[k] for r in rows) for k in ("tp", "fp", "fn", "iou_sum"))
            table[n][t] = memo[idx]
        feats[n] = photo_features(args.images / n, s, crops)

    def pq_rule(choice: dict, subset) -> float:
        return pq_of(tuple(sum(table[n][choice[n]][k] for n in subset) for k in range(4)))

    out = dict(grid=list(GRID), fixed={}, oracle={}, adaptive={})
    for t in GRID:
        out["fixed"][str(t)] = dict(pq=pq_rule({n: t for n in names}, names),
                                    halves=[pq_rule({n: t for n in names}, f) for f in folds])
    base = out["fixed"]["0.68"]
    print("FIXED", json.dumps({k: round(v["pq"], 5) for k, v in out["fixed"].items()}), flush=True)

    # in-sample oracle: per photograph maximise sum IoU - lambda (TP + FP), lambda = PQ / 2
    lam = base["pq"] / 2
    for _ in range(10):
        choice = {n: max(GRID, key=lambda t: table[n][t][3] - lam * (table[n][t][0] + table[n][t][1])) for n in names}
        lam = pq_rule(choice, names) / 2
    out["oracle"] = dict(pq=pq_rule(choice, names), thresholds={n: choice[n] for n in names})
    print("ORACLE", round(out["oracle"]["pq"], 5), flush=True)

    # one-feature rules, fitted on one fold and scored on the other
    keys = sorted(next(iter(feats.values())))
    z = {}
    for key in keys:
        values = np.array([feats[n][key] for n in names])
        mu_, sd = values.mean(), values.std() or 1.0
        z[key] = {n: (feats[n][key] - mu_) / sd for n in names}
    grid = np.array(GRID)

    def snap(value: float) -> float:
        return float(grid[np.abs(grid - value).argmin()])

    def rule(key, t0, beta):
        return {n: snap(t0 + beta * (z[key][n] if key else 0.0)) for n in names}

    def fit(key, subset):
        betas = BETAS if key else (0.0,)
        return max(((t0, beta) for t0 in GRID for beta in betas), key=lambda p: pq_rule(rule(key, *p), subset))

    for key in [None] + keys:
        held = {}
        params = []
        for k, f in enumerate(folds):
            t0, beta = fit(key, f)
            params.append((t0, beta))
            choice = rule(key, t0, beta)
            for n in folds[1 - k]:
                held[n] = choice[n]
        cv = pq_rule(held, names)
        whole = fit(key, names)
        out["adaptive"][str(key)] = dict(cv=cv, fold_params=params, in_sample=pq_rule(rule(key, *whole), names),
                                         in_sample_params=whole)
        print("ADAPT", json.dumps(dict(feature=key, cv=round(cv, 5), in_sample=round(out["adaptive"][str(key)]["in_sample"], 5),
                                       fold_params=params, whole=whole)), flush=True)
    out["features"] = feats
    args.output.write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
