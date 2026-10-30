"""exp200h: re-tune exp169's rule against pooled one-to-one truth, with exp168's cross-validation.

exp168 chose exp169's rule (views d4+shift, YOLO weight 4, exponent 0.3, threshold 0.68, area 100)
against per-record validation PQ. If the leaderboard scores pooled truth one-to-one (matching.py,
history.py), the rule was tuned to the wrong target. This searches the same family against the
pooled objective and reports both objectives for every configuration, and the cross-validated
gain of the pooled choice over exp169's on exp168's two folds (seed 2026).

Painting is greedy in descending score, so the instances kept at a threshold are a prefix of the
instances kept at any lower one. One painting pass per photograph and (weight, exponent, area)
therefore serves every threshold.

    python -m experiments.exp_200_test_truth.src.tune --annotations ... --cache ... --shift ...
        --yolo yolov8l=... --yolo yolo11m=... --ingredients val_ingredients.pkl --part 0/3 --sums part0.pkl
    (one process per part), then the same command with --merge part0.pkl part1.pkl part2.pkl --output tune.json
"""
from __future__ import annotations

import argparse
import itertools
import json
import pickle
from pathlib import Path

import numpy as np
from pycocotools import mask as mu

from experiments.exp_033_inference.src.sweep import records_for
from experiments.exp_149_evidence_fusion.src import evidence as E
from experiments.exp_157_shift_views.src.views import merged_cache
from experiments.exp_159_yolo_votes.src.votes import load_pool
from experiments.exp_168_joint_tuning.src.joint import prepared, scores
from experiments.exp_200_test_truth.src.hypotheses import as_bytes

WEIGHTS = (0.0, 1.0, 2.0, 4.0, 8.0)
EXPONENTS = (0.3, 0.5, 0.7)
AREAS = (40, 100)
THRESHOLDS = tuple(round(t, 3) for t in np.arange(0.30, 0.901, 0.02))
EXP169 = (4.0, 0.3, 0.68, 100)


def ingredients(args, records) -> dict:
    if args.ingredients.exists():
        return pickle.loads(args.ingredients.read_bytes())
    cache = merged_cache(args.cache, args.shift)
    pools = [load_pool(Path(p)) for _, p in (s.split("=", 1) for s in sorted(args.yolo))]
    out = {}
    for name in sorted(records):
        out[name] = prepared(cache[name], [pl.get(name, []) for pl in pools], ("d4", "shift"))
    args.ingredients.write_bytes(pickle.dumps(out))
    return out


def painted_prefix(entries: list[tuple[float, "E.Crop"]], min_area: int) -> tuple[list[float], list[dict]]:
    """E.paint over all entries, keeping each kept instance's score: thresholds select a prefix."""
    claimed = np.zeros((E.SIZE, E.SIZE), np.uint8)
    kept_scores, kept = [], []
    for score, crop in sorted(entries, key=lambda item: -item[0]):
        if crop.mask.size == 0:
            continue
        h, w = crop.mask.shape
        window = claimed[crop.y0:crop.y0 + h, crop.x0:crop.x0 + w]
        remaining = crop.mask & (1 - window)
        if int(remaining.sum()) < min_area:
            continue
        window |= remaining
        full = np.zeros((E.SIZE, E.SIZE), np.uint8)
        full[crop.y0:crop.y0 + h, crop.x0:crop.x0 + w] = remaining
        kept_scores.append(float(score))
        kept.append(mu.encode(np.asfortranarray(full)))
    return kept_scores, kept


def greedy(iou: np.ndarray) -> tuple[int, float]:
    """One-to-one matching above 0.5, highest IoU first: (true positives, IoU sum)."""
    pairs = sorted(((iou[i, j], i, j) for i, j in zip(*np.nonzero(iou > 0.5))), reverse=True)
    used_p, used_t, tp, total = set(), set(), 0, 0.0
    for value, i, j in pairs:
        if i in used_p or j in used_t:
            continue
        used_p.add(i); used_t.add(j); tp += 1; total += float(value)
    return tp, total


def photo_rows(kept_scores, kept, passes) -> dict:
    """{threshold: (pooled [iou, tp, fp, fn], record [iou, tp, fp, fn])} for one photograph."""
    flat = [t for truth in passes for t in truth]
    owner = [r for r, truth in enumerate(passes) for _ in truth]
    iou = (np.asarray(mu.iou(kept, flat, [0] * len(flat))).reshape(len(kept), len(flat))
           if kept and flat else np.zeros((len(kept), len(flat))))
    out = {}
    for t in THRESHOLDS:
        k = sum(1 for s in kept_scores if s >= t)
        sub = iou[:k]
        tp, total = greedy(sub)
        pooled = np.array([total, tp, k - tp, len(flat) - tp])
        record = np.zeros(4)
        for r, truth in enumerate(passes):
            cols = [j for j, o in enumerate(owner) if o == r]
            hit = sub[:, cols] > 0.5 if cols else np.zeros((k, 0), bool)
            rtp = int(hit.sum())                      # disjoint truth within one record: unique matches
            record += (float(sub[:, cols][hit].sum()) if cols else 0.0, rtp, int((hit.sum(1) == 0).sum()),
                       len(cols) - rtp)
        out[t] = (pooled, record)
    return out


def pq(acc) -> float:
    iou, tp, fp, fn = acc
    return float(iou / (tp + 0.5 * fp + 0.5 * fn)) if tp else 0.0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--shift", type=Path, required=True)
    parser.add_argument("--yolo", action="append", default=[])
    parser.add_argument("--ingredients", type=Path, required=True)
    parser.add_argument("--part", default="0/1", help="i/k: this process's share of the grid")
    parser.add_argument("--sums", type=Path, default=None, help="write this part's sums here and stop")
    parser.add_argument("--merge", type=Path, nargs="*", default=[], help="parts to merge into the report")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    records, _ = records_for(args.annotations)
    names = sorted(records)
    rng = np.random.default_rng(2026)                 # exp168's folds
    order = rng.permutation(len(names))
    folds = [set(names[i] for i in order[:len(names) // 2]), set(names[i] for i in order[len(names) // 2:])]
    if args.merge:
        sums = {}
        for part in args.merge:
            sums.update(pickle.loads(part.read_bytes()))
        report(sums, names, folds, args.output)
        return
    passes = {n: [[as_bytes(t) for t in truth] for _, truth in sorted(records[n])] for n in names}
    prep = ingredients(args, records)
    index, count = (int(v) for v in args.part.split("/"))
    grid = [c for k, c in enumerate(itertools.product(WEIGHTS, EXPONENTS, AREAS)) if k % count == index]
    sums = {}                                          # (w, a, t, area) -> {photo: (pooled, record)}
    for w, a, area in grid:
        per_photo = {}
        for n in names:
            entries = [(s, c) for s, c in zip(map(float, scores(prep[n], w, a)), prep[n]["crops"])
                       if s >= THRESHOLDS[0]]         # lower ones never enter a prefix
            per_photo[n] = photo_rows(*painted_prefix(entries, area), passes[n])
        for t in THRESHOLDS:
            sums[(w, a, t, area)] = {n: per_photo[n][t] for n in names}
        best = max(THRESHOLDS, key=lambda t: pq(sum(v[0] for v in sums[(w, a, t, area)].values())))
        print(f"w {w:g} a {a:g} area {area}: best pooled at t {best}: "
              f"pooled {pq(sum(v[0] for v in sums[(w, a, best, area)].values())):.4f} "
              f"record {pq(sum(v[1] for v in sums[(w, a, best, area)].values())):.4f}", flush=True)
    if args.sums:
        args.sums.write_bytes(pickle.dumps(sums))
    else:
        report(sums, names, folds, args.output)


def report(sums: dict, names: list, folds: list, output: Path) -> None:
    def score(key, photos, which):
        return pq(sum(sums[key][n][which] for n in photos))

    table = [dict(weight=k[0], exponent=k[1], conf=k[2], min_area=k[3], pooled=score(k, names, 0),
                  record=score(k, names, 1)) for k in sums]
    best_pooled = max(sums, key=lambda k: score(k, names, 0))
    cv_pooled, cv_exp169 = [], []
    for train, held in ((folds[0], folds[1]), (folds[1], folds[0])):
        choice = max(sums, key=lambda k: score(k, train, 0))
        cv_pooled.append(sum(sums[choice][n][0] for n in held))
        cv_exp169.append(sum(sums[EXP169][n][0] for n in held))
    summary = dict(
        exp169=dict(pooled=score(EXP169, names, 0), record=score(EXP169, names, 1)),
        best_pooled=dict(key=dict(weight=best_pooled[0], exponent=best_pooled[1], conf=best_pooled[2],
                                  min_area=best_pooled[3]),
                         pooled=score(best_pooled, names, 0), record=score(best_pooled, names, 1)),
        cross_validated=dict(pooled_choice=pq(sum(cv_pooled)), exp169=pq(sum(cv_exp169)),
                             gain=pq(sum(cv_pooled)) - pq(sum(cv_exp169))),
        table=table)
    print("EXP169", json.dumps(summary["exp169"]), flush=True)
    if abs(summary["exp169"]["record"] - 0.48894859995182627) > 1e-9:      # CLAUDE.md section 5
        raise SystemExit("exp169's per-record validation PQ was not reproduced")
    print("BEST_POOLED", json.dumps(summary["best_pooled"]), flush=True)
    print("CV", json.dumps(summary["cross_validated"]), flush=True)
    output.write_text(json.dumps(summary, indent=1, default=float))


if __name__ == "__main__":
    main()
