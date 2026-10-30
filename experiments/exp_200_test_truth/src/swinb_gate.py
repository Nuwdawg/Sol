"""exp200r: Swin-B as a gate on exp169's borderline instances, judged per record.

exp199 put Swin-B into exp169's score as one more voter; every unit of weight lowered per-record PQ,
because the vote moves every candidate's score. This uses it only near exp169's threshold, where the
YOLO support decides. exp169's score cannot reach 0.68 without YOLO support (11 view scores over 19
slots), and the Mask2Former instances it drops in the band 0.60-0.68 match their annotator 17-34% of
the time. Two rules, each with Swin-B support s_B = the score of the best Swin-B full-view candidate
(class probability >= 0.5, as on test) at IoU > 0.5 with the instance's exp139 mask, else 0:
  rescue: also keep instances scoring in [low, 0.68) whose s_B >= tau;
  veto:   drop kept instances scoring in [0.68, high) whose s_B < tau.
Scored per record (the leaderboard's scoring, README correction), with exp168's two folds (seed
2026): each fold picks the best rule on itself and is scored on the other, against exp169 on the
same halves.

    python -m experiments.exp_200_test_truth.src.swinb_gate --annotations ... --cache ... --shift ...
        --ingredients val_ingredients.pkl --swinb SHARD0 ... SHARD4 --output swinb_gate.json
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
from experiments.exp_199_swinb_votes.src.votes import merge_shards
from experiments.exp_200_test_truth.src.hypotheses import as_bytes, match, total

EXP169 = 0.48894859995182627
FLOOR = 0.5
RULES = ([("none", None, None)]
         + [("rescue", low, tau) for low in (0.50, 0.55, 0.60) for tau in (0.98, 0.95, 0.90, 0.85, 0.80, 0.70)]
         + [("veto", high, tau) for high in (0.72, 0.76, 0.80) for tau in (0.01, 0.30, 0.50, 0.70)])


def support(full: list[dict], swinb: list[dict]) -> np.ndarray:
    rles = [E.as_rle(c["rle"]) for c in full]
    others = [E.as_rle(c["rle"]) for c in swinb]
    if not rles or not others:
        return np.zeros(len(rles))
    iou = np.asarray(mu.iou(rles, others, [0] * len(others))).reshape(len(rles), len(others))
    best = iou.argmax(axis=1)
    values = np.array([c["score"] for c in swinb])
    return np.where(iou.max(axis=1) > 0.5, values[best], 0.0)


def keep(score: float, s_b: float, rule) -> bool:
    kind, edge, tau = rule
    if kind == "rescue" and edge <= score < 0.68:
        return s_b >= tau
    if kind == "veto" and 0.68 <= score < edge:
        return s_b >= tau
    return score >= 0.68


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--shift", type=Path, required=True)
    parser.add_argument("--ingredients", type=Path, required=True, help="tune.py's exp169 ingredients")
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
    inputs = {}
    for n in names:
        sc = scores(prep[n], 4.0, 0.3)
        s_b = support(cache[n]["full"], [c for c in swinb[n]["full"] if c["p"] >= FLOOR])
        inputs[n] = list(zip(map(float, sc), map(float, s_b), prep[n]["crops"]))

    results = []
    for rule in RULES:
        rows = {n: [] for n in names}
        for n in names:
            pred = [as_bytes(r) for _, r in E.paint([(s, c) for s, b, c in inputs[n] if keep(s, b, rule)], 100)]
            rows[n] = [match(pred, t) for t in passes[n]]
        whole = total([r for n in names for r in rows[n]])["pq"]
        halves = [total([r for n in f for r in rows[n]])["pq"] for f in folds]
        if rule[0] == "none" and abs(whole - EXP169) > 1e-9:
            raise SystemExit(f"exp169 not reproduced: {whole}")
        results.append(dict(rule=rule, pq=whole, halves=halves))
        print(json.dumps(dict(rule=rule, pq=round(whole, 5), halves=[round(h, 5) for h in halves])), flush=True)
    base = results[0]["halves"]
    picks = []
    for chooser in (0, 1):
        best = max(results, key=lambda r: r["halves"][chooser])
        picks.append(dict(chosen_on=chooser, rule=best["rule"], held_out_gain=best["halves"][1 - chooser] - base[1 - chooser]))
    print("CV", json.dumps(picks), flush=True)
    args.output.write_text(json.dumps(dict(results=results, cv=picks), indent=1))


if __name__ == "__main__":
    main()
