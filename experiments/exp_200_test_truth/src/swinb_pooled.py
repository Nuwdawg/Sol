"""exp200l: Swin-B (exp190b epoch 32) as a voter in exp169's rule, under pooled one-to-one truth.

exp199 found that Swin-B votes lower per-record validation PQ at every weight. Under pooled truth a
filament any annotator drew is a hit, so a second strong model may help where it did not before. Two
forms: Swin-B's eight views (as exp199), and its full view alone, the form that could be computed
on the test photographs on a CPU. Every row is scored pooled and per record on exp168's two halves,
against exp169 at its own threshold.

    python -m experiments.exp_200_test_truth.src.swinb_pooled --annotations ... --cache ... --shift ...
        --yolo yolov8l=... --yolo yolo11m=... --swinb SHARD0 ... SHARD4 --output swinb_pooled.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from experiments.exp_033_inference.src.sweep import records_for
from experiments.exp_157_shift_views.src.views import merged_cache
from experiments.exp_159_yolo_votes.src.votes import load_pool
from experiments.exp_185_m2f_votes.src.votes import scores
from experiments.exp_195_family_votes.src.votes import prepared
from experiments.exp_199_swinb_votes.src.votes import merge_shards
from experiments.exp_200_test_truth.src.hypotheses import as_bytes
from experiments.exp_200_test_truth.src.tune import painted_prefix, pq, greedy
from pycocotools import mask as mu

GRID = tuple(round(t, 2) for t in np.arange(0.40, 0.861, 0.02))


def rows_for(entries, passes):
    """{threshold: (pooled, record)} sums for one photograph, painting once."""
    ks, kept = painted_prefix(entries, 100)
    flat = [t for truth in passes for t in truth]
    owner = [r for r, truth in enumerate(passes) for _ in truth]
    iou = (np.asarray(mu.iou(kept, flat, [0] * len(flat))).reshape(len(kept), len(flat))
           if kept and flat else np.zeros((len(kept), len(flat))))
    out = {}
    for t in GRID:
        k = sum(1 for s in ks if s >= t)
        sub = iou[:k]
        tp, total = greedy(sub)
        pooled = np.array([total, tp, k - tp, len(flat) - tp])
        record = np.zeros(4)
        for r in range(len(passes)):
            cols = [j for j, o in enumerate(owner) if o == r]
            hit = sub[:, cols] > 0.5 if cols else np.zeros((k, 0), bool)
            rtp = int(hit.sum())
            record += (float(sub[:, cols][hit].sum()) if cols else 0.0, rtp, int((hit.sum(1) == 0).sum()), len(cols) - rtp)
        out[t] = (pooled, record)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--shift", type=Path, required=True)
    parser.add_argument("--yolo", action="append", default=[])
    parser.add_argument("--swinb", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    records, _ = records_for(args.annotations)
    names = sorted(records)
    passes = {n: [[as_bytes(t) for t in truth] for _, truth in sorted(records[n])] for n in names}
    cache = merged_cache(args.cache, args.shift)
    pools = [load_pool(Path(p)) for _, p in (s.split("=", 1) for s in sorted(args.yolo))]
    swinb, _ = merge_shards(args.swinb, names)
    rng = np.random.default_rng(2026)
    order = rng.permutation(len(names))
    folds = [set(names[i] for i in order[:len(names) // 2]), set(names[i] for i in order[len(names) // 2:])]
    forms = {"eight views": lambda n: swinb[n], "full view only": lambda n: {"full": swinb[n]["full"]}}
    results, base = [], None
    for form, other in forms.items():
        for with_masks in (False, True):
            prep = {n: prepared(cache[n], [p.get(n, []) for p in pools], [other(n)], ("d4", "shift"), with_masks)
                    for n in names}
            weights = (0.0, 1.0, 2.0, 4.0) if form == "eight views" else (4.0, 8.0, 16.0, 32.0)
            for yolo_w in (4.0, 0.0):
                for m in weights:
                    rows = {n: rows_for(list(zip(map(float, scores(prep[n], yolo_w, m, 0.3)), prep[n]["crops"])),
                                        passes[n]) for n in names}
                    for t in GRID:
                        pooled = [pq(sum(rows[n][t][0] for n in f)) for f in folds]
                        record = pq(sum(rows[n][t][1] for n in names))
                        row = dict(form=form, masks=with_masks, yolo_weight=yolo_w, swinb_weight=m, conf=t,
                                   pooled_all=pq(sum(rows[n][t][0] for n in names)), pooled_folds=pooled, record=record)
                        results.append(row)
                        if form == "eight views" and not with_masks and yolo_w == 4.0 and m == 0.0 and t == 0.68:
                            base = row
    print("EXP169", json.dumps(base), flush=True)
    if abs(base["record"] - 0.48894859995182627) > 1e-9:
        raise SystemExit("exp169's per-record validation PQ was not reproduced")
    for r in results:
        r["gain_folds"] = [r["pooled_folds"][i] - base["pooled_folds"][i] for i in range(2)]
    best = sorted(results, key=lambda r: -min(r["gain_folds"]))[:15]
    for r in best:
        print(f"{r['form']:15s} masks {str(r['masks']):5s} yolo {r['yolo_weight']:g} swinb {r['swinb_weight']:g} "
              f"conf {r['conf']:.2f}: pooled {r['pooled_all']:.4f} gain A {r['gain_folds'][0]:+.4f} "
              f"B {r['gain_folds'][1]:+.4f}  record {r['record']:.4f}", flush=True)
    args.output.write_text(json.dumps(dict(exp169=base, rows=results), indent=1))


if __name__ == "__main__":
    main()
