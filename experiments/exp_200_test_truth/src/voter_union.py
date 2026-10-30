"""exp200p: Swin-B both as a voter on exp139's candidates and as a source of uncovered detections.

C1 uses Swin-B only for detections no rule instance covers. Its full view can also re-score
exp139's candidates (exp185's `model_support`, as swinb_pooled.py did). On the test photographs Swin-B
was run with class-probability floor 0.5 (swinb_infer.py), so here its validation candidates are
filtered the same way before they vote (CLAUDE.md section 5). Rule: score = s^a x mean^(1-a)
over exp139's views, YOLO (weight w) and Swin-B (weight m); paint at the threshold; exp199's guard; then
Swin-B detections >= 0.90 that the instances leave uncovered. Scored pooled per half and per record,
against exp169 + guard.

    python -m experiments.exp_200_test_truth.src.voter_union --annotations ... --cache ... --shift ...
        --yolo yolov8l=... --yolo yolo11m=... --swinb SHARD0 ... SHARD4 --output voter_union.json
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
from experiments.exp_159_yolo_votes.src.votes import load_pool
from experiments.exp_185_m2f_votes.src.votes import scores
from experiments.exp_195_family_votes.src.votes import prepared
from experiments.exp_199_swinb_votes.src.guard import guard
from experiments.exp_199_swinb_votes.src.votes import merge_shards
from experiments.exp_200_test_truth.src.additions import add_uncovered
from experiments.exp_200_test_truth.src.hypotheses import as_bytes, match, total
from experiments.exp_200_test_truth.src.matching import one_to_one

FLOOR = 0.5                                    # swinb_infer.py's class-probability floor on test
GRID = tuple(round(t, 2) for t in np.arange(0.40, 0.841, 0.04))


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
    rng = np.random.default_rng(2026)
    order = rng.permutation(len(names))
    folds = [set(names[i] for i in order[:len(names) // 2]), set(names[i] for i in order[len(names) // 2:])]
    cache = merged_cache(args.cache, args.shift)
    pools = [load_pool(Path(p)) for _, p in (s.split("=", 1) for s in sorted(args.yolo))]
    swinb, _ = merge_shards(args.swinb, names)
    voters = {n: {"full": [c for c in swinb[n]["full"] if c["p"] >= FLOOR]} for n in names}
    exp139 = {n: [as_bytes(r) for _, r in E.paint([(c["score"], E.Crop(c["rle"])) for c in cache[n]["full"]
                                                   if c["score"] >= 0.85], 100)] for n in names}

    def scored(preds):
        pooled = [total([one_to_one(preds[n], [x for t in passes[n] for x in t]) for n in f])["pq"] for f in folds]
        record = total([match(preds[n], t) for n in names for t in passes[n]])["pq"]
        return pooled, record

    def finish(final_by_photo, tau=0.90):
        out = {}
        for n in names:
            base = final_by_photo[n] + guard(exp139[n], final_by_photo[n])
            out[n] = base + add_uncovered(base, [dict(score=c["score"], rle=c["rle"]) for c in swinb[n]["full"]], tau)
        return out

    rows = []
    reference = None
    for with_masks in (False,):
        prep = {n: prepared(cache[n], [p.get(n, []) for p in pools], [voters[n]], ("d4", "shift"), with_masks)
                for n in names}
        if reference is None:                   # exp169 + guard, the baseline of every gain
            base = {n: [as_bytes(r) for _, r in E.paint([(s, c) for s, c in zip(map(float, scores(prep[n], 4.0, 0.0, 0.3)),
                                                                                    prep[n]["crops"]) if s >= 0.68], 100)]
                    for n in names}
            reference, ref_record = scored({n: base[n] + guard(exp139[n], base[n]) for n in names})
            print("EXP169", json.dumps(dict(pooled=reference, record=ref_record)), flush=True)
        for w, m, a in itertools.product((0.0,), (8.0, 16.0), (0.3, 0.5)):
            entries = {n: list(zip(map(float, scores(prep[n], w, m, a)), prep[n]["crops"])) for n in names}
            for t in GRID:
                final = {n: [as_bytes(r) for _, r in E.paint([(s, c) for s, c in entries[n] if s >= t], 100)] for n in names}
                pooled, record = scored(finish(final))
                gains = [pooled[i] - reference[i] for i in range(2)]
                rows.append(dict(masks=with_masks, yolo_weight=w, swinb_weight=m, exponent=a, conf=t,
                                 pooled=pooled, gains=gains, record=record))
            best = max((r for r in rows if r["masks"] == with_masks and r["yolo_weight"] == w and r["swinb_weight"] == m
                        and r["exponent"] == a), key=lambda r: min(r["gains"]))
            print(f"masks {with_masks} w {w:g} m {m:g} a {a:g}: best conf {best['conf']}: gains A {best['gains'][0]:+.4f} "
                  f"B {best['gains'][1]:+.4f} record {best['record']:.4f}", flush=True)
    args.output.write_text(json.dumps(dict(reference=reference, rows=rows), indent=1))


if __name__ == "__main__":
    main()
