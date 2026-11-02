"""exp200y: merge exp169 fragments that another architecture sees as one filament, per record.

With the truth in hand, merging the exp169 fragments that make up one annotated filament would add
+0.007 per record (11 events; `shape_post`'s oracle in the README). The test-time signal used here
is another model's instance: when two or more exp169 instances lie mostly inside one Swin-B or YOLO
instance O (each with at least half its pixels in O), none of them alone matches O (IoU <= 0.5),
and their union does (IoU > 0.5), the fragments are replaced by one instance: their union, plus the
pixels of O between them that no other instance claims. `need` is how many of the three sources
(Swin-B, YOLO11m, YOLOv8l) must propose the same merge. Nothing is fitted; the YOLO score floor
and `need` are the only settings.

    python -m experiments.exp_200_test_truth.src.merges --annotations ... --predictions val_predictions.json
        --swinb SHARD0 ... SHARD4 --yolo yolo11m=... --yolo yolov8l=... --output merges.json
"""
from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import numpy as np
from pycocotools import mask as mu

from experiments.exp_033_inference.src.sweep import records_for
from experiments.exp_159_yolo_votes.src.votes import load_pool
from experiments.exp_199_swinb_votes.src.votes import merge_shards
from experiments.exp_200_test_truth.src.hypotheses import as_bytes, match, total

FLOOR = 0.5


def proposals(preds: list[dict], others: list[dict]) -> dict[tuple[int, ...], dict]:
    """{fragment indices: the other instance's mask} for every merge one source proposes."""
    out = {}
    if len(preds) < 2 or not others:
        return out
    areas = np.array([mu.area(p) for p in preds], float)
    for o in others:
        inter = np.array([mu.area(mu.merge([p, o], intersect=True)) for p in preds], float)
        parts = [i for i in range(len(preds)) if inter[i] >= 0.5 * areas[i]]
        if len(parts) < 2:
            continue
        single = np.asarray(mu.iou([preds[i] for i in parts], [o], [0])).ravel()
        if (single > 0.5).any():
            continue
        union = mu.merge([preds[i] for i in parts])
        if float(mu.iou([union], [o], [0])[0][0]) > 0.5:
            out[tuple(parts)] = o
    return out


def merged(preds: list[dict], plan: dict[tuple[int, ...], dict]) -> list[dict]:
    """Apply non-conflicting merges: each fragment set becomes its union plus O's unclaimed pixels."""
    used, out = set(), []
    claimed = mu.decode(mu.merge(preds)) if len(preds) > 1 else (mu.decode(preds[0]) if preds else None)
    groups = []
    for parts, o in sorted(plan.items(), key=lambda kv: -len(kv[0])):
        if used & set(parts):
            continue
        used |= set(parts)
        union = mu.decode(mu.merge([preds[i] for i in parts]))
        bridge = mu.decode(o) & (1 - claimed)
        groups.append((min(parts), mu.encode(np.asfortranarray(union | bridge))))
        claimed = claimed | bridge
    keep = {i: p for i, p in enumerate(preds) if i not in used}
    for first, rle in groups:
        keep[first] = rle
    return [keep[i] for i in sorted(keep)]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--key", default="exp169")
    parser.add_argument("--swinb", type=Path, nargs="+", required=True)
    parser.add_argument("--yolo", action="append", default=[])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    records, _ = records_for(args.annotations)
    names = sorted(records)
    passes = {n: [[as_bytes(t) for t in truth] for _, truth in sorted(records[n])] for n in names}
    rng = np.random.default_rng(2026)
    order = rng.permutation(len(names))
    folds = [set(names[i] for i in order[:len(names) // 2]), set(names[i] for i in order[len(names) // 2:])]
    base = {n: [as_bytes(r) for r in json.loads(args.predictions.read_text())[args.key][n]] for n in names}
    swinb, _ = merge_shards(args.swinb, names)
    pools = [load_pool(Path(p)) for _, p in (s.split("=", 1) for s in sorted(args.yolo))]

    def scored(preds):
        rows = {n: [match(preds[n], t) for t in passes[n]] for n in names}
        return (total([r for n in names for r in rows[n]])["pq"], [total([r for n in f for r in rows[n]])["pq"] for f in folds])

    reference, reference_halves = scored(base)
    print("EXP169", json.dumps(dict(pq=reference, halves=reference_halves)), flush=True)
    rows = []
    for yolo_floor, need in itertools.product((0.25, 0.5), (1, 2)):
        preds, events = {}, 0
        for n in names:
            sources = [[as_bytes(c["rle"]) for c in swinb[n]["full"] if c["p"] >= FLOOR]]
            sources += [[as_bytes(e["rle"]) for e in p.get(n, []) if e["score"] >= yolo_floor] for p in pools]
            votes, masks = {}, {}
            for source in sources:
                for parts, o in proposals(base[n], source).items():
                    votes[parts] = votes.get(parts, 0) + 1
                    masks.setdefault(parts, o)
            plan = {parts: masks[parts] for parts, v in votes.items() if v >= need}
            events += len(plan)
            preds[n] = merged(base[n], plan) if plan else base[n]
        whole, halves = scored(preds)
        rows.append(dict(yolo_floor=yolo_floor, need=need, merges=events, pq=whole, halves=halves,
                         gain=whole - reference, gains=[h - r for h, r in zip(halves, reference_halves)]))
        print(json.dumps(dict(yolo_floor=yolo_floor, need=need, merges=events, gain=round(whole - reference, 5),
                              gains=[round(h - r, 5) for h, r in zip(halves, reference_halves)])), flush=True)
    args.output.write_text(json.dumps(dict(reference=reference, rows=rows), indent=1))


if __name__ == "__main__":
    main()
