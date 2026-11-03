"""exp200ae: three ways to fuse an instance's member masks (exp139 views + YOLO partners), per record.

exp169's vote mask keeps a pixel when at least half of the members cover it. Alternatives, all
learning-free:
  eioeu: Nowozin (CVPR 2014) and Ahmed, Tarlow & Batra (ICCV 2015): with p(x) = fraction of members
         covering x, choose per instance the level k maximising expected intersection over expected
         union, sum_{x in M} p / (|M| + sum_{x not in M} p), over the nested masks M_k = {votes >= k};
  sdf:   shape averaging by signed distance functions (the level-set shape-prior construction of
         Leventon, Grimson & Faugeras, CVPR 2000): mean of the members' signed distances, cut at 0;
         boundaries are averaged geometrically, so thin parts survive small offsets between views;
  staple: Warfield, Zou & Wells (IEEE TMI 2004): EM estimate of each member's sensitivity and
         specificity from the members alone, then the posterior truth probability cut at 0.5.
The rest is the remnant + rescue file's rule: exp169's scores, rescue band, remnant filter at 0.7,
guard. Judged per record on validation with exp168's two folds (seed 2026).

    python -m experiments.exp_200_test_truth.src.fusion_math --annotations ... --cache ... --shift ...
        --ingredients val_ingredients.pkl --swinb SHARD0 ... SHARD4 --yolo yolo11m=... --yolo yolov8l=...
        --output fusion_math.json
"""
from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import numpy as np
from pycocotools import mask as mu
from scipy import ndimage

from experiments.exp_033_inference.src.sweep import records_for
from experiments.exp_149_evidence_fusion.src import evidence as E
from experiments.exp_157_shift_views.src.views import VIEWS, as_text, merged_cache
from experiments.exp_159_yolo_votes.src.votes import load_pool
from experiments.exp_166_yolo_mask_votes.src.mask_votes import yolo_partners
from experiments.exp_168_joint_tuning.src.joint import scores
from experiments.exp_199_swinb_votes.src.guard import guard
from experiments.exp_199_swinb_votes.src.votes import merge_shards
from experiments.exp_200_test_truth.src.hypotheses import as_bytes, match, total
from experiments.exp_200_test_truth.src.paint_order import ORDERS
from experiments.exp_200_test_truth.src.remnants import paint_fraction
from experiments.exp_200_test_truth.src.swinb_gate import support

PAD = 4


def groups_for(entry: dict, pools_n: list[list[dict]]) -> list[list[dict]]:
    """exp168's vote members for every full candidate: itself, its best partner in each other view
    (IoU > 0.5), and its eroded YOLO partners."""
    full = entry["full"]
    rles = [E.as_rle(c["rle"]) for c in full]
    groups = [[r] for r in rles]
    for source in ("d4", "shift"):
        for view in VIEWS[source]:
            members = [c for c in entry.get(source, []) if c["view"] == view]
            if not members or not rles:
                continue
            mr = [E.as_rle(c["rle"]) for c in members]
            ious = np.asarray(mu.iou(rles, mr, [0] * len(mr))).reshape(len(rles), len(mr))
            for i in range(len(rles)):
                j = int(ious[i].argmax())
                if ious[i, j] > 0.5:
                    groups[i].append(mr[j])
    for pool in pools_n:
        for i, extra in enumerate(yolo_partners(entry, pool, 1)):
            if extra is not None:
                groups[i].append(extra)
    return groups


def crop_stack(group: list[dict]) -> tuple[np.ndarray, tuple[int, int]]:
    union = mu.merge(group) if len(group) > 1 else group[0]
    x, y, w, h = (int(v) for v in mu.toBbox(union))
    y0, x0 = max(0, y - PAD), max(0, x - PAD)
    y1, x1 = min(E.SIZE, y + h + PAD), min(E.SIZE, x + w + PAD)
    stack = np.stack([mu.decode(g)[y0:y1, x0:x1] for g in group]).astype(np.uint8)
    return stack, (y0, x0)


def fuse(group: list[dict], method: str) -> dict:
    if len(group) == 1:
        return group[0]
    stack, (y0, x0) = crop_stack(group)
    n = len(group)
    votes = stack.sum(0)
    if method == "majority":
        out = votes >= int(np.ceil(0.5 * n))
    elif method == "eioeu":
        p = votes / n
        total_p = p.sum()
        best, best_k = -1.0, int(np.ceil(0.5 * n))
        for k in range(1, n + 1):
            m = votes >= k
            inside = p[m].sum()
            value = inside / (m.sum() + total_p - inside) if m.any() else 0.0
            if value > best:
                best, best_k = value, k
        out = votes >= best_k
    elif method == "sdf":
        acc = np.zeros(stack.shape[1:], np.float64)
        for m in stack:
            inside = ndimage.distance_transform_edt(m)
            outside = ndimage.distance_transform_edt(1 - m)
            acc += inside - outside
        out = acc > 0
    elif method == "staple":
        d = stack.reshape(n, -1).astype(np.float64)
        w = d.mean(0)                                   # start from the vote fraction
        prior = max(min(w.mean(), 0.99), 0.01)
        for _ in range(20):
            sens = (d @ w + 1e-6) / (w.sum() + 2e-6)
            spec = ((1 - d) @ (1 - w) + 1e-6) / ((1 - w).sum() + 2e-6)
            log_a = np.log(prior) + (d * np.log(sens[:, None]) + (1 - d) * np.log(1 - sens[:, None])).sum(0)
            log_b = np.log(1 - prior) + ((1 - d) * np.log(spec[:, None]) + d * np.log(1 - spec[:, None])).sum(0)
            w = 1.0 / (1.0 + np.exp(np.clip(log_b - log_a, -50, 50)))
        out = (w >= 0.5).reshape(stack.shape[1:])
    else:
        raise ValueError(method)
    full = np.zeros((E.SIZE, E.SIZE), np.uint8)
    full[y0:y0 + out.shape[0], x0:x0 + out.shape[1]] = out
    return as_text(mu.encode(np.asfortranarray(full)))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--shift", type=Path, required=True)
    parser.add_argument("--ingredients", type=Path, required=True)
    parser.add_argument("--swinb", type=Path, nargs="+", required=True)
    parser.add_argument("--yolo", action="append", default=[])
    parser.add_argument("--methods", nargs="+", default=["majority", "eioeu", "sdf", "staple"])
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
    pools = [load_pool(Path(p)) for _, p in (s.split("=", 1) for s in sorted(args.yolo))]
    exp139 = {n: [as_bytes(r) for _, r in E.paint([(c["score"], E.Crop(c["rle"])) for c in cache[n]["full"]
                                                  if c["score"] >= 0.85], 100)] for n in names}
    kept_idx, groups = {}, {}
    for n in names:
        s = scores(prep[n], 4.0, 0.3)
        s_b = support(cache[n]["full"], [c for c in swinb[n]["full"] if c["p"] >= 0.5])
        kept_idx[n] = [(i, float(s[i])) for i in range(len(s)) if s[i] >= 0.68 or (0.60 <= s[i] < 0.68 and s_b[i] >= 0.95)]
        g = groups_for(cache[n], [p.get(n, []) for p in pools])
        groups[n] = {i: g[i] for i, _ in kept_idx[n]}
    rows = []
    for method in args.methods:
        by = {}
        for n in names:
            kept = [(s, E.Crop(fuse(groups[n][i], method))) for i, s in kept_idx[n]]
            final = [as_bytes(r) for r in paint_fraction(kept, ORDERS["score_desc"], 0.7)]
            final = final + guard(exp139[n], final)
            by[n] = [match(final, t) for t in passes[n]]
        whole = total([r for n in names for r in by[n]])
        halves = [total([r for n in f for r in by[n]])["pq"] for f in folds]
        rows.append(dict(method=method, pq=whole["pq"], sq=whole["sq"], tp=whole["tp"], fp=whole["fp"], fn=whole["fn"],
                         halves=halves))
        print(json.dumps(dict(method=method, pq=round(whole["pq"], 5), sq=round(whole["sq"], 4), tp=whole["tp"],
                              fp=whole["fp"], fn=whole["fn"], halves=[round(h, 5) for h in halves])), flush=True)
    args.output.write_text(json.dumps(rows, indent=1))


if __name__ == "__main__":
    main()
