"""exp201b: does the deployed rule's gain carry over to another Swin-S model? (validation, per record)

exp201 will put a retrained Swin-S in exp139's place without a held-out photograph to check the
rule's constants on. exp190b's Swin-S epoch 50 (seed 7, 50-epoch schedule, training split only) is
the same recipe as exp201's Swin-S minus the validation photographs, so it is the closest model on
which the constants can be checked. Cached on CPU in exp139's 11 views (full, 7 dihedral, 3 shift).

For each model (exp139 e40 and exp190b Swin-S e50), per record:
  single:   the full view alone, painted, area 100, over a threshold grid (exp139's rule: 0.85);
  exp169:   exp168's ingredients over the model's 11 views + the two YOLO pools, score
            s^0.3 x mean^0.7 (YOLO weight 4), painted, area 100, over a threshold grid (deployed: 0.68);
  combo:    the deployed file's rule (rescue band [t - 0.08, t) with Swin-B support >= 0.95, remnant
            filter 0.7, guard on the model's own single-view file) over the same grid.
And one two-model rule:
  pair22:   exp139's candidates voted on by both models' 11 views (22 view scores, partner masks of
            both in the pixel vote) + YOLO, then the combo rule; pair22_w8 doubles the YOLO weight so
            the two YOLO votes keep their exp169 share of the score (8 of 19 slots).
Every threshold choice is also made on one of exp168's two photograph folds and scored on the other.
Parity: exp139 single 0.47052 at 0.85, exp169 0.48895 at 0.68, combo 0.49095 at 0.68.

    python -m experiments.exp_201_alldata.src.transfer --annotations ... --cache exp147_val_cache.json
        --shift exp157_shift_val_cache.json --other SUB0_val_cache.json SUB1_val_cache.json
        --yolo yolov8l=... --yolo yolo11m=... --swinb SHARD0 ... SHARD4 --output transfer.json
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
from pycocotools import mask as mu

from experiments.exp_033_inference.src.cached_tta import vote_rle
from experiments.exp_033_inference.src.sweep import records_for
from experiments.exp_149_evidence_fusion.src import evidence as E
from experiments.exp_157_shift_views.src.views import VIEWS, as_text, merged_cache
from experiments.exp_159_yolo_votes.src.votes import load_pool
from experiments.exp_168_joint_tuning.src.joint import prepared, scores
from experiments.exp_199_swinb_votes.src.guard import guard
from experiments.exp_199_swinb_votes.src.votes import merge_shards
from experiments.exp_200_test_truth.src.hypotheses import as_bytes, match
from experiments.exp_200_test_truth.src.paint_order import ORDERS
from experiments.exp_200_test_truth.src.remnants import paint_fraction
from experiments.exp_200_test_truth.src.swinb_gate import support

SINGLE = tuple(round(t, 2) for t in np.arange(0.70, 0.951, 0.01))
RULE = tuple(round(t, 2) for t in np.arange(0.58, 0.821, 0.01))
BAND, TAU, FRACTION = 0.08, 0.95, 0.7


def pq_of(sums) -> float:
    tp, fp, fn, iou = sums
    return (iou / tp) * (tp / (tp + 0.5 * fp + 0.5 * fn)) if tp else 0.0


def load_other(paths: list[Path], names: list[str]) -> dict:
    merged = {}
    for path in paths:
        for name, entry in json.loads(Path(path).read_text())["photographs"].items():
            if name in merged:
                raise SystemExit(f"{name} appears twice")
            merged[name] = entry
    if sorted(merged) != sorted(names):
        raise SystemExit(f"the other model covers {len(merged)} of {len(names)} validation photographs")
    for name, entry in merged.items():
        for mode in ("full", "d4", "shift"):
            if mode not in entry:
                raise SystemExit(f"{name}: no {mode} view")
    return merged


def cross(entry: dict, other: dict) -> tuple[np.ndarray, list[list[dict]]]:
    """For each of entry's full candidates: summed best-partner scores over the other model's 11
    views (0 without a partner at IoU > 0.5) and those partners' masks."""
    rles = [E.as_rle(c["rle"]) for c in entry["full"]]
    total = np.zeros(len(rles))
    masks = [[] for _ in rles]
    if not rles:
        return total, masks
    views = [("full", None)] + [(s, v) for s in ("d4", "shift") for v in VIEWS[s]]
    for source, view in views:
        members = [c for c in other.get(source, []) if view is None or c["view"] == view]
        if not members:
            continue
        mr = [E.as_rle(c["rle"]) for c in members]
        ious = np.asarray(mu.iou(rles, mr, [0] * len(mr))).reshape(len(rles), len(mr))
        for i in range(len(rles)):
            j = int(ious[i].argmax())
            if ious[i, j] > 0.5:
                total[i] += members[j]["score"]
                masks[i].append(mr[j])
    return total, masks


def pair_ingredients(entry: dict, other: dict, pools: list[list[dict]]) -> dict:
    """exp168's ingredients with the other model's 11 views added as voters (scores and masks)."""
    from experiments.exp_157_shift_views.src.views import partners
    from experiments.exp_159_yolo_votes.src.votes import yolo_support
    from experiments.exp_166_yolo_mask_votes.src.mask_votes import yolo_partners
    groups, total, views = partners(entry, ("d4", "shift"))
    extra_total, extra_masks = cross(entry, other)
    for i, ms in enumerate(extra_masks):
        groups[i].extend(ms)
    yolo = [yolo_support(entry, p) for p in pools]
    for pool_entries in pools:
        for i, extra in enumerate(yolo_partners(entry, pool_entries, 1)):
            if extra is not None:
                groups[i].append(extra)
    crops = []
    for i, group in enumerate(groups):
        mask = entry["full"][i]["rle"] if len(group) == 1 else as_text(vote_rle(group, max(1, math.ceil(0.5 * len(group)))))
        crops.append(E.Crop(mask))
    s = np.array([c["score"] for c in entry["full"]], np.float64)
    return dict(crops=crops, s=s, total=total + extra_total, views=views + 11, yolo=yolo)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--shift", type=Path, required=True)
    parser.add_argument("--other", type=Path, nargs="+", required=True)
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
    models = {"exp139": merged_cache(args.cache, args.shift), "s7e050": load_other(args.other, names)}
    pools = {t: load_pool(Path(p)) for t, p in (s.split("=", 1) for s in args.yolo)}
    tags = sorted(pools)
    if tags != ["yolo11m", "yolov8l"]:
        raise SystemExit(f"expected the yolo11m and yolov8l pools, got {tags}")
    swinb, _ = merge_shards(args.swinb, names)
    swinb_full = {n: [c for c in swinb[n]["full"] if c["p"] >= 0.5] for n in names}

    def sums(final, n):
        rows = [match(final, t) for t in passes[n]]
        return tuple(sum(r[k] for r in rows) for k in ("tp", "fp", "fn", "iou_sum"))

    def curve(per_photo: dict, grid) -> dict:
        """per_photo[n][t] -> sums; returns in-sample PQ per t, the cross-fitted PQ and the choices."""
        whole = {t: pq_of(tuple(sum(per_photo[n][t][k] for n in names) for k in range(4))) for t in grid}
        chosen, held = [], {}
        for k, f in enumerate(folds):
            best = max(grid, key=lambda t: pq_of(tuple(sum(per_photo[n][t][j] for n in f) for j in range(4))))
            chosen.append(float(best))
            for n in folds[1 - k]:
                held[n] = per_photo[n][best]
        cv = pq_of(tuple(sum(held[n][j] for n in names) for j in range(4)))
        return dict(pq={str(t): v for t, v in whole.items()}, cv=cv, fold_choices=chosen,
                    best=max(grid, key=lambda t: whole[t]))

    out = {}
    singles = {}
    for model, cache in models.items():
        per = {}
        for n in names:
            full = [(c["score"], E.Crop(c["rle"])) for c in cache[n]["full"]]
            per[n] = {t: sums([as_bytes(r) for _, r in E.paint([(s, c) for s, c in full if s >= t], 100)], n) for t in SINGLE}
            singles.setdefault(model, {})[n] = [as_bytes(r) for _, r in E.paint([(s, c) for s, c in full if s >= 0.85], 100)]
        out[f"{model}_single"] = curve(per, SINGLE)
        print("SINGLE", model, round(out[f"{model}_single"]["pq"]["0.85"], 5), "best",
              out[f"{model}_single"]["best"], round(out[f"{model}_single"]["cv"], 5), flush=True)

    def rule_rows(ingredients: dict, base: dict, s_b: dict, tag: str, weight: float = 4.0) -> None:
        per_rule, per_combo = {}, {}
        for n in names:
            s = np.asarray(scores(ingredients[n], weight, 0.3), np.float64)
            crops = ingredients[n]["crops"]
            per_rule[n], per_combo[n] = {}, {}
            memo = {}
            for t in RULE:
                kept = [(float(v), c) for v, c in zip(s, crops) if v >= t]
                per_rule[n][t] = sums([as_bytes(r) for _, r in E.paint(kept, 100)], n)
                idx = tuple(i for i in range(len(s)) if s[i] >= t or (t - BAND <= s[i] < t and s_b[n][i] >= TAU))
                if idx not in memo:
                    entries = [(float(s[i]), crops[i]) for i in idx]
                    final = [as_bytes(r) for r in paint_fraction(entries, ORDERS["score_desc"], FRACTION)]
                    memo[idx] = sums(final + guard(base[n], final), n)
                per_combo[n][t] = memo[idx]
        for kind, per in (("exp169", per_rule), ("combo", per_combo)):
            row = curve(per, RULE)
            out[f"{tag}_{kind}"] = row
            print("RULE", tag, kind, "at 0.68", round(row["pq"]["0.68"], 5), "best", row["best"],
                  round(row["pq"][str(row["best"])], 5), "cv", round(row["cv"], 5), row["fold_choices"], flush=True)

    for model, cache in models.items():
        ingredients = {n: prepared(cache[n], [pools[t].get(n, []) for t in tags], ("d4", "shift")) for n in names}
        s_b = {n: support(cache[n]["full"], swinb_full[n]) for n in names}
        rule_rows(ingredients, singles[model], s_b, model)
    ingredients = {n: pair_ingredients(models["exp139"][n], models["s7e050"][n], [pools[t].get(n, []) for t in tags])
                   for n in names}
    s_b = {n: support(models["exp139"][n]["full"], swinb_full[n]) for n in names}
    rule_rows(ingredients, singles["exp139"], s_b, "pair22")
    # 22 view slots dilute the two YOLO votes (8 of 19 slots in exp169, 8 of 30 here); weight 8 keeps their share
    rule_rows(ingredients, singles["exp139"], s_b, "pair22_w8", weight=8.0)
    args.output.write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
