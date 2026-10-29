"""exp185: separately trained Mask2Former models as votes in the exp168 rule, chosen out of sample.

The exp168/exp169 rule scores each exp139 candidate by s^a x mean^(1-a), where the mean runs over
exp139's own views (dihedral, optionally shifted) and the two YOLO models' support (weight w_y).
exp159 found separately trained models are the votes that help; exp139 was the only Mask2Former.
exp173 and later runs add Mask2Former models trained with other seeds and recipes. Each such
model m contributes a support in [0, 1] per exp139 candidate: the mean, over its eight dihedral
views (exp147 cache: full + d4), of the score of its best partner with IoU > 0.5 (0 without one),
at weight w_m. Its full-view partner mask also joins the candidate's pixel vote.

The grid adds w_m in {0, 1, 2, 4} (0 is the exp168 rule) to exp168's grid; the choice is made by
exp168's two-fold photograph cross-validation, and the cross-validated PQ is compared with that of
the adopted exp169 configuration under the same treatment.

    python -m experiments.exp_185_m2f_votes.src.votes --cache exp147_val_cache.json --shift exp157_val.json
        --yolo yolov8l=... --yolo yolo11m=... --m2f control=control_e028_val_cache.json ...
        --annotations ... --output ...
"""
from __future__ import annotations

import argparse
import itertools
import json
import math
from pathlib import Path

import numpy as np
from pycocotools import mask as mu

from experiments.exp_033_inference.src.cached_tta import vote_rle
from experiments.exp_033_inference.src.sweep import records_for
from experiments.exp_149_evidence_fusion.src import evidence as E
from experiments.exp_157_shift_views.src.views import as_text, merged_cache, partners
from experiments.exp_159_yolo_votes.src.votes import load_pool, yolo_support
from experiments.exp_166_yolo_mask_votes.src.mask_votes import yolo_partners
from experiments.exp_168_joint_tuning.src.joint import CONFS, EXPONENTS, WEIGHTS, fold_sums, pq

M2F_WEIGHTS = (0.0, 1.0, 2.0, 4.0)
ADOPTED = "d4+shift_w4_a0.3_m0"          # exp169's configuration in this grid's naming


def model_support(entry: dict, other: dict) -> tuple[np.ndarray, list]:
    """Mean best-partner score over the other model's eight views, and its full-view partner masks."""
    rles = [E.as_rle(c["rle"]) for c in entry["full"]]
    support = np.zeros(len(rles))
    masks: list = [None] * len(rles)
    if not rles:
        return support, masks
    views = [("full", None)] + [("d4", v) for v in range(1, 8)]
    for source, view in views:
        members = [c for c in other.get(source, []) if view is None or c.get("view") == view]
        if not members:
            continue
        member_rles = [E.as_rle(c["rle"]) for c in members]
        ious = np.asarray(mu.iou(rles, member_rles, [0] * len(member_rles))).reshape(len(rles), len(members))
        best = ious.argmax(axis=1)
        hit = ious.max(axis=1) > 0.5
        scores = np.array([c["score"] for c in members])
        support += np.where(hit, scores[best], 0.0)
        if source == "full":
            masks = [member_rles[best[i]] if hit[i] else None for i in range(len(rles))]
    return support / len(views), masks


def prepared(entry: dict, pools: list[list[dict]], others: list[dict], sources: tuple[str, ...]) -> dict:
    groups, total, views = partners(entry, sources)
    yolo = [yolo_support(entry, p) for p in pools]
    for pool_entries in pools:
        for i, extra in enumerate(yolo_partners(entry, pool_entries, 1)):
            if extra is not None:
                groups[i].append(extra)
    m2f = []
    for other in others:
        support, masks = model_support(entry, other)
        m2f.append(support)
        for i, mask in enumerate(masks):
            if mask is not None:
                groups[i].append(mask)
    crops = []
    for i, group in enumerate(groups):
        mask = entry["full"][i]["rle"] if len(group) == 1 else as_text(vote_rle(group, max(1, math.ceil(0.5 * len(group)))))
        crops.append(E.Crop(mask))
    s = np.array([c["score"] for c in entry["full"]], np.float64)
    return dict(crops=crops, s=s, total=total, views=views, yolo=yolo, m2f=m2f)


def scores(p: dict, weight: float, m2f_weight: float, exponent: float) -> np.ndarray:
    numerator = p["total"] + weight * sum(p["yolo"]) + m2f_weight * sum(p["m2f"])
    denominator = p["views"] + weight * len(p["yolo"]) + m2f_weight * len(p["m2f"])
    return np.power(p["s"], exponent) * np.power(np.maximum(numerator / denominator, 1e-12), 1.0 - exponent)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--shift", type=Path, required=True)
    parser.add_argument("--yolo", action="append", default=[])
    parser.add_argument("--m2f", action="append", default=[], help="tag=exp184 validation cache")
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    cache = merged_cache(args.cache, args.shift)
    records, _ = records_for(args.annotations)
    pools = [load_pool(Path(p)) for _, p in (s.split("=", 1) for s in sorted(args.yolo))]
    others = [E.load_cache(Path(p)) for _, p in (s.split("=", 1) for s in sorted(args.m2f))]
    tags = [t for t, _ in (s.split("=", 1) for s in sorted(args.m2f))]
    names = sorted(records)
    rng = np.random.default_rng(2026)                      # exp168's folds
    order = rng.permutation(len(names))
    folds = [set(names[i] for i in order[:len(names) // 2]), set(names[i] for i in order[len(names) // 2:])]

    # Each model alone, full view, its own best threshold at the deployed emission (no erosion,
    # area 100), paired against exp139 on the same photographs: the recipe comparison.
    single_confs = tuple(round(0.05 * i, 2) for i in range(1, 20))
    single = {}
    base_rows, base_per = E.score_rule({n: [(c["score"], E.Crop(c["rle"])) for c in cache[n]["full"]] for n in names},
                                       records, single_confs, min_area=100)
    base_best = max(base_rows, key=lambda r: r["pq"])
    single["exp139"] = dict(best=base_best)
    for tag, other in zip(tags, others):
        rows, per = E.score_rule({n: [(c["score"], E.Crop(c["rle"])) for c in other.get(n, {}).get("full", [])]
                                  for n in names}, records, single_confs, min_area=100)
        best = max(rows, key=lambda r: r["pq"])
        single[tag] = dict(best=best, vs_exp139=E.bootstrap_difference(base_per[base_best["conf"]], per[best["conf"]],
                                                                     records))
        print("SINGLE", tag, json.dumps(single[tag], default=float), flush=True)

    results = {}
    for sources in (("d4",), ("d4", "shift")):
        prep = {n: prepared(cache[n], [p.get(n, []) for p in pools], [o.get(n, {}) for o in others], sources)
                for n in names}
        weights = M2F_WEIGHTS if others else (0.0,)
        for weight, m2f_weight, exponent in itertools.product(WEIGHTS, weights, EXPONENTS):
            key = f"{'+'.join(sources)}_w{weight:g}_a{exponent:g}_m{m2f_weight:g}"
            pool = {n: list(zip(map(float, scores(prep[n], weight, m2f_weight, exponent)), prep[n]["crops"]))
                    for n in names}
            rows, per = E.score_rule(pool, records, CONFS, min_area=100)
            best = max(rows, key=lambda r: r["pq"])
            results[key] = dict(best=best, folds=[fold_sums(per, f) for f in folds], per_best=per[best["conf"]])
            print("CONFIG", key, json.dumps(best), flush=True)

    def cross_validated(keys: list[str]) -> tuple[float, list]:
        pooled, picks = np.zeros(4), []
        for held, chooser in ((0, 1), (1, 0)):
            key, t = max(((k, t) for k in keys for t in results[k]["folds"][chooser]),
                         key=lambda kt: pq(results[kt[0]]["folds"][chooser][kt[1]]))
            pooled += results[key]["folds"][held][t]
            picks.append(dict(key=key, conf=t, chosen_on=chooser, scored_on=held))
        return pq(pooled), picks

    without = [k for k in results if k.endswith("_m0")]
    adopted_cv, adopted_picks = cross_validated([ADOPTED])
    without_cv, without_picks = cross_validated(without)
    joint_cv, joint_picks = cross_validated(list(results))
    best_key = max(results, key=lambda k: results[k]["best"]["pq"])
    report = dict(models=tags, single=single,
                  adopted=dict(key=ADOPTED, in_sample=results[ADOPTED]["best"], cross_validated=adopted_cv,
                               picks=adopted_picks),
                  without_new_models=dict(cross_validated=without_cv, picks=without_picks),
                  joint=dict(cross_validated=joint_cv, picks=joint_picks, in_sample_best=best_key,
                             in_sample=results[best_key]["best"],
                             vs_adopted=E.bootstrap_difference(results[ADOPTED]["per_best"],
                                                               results[best_key]["per_best"], records)),
                  configs={k: v["best"] for k, v in results.items()})
    print("DECISION", json.dumps({k: report[k] for k in ("adopted", "without_new_models", "joint")}, default=float),
          flush=True)
    (args.output / "exp185_report.json").write_text(json.dumps(report, indent=1, default=float))


if __name__ == "__main__":
    main()
