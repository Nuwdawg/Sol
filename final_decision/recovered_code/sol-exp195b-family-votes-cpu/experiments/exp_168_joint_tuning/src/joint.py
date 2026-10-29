"""exp168: tune the final rule's constants jointly, with the choice made out of sample.

The exp162b rule has constants chosen one at a time on the full validation set: YOLO vote weight 2,
geometric exponent 1/2, dihedral views only, YOLO masks eroded 1 px in the pixel vote, threshold 0.74.
Joint tuning on the same photographs would fit noise, so the choice is cross-validated:
photographs are split into two folds (seeded); each fold picks the best (configuration, threshold) on
itself and is scored on the other fold; the two held-out halves are pooled. The fixed exp162b rule gets
the same treatment (its threshold is also chosen within each fold), so the comparison is fair.

Grid: YOLO vote weight w in {2, 3, 4}; exponent a in {0.5, 0.4, 0.3}, score = s^a x mean^(1-a);
views d4 or d4+shift (shift views vote on both score and mask). Masks: majority vote over the
candidate, its view partners and its eroded YOLO partners (exp166).

    python -m experiments.exp_168_joint_tuning.src.joint --cache ... --shift ... --yolo yolov8l=... --yolo yolo11m=...
        --annotations ... --output ...
"""
from __future__ import annotations

import argparse
import itertools
import json
import math
from pathlib import Path

import numpy as np

from experiments.exp_033_inference.src.cached_tta import vote_rle
from experiments.exp_033_inference.src.sweep import records_for
from experiments.exp_149_evidence_fusion.src import evidence as E
from experiments.exp_157_shift_views.src.views import as_text, merged_cache, partners
from experiments.exp_159_yolo_votes.src.votes import load_pool, yolo_support
from experiments.exp_166_yolo_mask_votes.src.mask_votes import yolo_partners

CONFS = tuple(np.round(np.arange(0.55, 0.91, 0.01), 2))
WEIGHTS, EXPONENTS = (2.0, 3.0, 4.0), (0.5, 0.4, 0.3)


def prepared(entry: dict, pools: list[list[dict]], sources: tuple[str, ...]):
    """Vote-mask crops and the score ingredients of one photograph, for one view set."""
    groups, total, views = partners(entry, sources)
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
    return dict(crops=crops, s=s, total=total, views=views, yolo=yolo)


def scores(p: dict, weight: float, exponent: float) -> np.ndarray:
    mean = (p["total"] + weight * sum(p["yolo"])) / (p["views"] + weight * len(p["yolo"]))
    return np.power(p["s"], exponent) * np.power(np.maximum(mean, 1e-12), 1.0 - exponent)


def fold_sums(per_record: dict, photos: set) -> dict:
    """{threshold: [iou_sum, tp, fp, fn]} over the records of `photos`."""
    out = {}
    for t, rows in per_record.items():
        acc = np.zeros(4)
        for row in rows.values():
            if row["photograph"] in photos:
                acc += (row["iou_sum"], row["tp"], row["fp"], row["fn"])
        out[t] = acc
    return out


def pq(acc) -> float:
    iou, tp, fp, fn = acc
    return float(iou / tp * tp / (tp + 0.5 * fp + 0.5 * fn)) if tp else 0.0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--shift", type=Path, required=True)
    parser.add_argument("--yolo", action="append", default=[])
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    cache = merged_cache(args.cache, args.shift)
    records, _ = records_for(args.annotations)
    pools = [load_pool(Path(p)) for _, p in (s.split("=", 1) for s in sorted(args.yolo))]
    names = sorted(records)
    rng = np.random.default_rng(2026)
    order = rng.permutation(len(names))
    folds = [set(names[i] for i in order[:len(names) // 2]), set(names[i] for i in order[len(names) // 2:])]

    results = {}
    for sources in (("d4",), ("d4", "shift")):
        prep = {n: prepared(cache[n], [p.get(n, []) for p in pools], sources) for n in names}
        for weight, exponent in itertools.product(WEIGHTS, EXPONENTS):
            key = f"{'+'.join(sources)}_w{weight:g}_a{exponent:g}"
            pool = {n: list(zip(map(float, scores(prep[n], weight, exponent)), prep[n]["crops"])) for n in names}
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

    fixed_key = "d4_w2_a0.5"
    fixed_cv, fixed_picks = cross_validated([fixed_key])
    joint_cv, joint_picks = cross_validated(list(results))
    best_key = max(results, key=lambda k: results[k]["best"]["pq"])
    report = dict(fixed=dict(key=fixed_key, in_sample=results[fixed_key]["best"], cross_validated=fixed_cv,
                             picks=fixed_picks),
                  joint=dict(cross_validated=joint_cv, picks=joint_picks, in_sample_best=best_key,
                             in_sample=results[best_key]["best"],
                             vs_fixed=E.bootstrap_difference(results[fixed_key]["per_best"],
                                                             results[best_key]["per_best"], records)),
                  configs={k: v["best"] for k, v in results.items()})
    print("DECISION", json.dumps({k: report[k] for k in ("fixed", "joint")}, default=float), flush=True)
    (args.output / "exp168_report.json").write_text(json.dumps(report, indent=1, default=float))


if __name__ == "__main__":
    main()
