"""exp195: a new architecture family (RF-DETR-Seg, exp194) as votes in the exp168 rule, chosen out of
sample, with exp169's configuration reproduced exactly as the baseline.

exp185 added the new models' masks to the pixel vote in every configuration, so its weight-0 rows
were not exp169 (0.48745 against 0.48895). Here the new models' masks join the pixel vote only with
`--with-masks`, and the run asserts that its weight-0 row of exp169's key reproduces the in-sample
row exp168 recorded for that key (`--exp168-report`: PQ to 1e-9 and TP/FP/FN exactly; 0.48895 @0.68,
892/304/433), so every comparison is with the rule that was actually submitted.

Grid: exp168's (views d4 / d4+shift, YOLO weight 2/3/4, exponent 0.5/0.4/0.3) x new-family weight
0/1/2/4. Choice by exp168's two-fold photograph cross-validation; decision against exp169's key
under the same treatment, bar +0.002.

    python -m experiments.exp_195_family_votes.src.votes --cache ... --shift ... --yolo yolov8l=... --yolo yolo11m=...
        --new rfdetr=<exp194 val cache> --annotations ... --output ...
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
from experiments.exp_168_joint_tuning.src.joint import CONFS, EXPONENTS, WEIGHTS, fold_sums, pq
from experiments.exp_185_m2f_votes.src.votes import model_support, scores

NEW_WEIGHTS = (0.0, 1.0, 2.0, 4.0)
ADOPTED = "d4+shift_w4_a0.3_m0"
BAR = 0.002


def prepared(entry: dict, pools: list[list[dict]], others: list[dict], sources: tuple[str, ...],
             with_masks: bool) -> dict:
    groups, total, views = partners(entry, sources)
    yolo = [yolo_support(entry, p) for p in pools]
    for pool_entries in pools:
        for i, extra in enumerate(yolo_partners(entry, pool_entries, 1)):
            if extra is not None:
                groups[i].append(extra)
    support = []
    for other in others:
        values, masks = model_support(entry, other)
        support.append(values)
        if with_masks:
            for i, mask in enumerate(masks):
                if mask is not None:
                    groups[i].append(mask)
    crops = []
    for i, group in enumerate(groups):
        mask = entry["full"][i]["rle"] if len(group) == 1 else as_text(vote_rle(group, max(1, math.ceil(0.5 * len(group)))))
        crops.append(E.Crop(mask))
    s = np.array([c["score"] for c in entry["full"]], np.float64)
    return dict(crops=crops, s=s, total=total, views=views, yolo=yolo, m2f=support)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--shift", type=Path, required=True)
    parser.add_argument("--yolo", action="append", default=[])
    parser.add_argument("--new", action="append", default=[], help="tag=exp147-format validation cache")
    parser.add_argument("--with-masks", action="store_true")
    parser.add_argument("--exp168-report", type=Path, required=True)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    expected = json.loads(args.exp168_report.read_text())["configs"][ADOPTED[:-3]]
    cache = merged_cache(args.cache, args.shift)
    records, _ = records_for(args.annotations)
    pools = [load_pool(Path(p)) for _, p in (s.split("=", 1) for s in sorted(args.yolo))]
    tags = [t for t, _ in (s.split("=", 1) for s in sorted(args.new))]
    others = [E.load_cache(Path(p)) for _, p in (s.split("=", 1) for s in sorted(args.new))]
    names = sorted(records)
    rng = np.random.default_rng(2026)                      # exp168's folds
    order = rng.permutation(len(names))
    folds = [set(names[i] for i in order[:len(names) // 2]), set(names[i] for i in order[len(names) // 2:])]

    single_confs = tuple(round(0.05 * i, 2) for i in range(1, 20))
    base_rows, base_per = E.score_rule({n: [(c["score"], E.Crop(c["rle"])) for c in cache[n]["full"]] for n in names},
                                       records, single_confs, min_area=100)
    base_best = max(base_rows, key=lambda r: r["pq"])
    single = {"exp139": dict(best=base_best)}
    for tag, other in zip(tags, others):
        rows, per = E.score_rule({n: [(c["score"], E.Crop(c["rle"])) for c in other.get(n, {}).get("full", [])]
                                  for n in names}, records, single_confs, min_area=100)
        best = max(rows, key=lambda r: r["pq"])
        single[tag] = dict(best=best, vs_exp139=E.bootstrap_difference(base_per[base_best["conf"]], per[best["conf"]],
                                                                     records))
        print("SINGLE", tag, json.dumps(single[tag], default=float), flush=True)

    results = {}
    for sources in (("d4",), ("d4", "shift")):
        prep = {n: prepared(cache[n], [p.get(n, []) for p in pools], [o.get(n, {}) for o in others], sources,
                            args.with_masks) for n in names}
        weights = NEW_WEIGHTS if others else (0.0,)
        for weight, new_weight, exponent in itertools.product(WEIGHTS, weights, EXPONENTS):
            key = f"{'+'.join(sources)}_w{weight:g}_a{exponent:g}_m{new_weight:g}"
            pool = {n: list(zip(map(float, scores(prep[n], weight, new_weight, exponent)), prep[n]["crops"]))
                    for n in names}
            rows, per = E.score_rule(pool, records, CONFS, min_area=100)
            best = max(rows, key=lambda r: r["pq"])
            results[key] = dict(best=best, folds=[fold_sums(per, f) for f in folds], per_best=per[best["conf"]])
            print("CONFIG", key, json.dumps(best), flush=True)
            if key == ADOPTED and not args.with_masks:
                parity = abs(best["pq"] - expected["pq"]) < 1e-9 and all(best[k] == expected[k] for k in ("tp", "fp", "fn"))
                print("PARITY", json.dumps(dict(key=key, row=best, expected=expected, passed=parity)), flush=True)
                if not parity:
                    raise SystemExit("the weight-0 row does not reproduce exp169's in-sample PQ; nothing is decided")

    def cross_validated(keys: list[str]) -> tuple[float, list]:
        pooled, picks = np.zeros(4), []
        for held, chooser in ((0, 1), (1, 0)):
            key, t = max(((k, t) for k in keys for t in results[k]["folds"][chooser]),
                         key=lambda kt: pq(results[kt[0]]["folds"][chooser][kt[1]]))
            pooled += results[key]["folds"][held][t]
            picks.append(dict(key=key, conf=t, chosen_on=chooser, scored_on=held))
        return pq(pooled), picks

    adopted_cv, adopted_picks = cross_validated([ADOPTED])
    joint_cv, joint_picks = cross_validated(list(results))
    best_key = max(results, key=lambda k: results[k]["best"]["pq"])
    gain = joint_cv - adopted_cv
    report = dict(models=tags, with_masks=args.with_masks, single=single,
                  adopted=dict(key=ADOPTED, in_sample=results[ADOPTED]["best"], cross_validated=adopted_cv,
                               picks=adopted_picks),
                  joint=dict(cross_validated=joint_cv, picks=joint_picks, in_sample_best=best_key,
                             in_sample=results[best_key]["best"],
                             vs_adopted=E.bootstrap_difference(results[ADOPTED]["per_best"],
                                                               results[best_key]["per_best"], records)),
                  decision=dict(cross_validated_gain=gain, bar=BAR, adopted=gain >= BAR),
                  configs={k: v["best"] for k, v in results.items()})
    print("DECISION", json.dumps({k: report[k] for k in ("adopted", "joint", "decision")}, default=float), flush=True)
    (args.output / "exp195_report.json").write_text(json.dumps(report, indent=1, default=float))


if __name__ == "__main__":
    main()
