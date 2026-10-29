"""exp199: Swin-B epoch 32 as a second Mask2Former voter in exp169's rule, decided out of sample.

exp169's rule votes with one Mask2Former (exp139 e40, Swin-S) and two YOLO models. exp185/exp195a
added three weaker Mask2Formers (validation 0.452-0.464) as voters and PQ fell as their weight rose.
exp190b's Swin-B e32 (0.4698, other backbone and seed) is the first voter as strong as exp139
(0.4705); exp198 cached it on the 106 validation photographs (full + dihedral views, five shards).

What is fixed and what varies (pre-registered in final_decision/HANDOFF.md, section 1):
- exp169's constants stay: views d4+shift, YOLO weight 4, exponent 0.3, area 100, exp168's
  threshold grid and its two photograph folds (seed 2026);
- only the Swin-B voter varies: score weight m in {0, 1, 2, 4}, with and without its full-view
  partner masks in the pixel vote (exp195's `--with-masks`). exp195 searched 72 configurations,
  which cost its cross-validation about 0.004 for the search alone; this grid has 8.

Gates, in order, all fatal:
1. PARITY: m = 0 without masks reproduces exp168's in-sample row for exp169's key exactly
   (PQ 0.48894859995182627, TP/FP/FN 892/304/433);
2. SINGLE: Swin-B alone (full view, its own threshold, area 100) scores 0.4698 +- 0.005, the
   training-time evaluation of exp190b's e32 (this cache is CPU float32);
3. DECISION: adopt only if the cross-validated PQ over the 8 configurations beats exp169's own
   cross-validated PQ by >= +0.002 AND the in-sample best configuration has m > 0.

    python -m experiments.exp_199_swinb_votes.src.votes --cache VAL147 --shift SHIFTVAL
        --yolo yolov8l=V8LVAL --yolo yolo11m=Y11VAL --swinb SHARD0 ... SHARD4
        --exp168-report exp168_report.json --annotations ANN --output OUT
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from experiments.exp_033_inference.src.sweep import records_for
from experiments.exp_149_evidence_fusion.src import evidence as E
from experiments.exp_157_shift_views.src.views import merged_cache
from experiments.exp_159_yolo_votes.src.votes import load_pool
from experiments.exp_168_joint_tuning.src.joint import CONFS, fold_sums, pq
from experiments.exp_185_m2f_votes.src.votes import scores
from experiments.exp_195_family_votes.src.votes import prepared

KEY = "d4+shift_w4_a0.3"                  # exp169's configuration
SOURCES, YOLO_WEIGHT, EXPONENT = ("d4", "shift"), 4.0, 0.3
NEW_WEIGHTS = (0.0, 1.0, 2.0, 4.0)
BAR = 0.002
SINGLE_EXPECTED, SINGLE_TOLERANCE = 0.4698, 0.005


def merge_shards(paths: list[Path], names: list[str]) -> tuple[dict, dict]:
    """Five exp198 shards -> one cache in exp147's layout; every validation photograph exactly once."""
    merged, models = {}, set()
    for path in paths:
        raw = json.loads(Path(path).read_text())
        models.add(json.dumps(raw.get("model", {}), sort_keys=True))
        for name, entry in raw["photographs"].items():
            if name in merged:
                raise SystemExit(f"{name} appears in two shards")
            if entry.get("split") != "val" or "full" not in entry or "d4" not in entry:
                raise SystemExit(f"{name}: not a validation entry with full and d4 views")
            merged[name] = entry
    if len(models) != 1:
        raise SystemExit(f"shards come from different checkpoints: {models}")
    if sorted(merged) != sorted(names):
        missing = sorted(set(names) - set(merged))[:5]
        extra = sorted(set(merged) - set(names))[:5]
        raise SystemExit(f"shards do not cover the validation split: missing {missing}, extra {extra}")
    return merged, json.loads(models.pop())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--shift", type=Path, required=True)
    parser.add_argument("--yolo", action="append", default=[])
    parser.add_argument("--swinb", type=Path, nargs="+", required=True, help="the exp198 validation shards")
    parser.add_argument("--exp168-report", type=Path, required=True)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    expected = json.loads(args.exp168_report.read_text())["configs"][KEY]
    cache = merged_cache(args.cache, args.shift)
    records, _ = records_for(args.annotations)
    names = sorted(records)
    pools = [load_pool(Path(p)) for _, p in (s.split("=", 1) for s in sorted(args.yolo))]
    swinb, model = merge_shards(args.swinb, names)
    print("SWINB", json.dumps(dict(model=model, photographs=len(swinb),
                                   full=sum(len(e["full"]) for e in swinb.values()),
                                   d4=sum(len(e["d4"]) for e in swinb.values()))), flush=True)

    rng = np.random.default_rng(2026)                      # exp168's folds, exactly
    order = rng.permutation(len(names))
    folds = [set(names[i] for i in order[:len(names) // 2]), set(names[i] for i in order[len(names) // 2:])]

    # Gate 2: Swin-B alone, as exp195 scores a new model (full view, 0.05 grid, area 100).
    single_confs = tuple(round(0.05 * i, 2) for i in range(1, 20))
    rows, _ = E.score_rule({n: [(c["score"], E.Crop(c["rle"])) for c in swinb[n]["full"]] for n in names},
                           records, single_confs, min_area=100)
    single = max(rows, key=lambda r: r["pq"])
    single_ok = abs(single["pq"] - SINGLE_EXPECTED) <= SINGLE_TOLERANCE
    print("SINGLE", json.dumps(dict(best=single, expected=SINGLE_EXPECTED, passed=single_ok)), flush=True)

    results = {}
    for with_masks in (False, True):
        prep = {n: prepared(cache[n], [p.get(n, []) for p in pools], [swinb[n]], SOURCES, with_masks)
                for n in names}
        for m in NEW_WEIGHTS:
            key = f"{KEY}_m{m:g}" + ("_masks" if with_masks else "")
            pool = {n: list(zip(map(float, scores(prep[n], YOLO_WEIGHT, m, EXPONENT)), prep[n]["crops"]))
                    for n in names}
            rows, per = E.score_rule(pool, records, CONFS, min_area=100)
            best = max(rows, key=lambda r: r["pq"])
            results[key] = dict(best=best, folds=[fold_sums(per, f) for f in folds], m=m, masks=with_masks)
            print("CONFIG", key, json.dumps(best), flush=True)
            if key == f"{KEY}_m0":
                parity = (abs(best["pq"] - expected["pq"]) < 1e-9
                          and all(best[k] == expected[k] for k in ("tp", "fp", "fn")))
                print("PARITY", json.dumps(dict(row=best, expected=expected, passed=parity)), flush=True)
                if not parity:
                    raise SystemExit("m = 0 does not reproduce exp169's in-sample row; nothing is decided")
    if not single_ok:
        raise SystemExit("Swin-B alone does not reproduce its recorded validation PQ; nothing is decided")

    def cross_validated(keys: list[str]) -> tuple[float, list]:
        pooled, picks = np.zeros(4), []
        for held, chooser in ((0, 1), (1, 0)):
            key, t = max(((k, t) for k in keys for t in results[k]["folds"][chooser]),
                         key=lambda kt: pq(results[kt[0]]["folds"][chooser][kt[1]]))
            pooled += results[key]["folds"][held][t]
            picks.append(dict(key=key, conf=t, chosen_on=chooser, scored_on=held))
        return pq(pooled), picks

    baseline = f"{KEY}_m0"
    base_cv, base_picks = cross_validated([baseline])
    grid_cv, grid_picks = cross_validated(list(results))
    in_sample_best = max(results, key=lambda k: results[k]["best"]["pq"])
    gain = grid_cv - base_cv
    adopted = gain >= BAR and results[in_sample_best]["m"] > 0
    decision = dict(exp169_cv=base_cv, exp169_picks=base_picks, grid_cv=grid_cv, grid_picks=grid_picks,
                    cross_validated_gain=gain, bar=BAR, in_sample_best=in_sample_best,
                    in_sample_best_row=results[in_sample_best]["best"], adopted=adopted)
    print("DECISION", json.dumps(decision, default=float), flush=True)
    report = dict(swinb_model=model, single=single, decision=decision,
                  configs={k: v["best"] for k, v in results.items()})
    (args.output / "exp199_report.json").write_text(json.dumps(report, indent=1, default=float))


if __name__ == "__main__":
    main()
