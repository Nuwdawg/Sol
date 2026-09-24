"""exp121: score the verifier's actual operating points by the official metric.

exp120 screened nine features one at a time and found none reaching the 0.500
precision the ceiling analysis demands. The best point was `score` at q0.90,
precision 0.396 on 53 candidates - and `score` is what the 0.35 floor already
uses, so that row mostly re-derives the existing threshold.

Two reasons that is not yet the family's verdict:

1. **Precision is a surrogate, and the manual says score the operator.** A point
   with modest precision but favourable mask quality can still pay: at k=40 and
   p=0.70 the arithmetic gives +0.0099. Nothing in exp120 was ever rescored by
   `aggregate_pq` after global rematching, which is the only number that counts.
2. **No combination was tested.** `score` (AUC 0.641) and `limb` flipped (0.602)
   are the two features above chance, and they measure unrelated things - detector
   confidence and position on the solar disk. A pair can clear a bar neither member
   clears alone.

So this admits candidates under each rule, re-emits through the frozen path, and
reports official PQ with crossings up and down. A rule that raises PQ is a real
gain regardless of its precision; a rule that does not is dead regardless of its
AUC.

Nothing here reads truth to *choose* what to admit. The oracle row is reproduced
only as the ceiling to measure against.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from pycocotools import mask as mu

from experiments.exp_033_inference.src.sweep import encode, pq_rle, records_for
from shared.utils import aggregate_pq, paint_panoptic

KERNEL = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
BASELINE_PQ = 0.4558638687292126
BASE_CONF = 0.35
MATCH = 0.5


def prepare(entries, grow, min_area):
    """Decode once; every rule reuses these (score, mask) pairs."""
    prepared = []
    for entry in entries:
        mask = mu.decode(entry["rle"])
        if grow < 0:
            mask = cv2.erode(mask, KERNEL, iterations=-grow)
        if int(mask.sum()) < min_area:
            continue
        prepared.append((float(entry["score"]), mask))
    return prepared


def paint(selected, min_area):
    return [encode(mask) for _, mask, _ in paint_panoptic(list(selected), min_area=min_area)]


def score_all(per_record):
    rows = [pq_rle(predicted, truths) for predicted, truths in per_record]
    aggregate = aggregate_pq(rows)
    aggregate["iou_sum"] = float(sum(row["iou_sum"] for row in rows))
    return aggregate


def per_truth_best(predicted, truths):
    if not predicted or not truths:
        return np.zeros(len(truths))
    return np.asarray(mu.iou(predicted, truths, [0] * len(truths))).max(axis=0)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pool", type=Path,
                        default=Path("data/exp033_cached_tta/candidates_0.5_0.5.json"))
    parser.add_argument("--features", type=Path,
                        default=Path("data/exp120_verifier/candidates.json"))
    parser.add_argument("--output", type=Path, default=Path("data/exp121_operating_points"))
    parser.add_argument("--grow", type=int, default=-1)
    parser.add_argument("--min-area", type=int, default=300)
    args = parser.parse_args()
    cv2.setNumThreads(2)
    args.output.mkdir(parents=True, exist_ok=True)
    annotations = next(Path(".").rglob("MAGFiLO_1.0_Annotations_kaggle2026_train.json"))
    records, _ = records_for(annotations)
    pool = json.loads(args.pool.read_text())
    scored = json.loads(args.features.read_text())

    # exp120 keyed candidates by photograph and score; rebuild that index so a rule
    # expressed over features can be applied to the pool it came from.
    index: dict[tuple[str, float], dict] = {}
    for row in scored:
        index[(row["photograph"], round(row["score"], 9))] = row
    print(f"feature rows {len(scored)}, admissible {sum(r['label'] for r in scored)}")

    values = {name: np.array([r[name] for r in scored], float)
              for name in ("score", "limb", "half_width")}
    cuts = {name: {q: float(np.quantile(values[name], q)) for q in
                   (0.50, 0.70, 0.80, 0.90, 0.95)} for name in values}

    def rule_none(_row):
        return False

    def make_single(name, quantile, high=True):
        threshold = cuts[name][quantile]
        if high:
            return lambda row: row[name] >= threshold
        return lambda row: row[name] <= threshold

    def make_pair(quantile_score, quantile_limb):
        # score high AND limb low: confident proposals away from the limb, where
        # exp120's two above-chance features point in opposite directions.
        score_cut = cuts["score"][quantile_score]
        limb_cut = cuts["limb"][quantile_limb]
        return lambda row: row["score"] >= score_cut and row["limb"] <= limb_cut

    rules = [
        ("baseline (admit nothing)", rule_none),
        ("score q0.50", make_single("score", 0.50)),
        ("score q0.70", make_single("score", 0.70)),
        ("score q0.80", make_single("score", 0.80)),
        ("score q0.90", make_single("score", 0.90)),
        ("score q0.95", make_single("score", 0.95)),
        ("limb low q0.50", make_single("limb", 0.50, high=False)),
        ("half_width q0.70", make_single("half_width", 0.70)),
        ("score q0.70 AND limb low q0.50", make_pair(0.70, 0.50)),
        ("score q0.80 AND limb low q0.50", make_pair(0.80, 0.50)),
        ("score q0.80 AND limb low q0.70", make_pair(0.80, 0.70)),
        ("score q0.90 AND limb low q0.70", make_pair(0.90, 0.70)),
    ]

    prepared = {name: prepare(pool[name], args.grow, args.min_area) for name in records}
    base_emissions, base_records = {}, []
    for name, passes in records.items():
        emitted = paint([(s, m) for s, m in prepared[name] if s >= BASE_CONF], args.min_area)
        base_emissions[name] = emitted
        for _, truths in passes:
            base_records.append((emitted, truths))
    baseline = score_all(base_records)
    assert abs(baseline["pq"] - BASELINE_PQ) < 1e-9, baseline["pq"]
    print(f"baseline PQ {baseline['pq']:.10f}  TP/FP/FN "
          f"{baseline['tp']}/{baseline['fp']}/{baseline['fn']}\n")

    print(f"{'rule':34} {'admitted':>9} {'PQ':>10} {'delta':>10} {'TP':>5} {'FP':>5} "
          f"{'up':>4} {'down':>5}")
    report = dict(experiment_id="exp121", baseline_pq=baseline["pq"], rules=[])
    for label, rule in rules:
        admitted, per_record = 0, []
        for name, passes in records.items():
            selected = [(s, m) for s, m in prepared[name] if s >= BASE_CONF]
            for s, m in prepared[name]:
                if s >= BASE_CONF:
                    continue
                row = index.get((name, round(s, 9)))
                if row is not None and rule(row):
                    selected.append((s, m))
                    admitted += 1
            emitted = paint(selected, args.min_area) if admitted else base_emissions[name]
            for _, truths in passes:
                per_record.append((emitted, truths))
        result = score_all(per_record)
        up = down = 0
        for (before, truths), (after, _) in zip(base_records, per_record):
            was, now = per_truth_best(before, truths), per_truth_best(after, truths)
            up += int(((was < MATCH) & (now >= MATCH)).sum())
            down += int(((was >= MATCH) & (now < MATCH)).sum())
        delta = result["pq"] - baseline["pq"]
        print(f"{label:34} {admitted:>9} {result['pq']:10.6f} {delta:>+10.6f} "
              f"{result['tp']:>5} {result['fp']:>5} {up:>4} {down:>5}")
        report["rules"].append(dict(label=label, admitted=admitted,
                                    pq=result["pq"], delta_pq=round(delta, 6),
                                    tp=result["tp"], fp=result["fp"], fn=result["fn"],
                                    crossings_up=up, crossings_down=down))

    best = max(report["rules"], key=lambda row: row["delta_pq"])
    (args.output / "report.json").write_text(json.dumps(report, indent=2))
    print(f"\nbest rule: {best['label']} at {best['delta_pq']:+.6f}")
    print("Compare against the +0.020822 oracle ceiling from exp118.")


if __name__ == "__main__":
    main()
