"""exp124: does a second architecture confirm a discarded candidate?

exp120 screened nine features on the 526 discarded candidates in [0.10, 0.35) and
found none reaching the precision the ceiling demands. Only `score` (AUC 0.641) and
`limb` flipped (0.602) beat chance, and exp121 showed their best combination earns
+0.000480 of a +0.020822 ceiling. exp122 then ruled out the admission geometry, so
the gate is the binding constraint and one feature family remains untested.

Why this is not exp107 repeated
-------------------------------
exp107 rejected cross-architecture *disagreement* as a signal that an emitted
mask's boundary was wrong, and that rejection stands. This asks a different
question of the same evidence: **does exp097 independently propose a filament here
at all?** A second architecture, trained at a different resolution with a different
optimiser, proposing the same object is positive evidence that the object exists.
Disagreeing about where its edge lies is not. exp102 already showed these two pools
carry complementary recognition - the union found 852 filaments where exp002 alone
found 835 - so the information is present at the level of proposals.

The feature is therefore *confirmation*, not disagreement: for each discarded
exp002 candidate, the best IoU any exp097 instance reaches with it, and that
instance's own confidence.

Runs on Kaggle CPU because exp097's validation pool lives in exp098's kernel
output and is deliberately never copied to the user's machine.

Counting note, learned the hard way in exp121
---------------------------------------------
TP and FP are pooled **per record**, and each photograph is scored once per
annotation pass (180 records over 106 photographs). Instance counts here are
therefore reported in record units, so precision figures are comparable with the
oracle's 120/215 = 0.558.
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
CEILING = 0.020822


def load_pool(path):
    pool = json.loads(Path(path).read_text())
    cleaned = {}
    for photograph, entries in pool.items():
        rows = []
        for entry in entries:
            rle = entry["rle"]
            counts = rle["counts"]
            rows.append(dict(score=float(entry["score"]),
                             rle=dict(size=rle["size"],
                                      counts=counts.decode("ascii")
                                      if isinstance(counts, bytes) else counts)))
        cleaned[Path(photograph).name] = rows
    return cleaned


def prepare(entries, grow, min_area):
    prepared = []
    for entry in entries:
        mask = mu.decode(entry["rle"])
        if grow < 0:
            mask = cv2.erode(mask, KERNEL, iterations=-grow)
        if int(mask.sum()) < min_area:
            continue
        prepared.append((float(entry["score"]), mask))
    return prepared


def score_all(per_record):
    rows = [pq_rle(predicted, truths) for predicted, truths in per_record]
    aggregate = aggregate_pq(rows)
    aggregate["iou_sum"] = float(sum(row["iou_sum"] for row in rows))
    return aggregate


def rank_auc(positive, negative):
    positive, negative = np.asarray(positive, float), np.asarray(negative, float)
    if not positive.size or not negative.size:
        return float("nan")
    ranks = np.concatenate([positive, negative]).argsort().argsort() + 1
    return float((ranks[:positive.size].sum() - positive.size * (positive.size + 1) / 2)
                 / (positive.size * negative.size))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--exp002-pool", type=Path, required=True)
    parser.add_argument("--exp097-pool", type=Path, required=True)
    parser.add_argument("--annotations", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--grow", type=int, default=-1)
    parser.add_argument("--min-area", type=int, default=300)
    parser.add_argument("--floor", type=float, default=0.10)
    args = parser.parse_args()
    cv2.setNumThreads(2)
    args.output.mkdir(parents=True, exist_ok=True)
    records, split = records_for(args.annotations)
    exp002, exp097 = load_pool(args.exp002_pool), load_pool(args.exp097_pool)
    missing = [n for n in records if n not in exp002 or n not in exp097]
    if missing:
        raise SystemExit(f"pools miss {len(missing)} photographs: {missing[:3]}")

    rows, base_rle, base_masks, base_records, passes_for = [], {}, {}, [], {}
    for photograph, passes in records.items():
        prepared = prepare(exp002[photograph], args.grow, args.min_area)
        emitted = [encode(m) for _, m, _ in paint_panoptic(
            [(s, m) for s, m in prepared if s >= BASE_CONF], min_area=args.min_area)]
        base_rle[photograph] = emitted
        base_masks[photograph] = [mu.decode(r) for r in emitted]
        passes_for[photograph] = len(passes)
        for _, truths in passes:
            base_records.append((emitted, truths))
        unmatched = []
        for _, truths in passes:
            if not truths:
                continue
            best = (np.asarray(mu.iou(emitted, truths, [0] * len(truths))).max(axis=0)
                    if emitted else np.zeros(len(truths)))
            unmatched.extend(truths[g] for g in range(len(truths)) if best[g] < MATCH)
        partner = exp097[photograph]
        partner_rle = [e["rle"] for e in partner]
        for score, mask in prepared:
            if score >= BASE_CONF or score < args.floor:
                continue
            rle = encode(mask)
            support, partner_score = 0.0, 0.0
            if partner_rle:
                ious = np.asarray(mu.iou([rle], partner_rle, [0] * len(partner_rle)))[0]
                position = int(np.argmax(ious))
                support = float(ious[position])
                partner_score = float(partner[position]["score"])
            label = 0
            if unmatched:
                label = int(float(np.asarray(
                    mu.iou([rle], unmatched, [0] * len(unmatched))).max()) >= MATCH)
            rows.append(dict(photograph=photograph, score=score, label=label,
                             confirm_iou=support, partner_score=partner_score,
                             confirmed=float(support >= MATCH),
                             joint=score * (0.5 + support)))
    baseline = score_all(base_records)
    assert abs(baseline["pq"] - BASELINE_PQ) < 1e-9, baseline["pq"]
    positives = sum(r["label"] for r in rows)
    print(f"baseline PQ {baseline['pq']:.10f}  TP/FP/FN "
          f"{baseline['tp']}/{baseline['fp']}/{baseline['fn']}")
    print(f"discarded candidates {len(rows)}, admissible {positives}, "
          f"base precision {positives / max(len(rows), 1):.4f}")
    print(f"ceiling +{CEILING:.6f}\n")

    print(f"{'feature':16} {'AUC':>6} {'flipped':>8}")
    report = dict(experiment_id="exp124", split=split.summary(), candidates=len(rows),
                  admissible=positives, features={}, rules=[])
    for feature in ("score", "confirm_iou", "partner_score", "confirmed", "joint"):
        positive = [r[feature] for r in rows if r["label"] == 1]
        negative = [r[feature] for r in rows if r["label"] == 0]
        area = rank_auc(positive, negative)
        report["features"][feature] = round(area, 4)
        print(f"{feature:16} {area:6.3f} {max(area, 1 - area):8.3f}")

    confirm_cuts = [0.10, 0.30, 0.50]
    score_cuts = [float(np.quantile([r["score"] for r in rows], q)) for q in (0.0, 0.5, 0.8)]
    rules = [("admit nothing", lambda r: False)]
    for confirm in confirm_cuts:
        for quantile, cut in zip((0.0, 0.5, 0.8), score_cuts):
            rules.append((f"confirm>={confirm:.2f} score q{quantile:.1f}",
                          lambda r, c=confirm, s=cut: r["confirm_iou"] >= c and r["score"] >= s))

    index = {(r["photograph"], round(r["score"], 9)): r for r in rows}
    print(f"\n{'rule':32} {'admitted':>9} {'PQ':>10} {'delta':>10} {'up':>4} "
          f"{'prec':>6}")
    for label, rule in rules:
        admitted, per_record = 0, []
        for photograph, passes in records.items():
            claimed = np.zeros((2048, 2048), np.uint8)
            for mask in base_masks[photograph]:
                claimed |= mask
            extra = []
            prepared = prepare(exp002[photograph], args.grow, args.min_area)
            for score, mask in sorted((p for p in prepared if p[0] < BASE_CONF),
                                      key=lambda pair: -pair[0]):
                row = index.get((photograph, round(score, 9)))
                if row is None or not rule(row):
                    continue
                remainder = ((mask > 0) & (claimed == 0)).astype(np.uint8)
                if int(remainder.sum()) < args.min_area:
                    continue
                extra.append(encode(remainder))
                claimed |= remainder
            admitted += len(extra) * passes_for[photograph]
            emitted = base_rle[photograph] + extra
            for _, truths in passes:
                per_record.append((emitted, truths))
        result = score_all(per_record)
        up = result["tp"] - baseline["tp"]
        delta = result["pq"] - baseline["pq"]
        precision = up / admitted if admitted else 0.0
        print(f"{label:32} {admitted:>9} {result['pq']:>10.6f} {delta:>+10.6f} "
              f"{up:>4} {precision:>6.3f}")
        report["rules"].append(dict(label=label, admitted_records=admitted,
                                    pq=result["pq"], delta_pq=round(delta, 6),
                                    crossings_up=up, precision=round(precision, 4)))

    best = max(report["rules"], key=lambda row: row["delta_pq"])
    (args.output / "report.json").write_text(json.dumps(report, indent=2))
    (args.output / "candidates.json").write_text(json.dumps(rows))
    print(f"\nbest: {best['label']} at {best['delta_pq']:+.6f} "
          f"({100 * best['delta_pq'] / CEILING:.1f}% of ceiling)")
    print("exp121's best, without any cross-architecture evidence, was +0.000480.")


if __name__ == "__main__":
    main()
