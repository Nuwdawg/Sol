"""exp127: the cross-architecture feature at the operating point that actually matters.

exp124 measured `confirmed` - whether exp097 independently proposes a mask
overlapping this exp002 candidate at IoU >= 0.5 - at **AUC 0.688**. That is the
strongest single feature anyone has measured on this question, above `score`
(0.641), `confirm_iou` (0.651) and the fitted nine-feature model's out-of-fold
0.652.

Every rule in exp124 was still negative, and the reason is the grid, not the
feature. Its score cuts were quantiles over all candidates, so even the tightest
rule admitted 123 records. exp125 and exp126 found the paying region is roughly
24 to 36 admitted records. The feature was never tested where gains exist.

So this joins exp124's three cross-architecture features to exp120's nine, refits
the same out-of-fold logistic model grouped by photograph, and sweeps tight
operating points.

Honest accounting of what this can and cannot show
--------------------------------------------------
Every number here is optimistic in the same way exp125's and exp126's were: the
operating point is chosen after seeing the official metric on the same validation
split. With four feature sets and six values of k that freedom is wider still. The
defensible reading is the **ordering** of the feature sets, not any single value,
and a set that wins only at one k is noise rather than signal.

`joint` from exp124 is excluded deliberately: it is score x (0.5 + confirm_iou),
a product of two features already present, and adding it would let the model fit
the same evidence twice.
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
CEILING = 0.020822
SHAPE = (2048, 2048)
NINE = ("score", "area", "elongation", "solidity", "half_width", "extent",
        "limb", "absorption", "contrast")
CROSS = ("confirm_iou", "partner_score", "confirmed")


def eroded(entry, grow):
    mask = mu.decode(entry["rle"])
    if grow < 0:
        mask = cv2.erode(mask, KERNEL, iterations=-grow)
    return mask


def base_emission(entries, grow, min_area):
    candidates = []
    for entry in entries:
        if entry["score"] < BASE_CONF:
            continue
        mask = eroded(entry, grow)
        if int(mask.sum()) >= min_area:
            candidates.append((float(entry["score"]), mask))
    return [encode(m) for _, m, _ in paint_panoptic(candidates, min_area=min_area)]


def claimed_from(rles):
    if not rles:
        return np.zeros(SHAPE, np.uint8)
    merged = mu.merge([dict(size=r["size"], counts=r["counts"].encode("ascii"))
                       if isinstance(r["counts"], str) else r for r in rles])
    return mu.decode(merged)


def score_all(per_record):
    rows = [pq_rle(predicted, truths) for predicted, truths in per_record]
    aggregate = aggregate_pq(rows)
    aggregate["iou_sum"] = float(sum(row["iou_sum"] for row in rows))
    return aggregate


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pool", type=Path,
                        default=Path("data/exp033_cached_tta/candidates_0.5_0.5.json"))
    parser.add_argument("--nine", type=Path,
                        default=Path("data/exp120_verifier/candidates.json"))
    parser.add_argument("--cross", type=Path,
                        default=Path("data/exp124_cross_architecture/candidates.json"))
    parser.add_argument("--output", type=Path, default=Path("data/exp127_confirmed"))
    parser.add_argument("--grow", type=int, default=-1)
    parser.add_argument("--min-area", type=int, default=300)
    parser.add_argument("--folds", type=int, default=5)
    args = parser.parse_args()
    cv2.setNumThreads(2)
    args.output.mkdir(parents=True, exist_ok=True)

    try:
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import roc_auc_score
        from sklearn.model_selection import GroupKFold
        from sklearn.preprocessing import StandardScaler
    except ImportError:
        raise SystemExit("scikit-learn is required for exp127")

    nine = json.loads(args.nine.read_text())
    cross = {(r["photograph"], round(float(r["score"]), 9)): r
             for r in json.loads(args.cross.read_text())}
    rows = []
    for row in nine:
        partner = cross.get((row["photograph"], round(float(row["score"]), 9)))
        if partner is None:
            continue
        if partner["label"] != row["label"]:
            raise SystemExit(f"label mismatch for {row['photograph']} {row['score']}")
        rows.append({**row, **{name: float(partner[name]) for name in CROSS}})
    print(f"joined {len(rows)} of {len(nine)} candidates, "
          f"admissible {sum(r['label'] for r in rows)}")
    if len(rows) < 0.9 * len(nine):
        raise SystemExit("join lost too many candidates; keys disagree")

    labels = np.array([r["label"] for r in rows], int)
    groups = np.array([r["photograph"] for r in rows])
    print("\nsingle-feature AUC on the joined set:")
    for name in CROSS + ("score",):
        values = np.array([r[name] for r in rows], float)
        print(f"  {name:16} {roc_auc_score(labels, values):6.3f}")

    annotations = next(Path(".").rglob("MAGFiLO_1.0_Annotations_kaggle2026_train.json"))
    records, _ = records_for(annotations)
    pool = json.loads(args.pool.read_text())
    base_rle, base_records = {}, []
    for photograph, passes in records.items():
        base_rle[photograph] = base_emission(pool[photograph], args.grow, args.min_area)
        for _, truths in passes:
            base_records.append((base_rle[photograph], truths))
    baseline = score_all(base_records)
    assert abs(baseline["pq"] - BASELINE_PQ) < 1e-9, baseline["pq"]
    print(f"\nbaseline PQ {baseline['pq']:.10f}  TP/FP/FN "
          f"{baseline['tp']}/{baseline['fp']}/{baseline['fn']}")
    print(f"ceiling +{CEILING:.6f}   exp125 nine-feature best +0.001698\n")

    sets = {"nine": NINE, "cross only": CROSS, "twelve": NINE + CROSS}
    splitter = GroupKFold(n_splits=args.folds)
    out_of_fold = {}
    for label, names in sets.items():
        matrix = np.array([[r[name] for name in names] for r in rows], float)
        predictions = np.zeros(len(rows))
        for train, test in splitter.split(matrix, labels, groups):
            scaler = StandardScaler().fit(matrix[train])
            model = LogisticRegression(max_iter=2000, class_weight="balanced").fit(
                scaler.transform(matrix[train]), labels[train])
            predictions[test] = model.predict_proba(scaler.transform(matrix[test]))[:, 1]
        out_of_fold[label] = predictions
        ranked = labels[np.argsort(-predictions)]
        print(f"{label:12} out-of-fold AUC {roc_auc_score(labels, predictions):.3f}  "
              f"top10 {ranked[:10].mean():.3f}  top20 {ranked[:20].mean():.3f}  "
              f"top30 {ranked[:30].mean():.3f}")

    report = dict(experiment_id="exp127", baseline_pq=baseline["pq"], ceiling=CEILING,
                  candidates=len(rows), admissible=int(labels.sum()), rules=[])
    print(f"\n{'feature set':12} {'top k':>6} {'admitted':>9} {'PQ':>10} {'delta':>10} "
          f"{'up':>4} {'prec':>6}")
    for label, predictions in out_of_fold.items():
        lookup = {(r["photograph"], round(float(r["score"]), 9)): value
                  for r, value in zip(rows, predictions)}
        for k in (10, 15, 20, 25, 30, 40):
            cut = float(np.sort(predictions)[::-1][min(k, len(predictions)) - 1])
            admitted, per_record = 0, []
            for photograph, passes in records.items():
                claimed = claimed_from(base_rle[photograph])
                extra = []
                for entry in sorted((e for e in pool[photograph] if e["score"] < BASE_CONF),
                                    key=lambda e: -e["score"]):
                    value = lookup.get((photograph, round(float(entry["score"]), 9)))
                    if value is None or value < cut:
                        continue
                    mask = eroded(entry, args.grow)
                    if int(mask.sum()) < args.min_area:
                        continue
                    remainder = ((mask > 0) & (claimed == 0)).astype(np.uint8)
                    if int(remainder.sum()) < args.min_area:
                        continue
                    extra.append(encode(remainder))
                    claimed |= remainder
                admitted += len(extra) * len(passes)
                emitted = base_rle[photograph] + extra
                for _, truths in passes:
                    per_record.append((emitted, truths))
                del claimed
            result = score_all(per_record)
            up = result["tp"] - baseline["tp"]
            delta = result["pq"] - baseline["pq"]
            precision = up / admitted if admitted else 0.0
            print(f"{label:12} {k:>6} {admitted:>9} {result['pq']:>10.6f} {delta:>+10.6f} "
                  f"{up:>4} {precision:>6.3f}")
            report["rules"].append(dict(features=label, top_k=k, admitted_records=admitted,
                                        pq=result["pq"], delta_pq=round(delta, 6),
                                        crossings_up=up, precision=round(precision, 4)))

    best = max(report["rules"], key=lambda r: r["delta_pq"])
    (args.output / "report.json").write_text(json.dumps(report, indent=2))
    print(f"\nbest: {best['features']} top {best['top_k']} at {best['delta_pq']:+.6f} "
          f"({100 * best['delta_pq'] / CEILING:.1f}% of ceiling)")
    print("Read the ordering of the feature sets, not any single value: every number")
    print("here picks its operating point after seeing this same validation split.")


if __name__ == "__main__":
    main()
