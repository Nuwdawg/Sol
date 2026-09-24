"""exp126: is a discarded candidate more credible when it stands apart from what was emitted?

The nine features in exp120 all describe a candidate in isolation - its score, its
shape, its darkness, its position on the disk. None of them asks the one relational
question available for free: **how much of this candidate is already covered by an
instance we emitted, and how far is it from the nearest one?**

The reasoning is mechanical rather than physical. A sub-threshold candidate that
sits on top of an emitted mask is usually the same filament proposed again at lower
confidence; carving removes its overlapping area and what survives is a sliver. A
candidate lying in empty disk is a genuinely new proposal. So overlap should predict
the label negatively and distance positively, and neither is in the feature set.

Two features are added to exp125's nine:

- `claimed_fraction`: the share of the candidate's pixels already emitted.
- `nearest_gap`: distance in pixels from the candidate to the nearest emitted
  instance, zero when they touch, capped so an empty photograph does not dominate.

Then the same honest protocol as exp125: out-of-fold `GroupKFold` predictions
grouped by photograph, admission carved against claimed pixels, and rescoring by
`aggregate_pq` after global rematching, in record units.

Selection caution
-----------------
exp125's +0.001698 was already the best of eight configurations scored on this same
split, so it is optimistic. Adding features and operating points here widens that
freedom further. The comparison below is therefore between two optimistic numbers,
and the honest reading is the direction and size of the difference, not either
value as an unbiased estimate.

Memory: masks are held one photograph at a time, after the first version of exp125
was killed for decoding 1414 dense 2048x2048 masks at once.
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
EXP125_BEST = 0.001698
SHAPE = (2048, 2048)
GAP_CAP = 200.0
BASE_FEATURES = ("score", "area", "elongation", "solidity", "half_width", "extent",
                 "limb", "absorption", "contrast")
ADDED = ("claimed_fraction", "nearest_gap")


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
    parser.add_argument("--features", type=Path,
                        default=Path("data/exp120_verifier/candidates.json"))
    parser.add_argument("--output", type=Path, default=Path("data/exp126_isolation"))
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
        raise SystemExit("scikit-learn is required for exp126")

    rows = json.loads(args.features.read_text())
    index = {(row["photograph"], round(row["score"], 9)): row for row in rows}
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
    print(f"baseline PQ {baseline['pq']:.10f}  TP/FP/FN "
          f"{baseline['tp']}/{baseline['fp']}/{baseline['fn']}")

    # The two relational features, one photograph at a time.
    for photograph in records:
        claimed = claimed_from(base_rle[photograph])
        # Distance to the nearest emitted pixel; the transform runs on the complement.
        gap = (cv2.distanceTransform((claimed == 0).astype(np.uint8), cv2.DIST_L2, 5)
               if claimed.any() else np.full(SHAPE, GAP_CAP, np.float32))
        for entry in pool[photograph]:
            score = float(entry["score"])
            row = index.get((photograph, round(score, 9)))
            if row is None:
                continue
            mask = eroded(entry, args.grow)
            area = int(mask.sum())
            if area <= 0:
                row["claimed_fraction"], row["nearest_gap"] = 1.0, 0.0
                continue
            row["claimed_fraction"] = float((mask & claimed).sum() / area)
            row["nearest_gap"] = float(min(gap[mask > 0].min(), GAP_CAP))
        del claimed, gap

    scored = [row for row in rows if "claimed_fraction" in row]
    labels = np.array([row["label"] for row in scored], int)
    groups = np.array([row["photograph"] for row in scored])
    print(f"candidates {len(scored)}, admissible {labels.sum()}, "
          f"base rate {labels.mean():.4f}")
    for feature in ADDED:
        values = np.array([row[feature] for row in scored], float)
        auc = roc_auc_score(labels, values)
        print(f"  {feature:18} AUC {auc:.3f}  flipped {max(auc, 1 - auc):.3f}  "
              f"positive median {np.median(values[labels == 1]):.3f}  "
              f"negative median {np.median(values[labels == 0]):.3f}")

    report = dict(experiment_id="exp126", baseline_pq=baseline["pq"], ceiling=CEILING,
                  exp125_best=EXP125_BEST, candidates=len(scored),
                  admissible=int(labels.sum()), sets={}, rules=[])
    splitter = GroupKFold(n_splits=args.folds)
    out_of_fold = {}
    for label, names in (("nine (exp125)", BASE_FEATURES),
                         ("eleven (+isolation)", BASE_FEATURES + ADDED)):
        matrix = np.array([[row[name] for name in names] for row in scored], float)
        predictions = np.zeros(len(scored))
        for train, test in splitter.split(matrix, labels, groups):
            scaler = StandardScaler().fit(matrix[train])
            model = LogisticRegression(max_iter=2000, class_weight="balanced").fit(
                scaler.transform(matrix[train]), labels[train])
            predictions[test] = model.predict_proba(scaler.transform(matrix[test]))[:, 1]
        out_of_fold[label] = predictions
        ranked = labels[np.argsort(-predictions)]
        auc = roc_auc_score(labels, predictions)
        report["sets"][label] = dict(auc=round(float(auc), 4),
                                     precision_at={k: round(float(ranked[:k].mean()), 4)
                                                   for k in (20, 40, 60)})
        print(f"\n{label}: out-of-fold AUC {auc:.3f}  "
              f"top20 {ranked[:20].mean():.3f}  top40 {ranked[:40].mean():.3f}")

    print(f"\n{'feature set':22} {'top k':>6} {'admitted':>9} {'PQ':>10} {'delta':>10} "
          f"{'up':>4} {'prec':>6}")
    for label, predictions in out_of_fold.items():
        lookup = {(row["photograph"], round(row["score"], 9)): value
                  for row, value in zip(scored, predictions)}
        for k in (15, 20, 30, 40):
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
            print(f"{label:22} {k:>6} {admitted:>9} {result['pq']:>10.6f} {delta:>+10.6f} "
                  f"{up:>4} {precision:>6.3f}")
            report["rules"].append(dict(features=label, top_k=k, admitted_records=admitted,
                                        pq=result["pq"], delta_pq=round(delta, 6),
                                        crossings_up=up, precision=round(precision, 4)))

    best = max(report["rules"], key=lambda row: row["delta_pq"])
    (args.output / "report.json").write_text(json.dumps(report, indent=2))
    print(f"\nbest: {best['features']} top {best['top_k']} at {best['delta_pq']:+.6f} "
          f"({100 * best['delta_pq'] / CEILING:.1f}% of ceiling)")
    print(f"exp125's nine-feature best was +{EXP125_BEST:.6f}; both are optimistic,")
    print("so the difference between them is the signal, not either number alone.")


if __name__ == "__main__":
    main()
