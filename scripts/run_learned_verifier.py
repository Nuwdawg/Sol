"""exp125: does a fitted combination of the nine features verify a candidate?

exp120 screened the features one at a time and none reached the precision the
ceiling demands. exp121 then found that pairing `score` with `limb` beat either
alone (+0.000480 against -0.003296 for score alone), which is weak evidence that a
fitted combination might do better than any hand-drawn pair.

This fits one, honestly:

- **Out-of-fold predictions only.** Candidates are grouped by photograph and split
  with `GroupKFold`, so no candidate is scored by a model that saw another
  candidate from the same photograph. Fitting and scoring on the same rows would
  invent precision that does not exist at inference time.
- **Official metric, not precision alone.** Out-of-fold probabilities pick what to
  admit, admission carves against claimed pixels exactly as exp122's `whole` mode
  does, and the result is rescored by `aggregate_pq` after global rematching.
- **Record units.** TP and FP pool over 180 records across 106 photographs, so a
  candidate admitted on a two-pass photograph counts twice. exp121 was misread for
  exactly this reason, so precision here is reported in record units and is
  comparable with the oracle's 120/215 = 0.558.

Memory
------
The first version of this script decoded every candidate of every photograph up
front - about 1414 dense 2048x2048 masks, some 5.6 GB - and the run was killed.
Masks are now held one photograph at a time: only RLEs are cached, the claimed-pixel
map is built with `mu.merge` (an RLE union needing no dense decode), and candidates
are decoded singly. Peak is a few tens of megabytes.

The honest prior is that this is a thin dataset: 132 positives in 526 rows over
nine features, against a base rate of 0.251. A fitted model may simply recover
`score` and `limb`. That is a result too, and it is stated rather than hidden.
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
SHAPE = (2048, 2048)
FEATURES = ("score", "area", "elongation", "solidity", "half_width", "extent",
            "limb", "absorption", "contrast")


def eroded(entry, grow):
    mask = mu.decode(entry["rle"])
    if grow < 0:
        mask = cv2.erode(mask, KERNEL, iterations=-grow)
    return mask


def base_emission(entries, grow, min_area):
    """Paint the >= 0.35 candidates for one photograph, returning RLEs only."""
    candidates = []
    for entry in entries:
        if entry["score"] < BASE_CONF:
            continue
        mask = eroded(entry, grow)
        if int(mask.sum()) >= min_area:
            candidates.append((float(entry["score"]), mask))
    return [encode(m) for _, m, _ in paint_panoptic(candidates, min_area=min_area)]


def claimed_from(rles):
    """Union of emitted instances as a dense mask, via an RLE merge (no bulk decode)."""
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
    parser.add_argument("--output", type=Path, default=Path("data/exp125_learned_verifier"))
    parser.add_argument("--grow", type=int, default=-1)
    parser.add_argument("--min-area", type=int, default=300)
    parser.add_argument("--folds", type=int, default=5)
    args = parser.parse_args()
    cv2.setNumThreads(2)
    args.output.mkdir(parents=True, exist_ok=True)

    try:
        from sklearn.ensemble import GradientBoostingClassifier
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import roc_auc_score
        from sklearn.model_selection import GroupKFold
        from sklearn.preprocessing import StandardScaler
    except ImportError:
        raise SystemExit("scikit-learn is required for exp125")

    rows = json.loads(args.features.read_text())
    matrix = np.array([[row[name] for name in FEATURES] for row in rows], float)
    labels = np.array([row["label"] for row in rows], int)
    groups = np.array([row["photograph"] for row in rows])
    print(f"candidates {len(rows)}, admissible {labels.sum()}, "
          f"base rate {labels.mean():.4f}, photographs {len(set(groups))}")

    models = {
        "logistic": lambda: LogisticRegression(max_iter=2000, class_weight="balanced"),
        "gradient_boosting": lambda: GradientBoostingClassifier(
            n_estimators=120, max_depth=2, learning_rate=0.05, random_state=0),
    }
    out_of_fold = {}
    splitter = GroupKFold(n_splits=args.folds)
    for name, build in models.items():
        predictions = np.zeros(len(rows))
        for train, test in splitter.split(matrix, labels, groups):
            scaler = StandardScaler().fit(matrix[train])
            model = build().fit(scaler.transform(matrix[train]), labels[train])
            predictions[test] = model.predict_proba(scaler.transform(matrix[test]))[:, 1]
        out_of_fold[name] = predictions
        ranked = labels[np.argsort(-predictions)]
        print(f"\n{name}: out-of-fold AUC {roc_auc_score(labels, predictions):.3f}")
        print(f"  {'top k':>6} {'precision':>10}")
        for k in (20, 40, 60, 100, 150):
            print(f"  {k:>6} {ranked[:k].mean():>10.3f}")

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
    print(f"ceiling +{CEILING:.6f}   exp121 best +0.000480\n")

    report = dict(experiment_id="exp125", baseline_pq=baseline["pq"], ceiling=CEILING,
                  candidates=len(rows), admissible=int(labels.sum()), rules=[])
    print(f"{'model':18} {'top k':>6} {'admitted':>9} {'PQ':>10} {'delta':>10} "
          f"{'up':>4} {'prec':>6}")
    for name, predictions in out_of_fold.items():
        index = {(row["photograph"], round(row["score"], 9)): value
                 for row, value in zip(rows, predictions)}
        for k in (20, 40, 60, 100):
            cut = float(np.sort(predictions)[::-1][min(k, len(predictions)) - 1])
            admitted, per_record = 0, []
            for photograph, passes in records.items():
                claimed = claimed_from(base_rle[photograph])
                extra = []
                below = sorted((e for e in pool[photograph] if e["score"] < BASE_CONF),
                               key=lambda e: -e["score"])
                for entry in below:
                    value = index.get((photograph, round(float(entry["score"]), 9)))
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
            result = score_all(per_record)
            up = result["tp"] - baseline["tp"]
            delta = result["pq"] - baseline["pq"]
            precision = up / admitted if admitted else 0.0
            print(f"{name:18} {k:>6} {admitted:>9} {result['pq']:>10.6f} {delta:>+10.6f} "
                  f"{up:>4} {precision:>6.3f}")
            report["rules"].append(dict(model=name, top_k=k, admitted_records=admitted,
                                        pq=result["pq"], delta_pq=round(delta, 6),
                                        crossings_up=up, precision=round(precision, 4)))

    best = max(report["rules"], key=lambda row: row["delta_pq"])
    (args.output / "report.json").write_text(json.dumps(report, indent=2))
    print(f"\nbest: {best['model']} top {best['top_k']} at {best['delta_pq']:+.6f} "
          f"({100 * best['delta_pq'] / CEILING:.1f}% of ceiling)")
    print("Beating +0.000480 means the combination carries more than score and limb.")


if __name__ == "__main__":
    main()
