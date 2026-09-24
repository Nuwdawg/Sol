"""exp120: can any observable feature sort discarded candidates from 24% to 50% precision?

exp118 measured that the 601 candidates in [0.10, 0.35) contain 132 admissible ones
worth +0.020822, and that admitting them blind yields 23.5-25.2% precision. The
verifier table says break-even is about 35% and that clearing the 0.472 bar needs
about **50%** at k=120. So this audit asks one question: does anything observable
rank a discarded candidate as real?

Why this is not a repeat of the failed gates
--------------------------------------------
exp105c through exp107 rejected nine feature families, but every one of them asked
"is this *already emitted* mask too small", sampling a 4 px ring around a mask that
was already believed. This asks "is this *discarded proposal* a filament at all".
Different population, different question, and no feature here has been scored on it.

Cross-architecture agreement deserves particular note. exp107 rejected it as a
boundary-uncertainty signal, and that rejection stands. Using exp097's pool to
*confirm a proposal exists* is a different use of the same evidence, and it is the
one feature with an obvious physical reason to work: two architectures proposing
the same filament independently is real evidence, whereas two architectures
disagreeing about an edge is not.

Every feature is scored by rank AUC in both directions (a feature below chance is
a feature with its sign flipped, per the user's correction), and then by the only
thing that matters here - precision at operating points, against the 0.350
break-even and 0.500 target lines.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from pycocotools import mask as mu

from experiments.exp_033_inference.src.sweep import encode, records_for
from shared.utils import paint_panoptic

KERNEL = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
BASE_CONF = 0.35
MATCH = 0.5
BREAK_EVEN = 0.350
TARGET = 0.500


def emit(entries, conf, grow, min_area):
    candidates = []
    for entry in entries:
        if entry["score"] < conf:
            continue
        mask = mu.decode(entry["rle"])
        if grow < 0:
            mask = cv2.erode(mask, KERNEL, iterations=-grow)
        candidates.append((entry["score"], mask))
    return [encode(m) for _, m, _ in paint_panoptic(candidates, min_area=min_area)]


def disk_geometry(image):
    """Centre and radius of the solar disk, for the limb-distance feature."""
    rough = (image > 25).astype(np.uint8)
    rough = cv2.morphologyEx(rough, cv2.MORPH_CLOSE,
                             cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (31, 31)))
    contours, _ = cv2.findContours(rough, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    (x, y), radius = cv2.minEnclosingCircle(max(contours, key=cv2.contourArea))
    return float(x), float(y), float(radius)


def shape_features(mask):
    ys, xs = np.nonzero(mask)
    area = int(mask.sum())
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours or area < 20:
        return None
    biggest = max(contours, key=cv2.contourArea)
    (_, _), (width, height), _ = cv2.minAreaRect(biggest)
    short, long = sorted((max(width, 1.0), max(height, 1.0)))
    hull = cv2.contourArea(cv2.convexHull(biggest))
    distance = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
    return dict(area=float(area), elongation=float(long / short),
                solidity=float(area / hull) if hull > 0 else 0.0,
                half_width=float(np.median(distance[mask > 0])),
                extent=float(area / max(long * short, 1.0)),
                centre=(float(xs.mean()), float(ys.mean())))


def rank_auc(positive, negative):
    positive, negative = np.asarray(positive, float), np.asarray(negative, float)
    if not positive.size or not negative.size:
        return float("nan")
    ranks = np.concatenate([positive, negative]).argsort().argsort() + 1
    return float((ranks[:positive.size].sum() - positive.size * (positive.size + 1) / 2)
                 / (positive.size * negative.size))


def operating_points(positive, negative):
    """Precision and yield at quantile cuts, in whichever direction is better."""
    best = []
    for direction in (+1, -1):
        values = np.concatenate([positive, negative]) * direction
        for quantile in (0.30, 0.50, 0.70, 0.80, 0.90):
            cut = float(np.quantile(values, quantile))
            kept = int((positive * direction >= cut).sum())
            damaged = int((negative * direction >= cut).sum())
            if kept + damaged < 10:
                continue
            best.append(dict(side="high" if direction > 0 else "low", quantile=quantile,
                             kept=kept, admitted=kept + damaged,
                             precision=round(kept / (kept + damaged), 4)))
    return sorted(best, key=lambda row: -row["precision"])[:3]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pool", type=Path,
                        default=Path("data/exp033_cached_tta/candidates_0.5_0.5.json"))
    parser.add_argument("--images", type=Path,
                        default=Path("data/competition/unpacked/MAGFiLO_1.0_Kaggle_2026/"
                                     "train/train_images"))
    parser.add_argument("--output", type=Path, default=Path("data/exp120_verifier"))
    parser.add_argument("--floor", type=float, default=0.10)
    parser.add_argument("--grow", type=int, default=-1)
    parser.add_argument("--min-area", type=int, default=300)
    args = parser.parse_args()
    cv2.setNumThreads(2)
    args.output.mkdir(parents=True, exist_ok=True)
    annotations = next(Path(".").rglob("MAGFiLO_1.0_Annotations_kaggle2026_train.json"))
    records, split = records_for(annotations)
    pool = json.loads(args.pool.read_text())

    rows = []
    for photograph, passes in records.items():
        entries = pool[photograph]
        base = emit(entries, BASE_CONF, args.grow, args.min_area)
        unmatched = []
        for _, truths in passes:
            if not truths:
                continue
            best = (np.asarray(mu.iou(base, truths, [0] * len(truths))).max(axis=0)
                    if base else np.zeros(len(truths)))
            unmatched.extend(truths[g] for g in range(len(truths)) if best[g] < MATCH)
        image = cv2.imread(str(args.images / photograph), cv2.IMREAD_GRAYSCALE)
        if image is None:
            continue
        disk = disk_geometry(image)
        floated = image.astype(np.float32)
        dark = cv2.GaussianBlur(floated, (0, 0), 25.0) - floated
        for entry in entries:
            score = float(entry["score"])
            if score >= BASE_CONF or score < args.floor:
                continue
            mask = mu.decode(entry["rle"])
            if args.grow < 0:
                mask = cv2.erode(mask, KERNEL, iterations=-args.grow)
            if int(mask.sum()) < args.min_area:
                continue
            shape = shape_features(mask)
            if shape is None:
                continue
            label = 0
            if unmatched:
                reached = float(np.asarray(
                    mu.iou([encode(mask)], unmatched, [0] * len(unmatched))).max())
                label = int(reached >= MATCH)
            centre = shape.pop("centre")
            limb = 1.0
            if disk:
                x0, y0, radius = disk
                limb = float(np.hypot(centre[0] - x0, centre[1] - y0) / max(radius, 1.0))
            interior = dark[mask > 0]
            rows.append(dict(photograph=photograph, label=label, score=score, limb=limb,
                             absorption=float(np.median(interior)) if interior.size else 0.0,
                             contrast=float(np.percentile(interior, 90)) if interior.size else 0.0,
                             **shape))

    positives = [r for r in rows if r["label"] == 1]
    print(f"discarded candidates scored: {len(rows)}   admissible: {len(positives)}"
          f"   base precision: {len(positives) / max(len(rows), 1):.4f}")
    print(f"break-even {BREAK_EVEN:.3f}   target {TARGET:.3f}\n")

    features = ("score", "area", "elongation", "solidity", "half_width", "extent",
                "limb", "absorption", "contrast")
    print(f"{'feature':12} {'AUC':>6} {'flipped':>8}  best operating points")
    report = dict(experiment_id="exp120", split=split.summary(), candidates=len(rows),
                  admissible=len(positives),
                  base_precision=round(len(positives) / max(len(rows), 1), 4),
                  break_even=BREAK_EVEN, target=TARGET, features={})
    positive_rows = np.array([[r[f] for f in features] for r in rows if r["label"] == 1])
    negative_rows = np.array([[r[f] for f in features] for r in rows if r["label"] == 0])
    for index, feature in enumerate(features):
        if not positive_rows.size or not negative_rows.size:
            continue
        area = rank_auc(positive_rows[:, index], negative_rows[:, index])
        points = operating_points(positive_rows[:, index], negative_rows[:, index])
        report["features"][feature] = dict(auc=round(area, 4), operating_points=points)
        summary = "  ".join(f"q{p['quantile']:.2f} {p['precision']:.3f} ({p['kept']}/{p['admitted']})"
                            for p in points)
        print(f"{feature:12} {area:6.3f} {max(area, 1 - area):8.3f}  {summary}")

    (args.output / "report.json").write_text(json.dumps(report, indent=2))
    (args.output / "candidates.json").write_text(json.dumps(rows))
    print(f"\nwrote {args.output / 'report.json'}")
    print("A feature clearing 0.500 precision at useful yield is a deployable verifier.")
    print("Nothing here is deployable until that is shown on precision, not AUC.")


if __name__ == "__main__":
    main()
