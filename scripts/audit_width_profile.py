"""The width profile w(s), not one scalar summary of it.

Two statistics have already been tested and rejected for identifying merged
instances: width CV (AUC 0.457) and rho = w_min / median(w) (AUC 0.712 but 3-7%
precision, net -207 objects). The user's objection to stopping there is exact:

    [5,5,5,5,5,5,2,5,5]  and  [4,5,6,5,4,5,6,5,4]

have the same coefficient of variation, yet the first has an obvious neck and the
second does not. A single scalar destroys the structure that distinguishes them.

This audit computes the full profile and derives statistics that preserve local
structure: the sharpest local dip, the steepest derivative, how many distinct
maxima the profile has, and the depth of the deepest interior minimum relative to
the peaks on either side of it - a saddle measure, which is what a merge of two
objects actually produces.

The profile comes from the distance transform sampled along the mask's principal
axis: for each of `bins` slabs, the maximum distance-to-background inside that
slab, which is the local half-width. No skeleton is used, because skeletons grow
spurious branches under small boundary perturbations.

Populations: oracle-splittable 1:N merges against healthy 1:1 masks, the same two
groups exp105c used, so the numbers are directly comparable. Both directions of
every feature are scored with precision at operating points.
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
TOUCH = 0.05


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


def width_profile(mask, bins=32, edge=3):
    """Local half-width along the principal axis, as a sequence."""
    ys, xs = np.nonzero(mask)
    if len(xs) < 80:
        return None
    points = np.stack([xs, ys], 1).astype(np.float32)
    centre = points.mean(0)
    _, _, axes = np.linalg.svd(points - centre, full_matrices=False)
    along = (points - centre) @ axes[0]
    distance = cv2.distanceTransform(mask, cv2.DIST_L2, 5)[ys, xs]
    edges = np.linspace(along.min(), along.max(), bins + 1)
    profile = []
    for index in range(bins):
        selected = (along >= edges[index]) & (along < edges[index + 1])
        if selected.sum() >= 5:
            profile.append(float(distance[selected].max()))
    if len(profile) < 2 * edge + 3:
        return None
    return np.asarray(profile[edge:-edge], dtype=float)


def profile_features(profile):
    """Statistics that keep local structure instead of averaging it away."""
    median = float(np.median(profile))
    if median <= 1e-6:
        return None
    normalised = profile / median
    derivative = np.abs(np.diff(normalised))
    interior = normalised[1:-1]
    # Saddle depth: the deepest interior minimum against the higher of the two
    # peaks flanking it. A merge of two objects leaves a dip between two bodies.
    saddle = 0.0
    if interior.size:
        position = int(np.argmin(interior)) + 1
        left = normalised[:position].max() if position else normalised[0]
        right = normalised[position + 1:].max() if position + 1 < normalised.size else normalised[-1]
        saddle = float(min(left, right) - normalised[position])
    # Count of distinct local maxima above the median, a proxy for "two bodies".
    peaks = int(sum(1 for i in range(1, normalised.size - 1)
                    if normalised[i] >= normalised[i - 1] and normalised[i] > normalised[i + 1]
                    and normalised[i] > 1.0))
    return dict(rho=float(normalised.min()),
                saddle_depth=saddle,
                max_slope=float(derivative.max()) if derivative.size else 0.0,
                mean_slope=float(derivative.mean()) if derivative.size else 0.0,
                peak_count=peaks,
                peak_ratio=float(normalised.max() / max(normalised.min(), 1e-6)),
                width_cv=float(np.std(profile) / max(np.mean(profile), 1e-6)))


def rank_auc(positive, negative):
    positive, negative = np.asarray(positive, float), np.asarray(negative, float)
    if not positive.size or not negative.size:
        return float("nan")
    ranks = np.concatenate([positive, negative]).argsort().argsort() + 1
    return float((ranks[:positive.size].sum() - positive.size * (positive.size + 1) / 2)
                 / (positive.size * negative.size))


def best_operating_point(positive, negative):
    best = None
    for direction in (+1, -1):
        for quantile in (0.70, 0.80, 0.90, 0.95, 0.98):
            cut = float(np.quantile(negative * direction, quantile))
            caught = int((positive * direction >= cut).sum())
            damaged = int((negative * direction >= cut).sum())
            if best is None or caught - damaged > best["net"]:
                best = dict(side="high" if direction > 0 else "low", quantile=quantile,
                            caught=caught, damaged=damaged, net=caught - damaged,
                            precision=round(caught / max(caught + damaged, 1), 4))
    return best


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pool", type=Path,
                        default=Path("data/exp033_cached_tta/candidates_0.5_0.5.json"))
    parser.add_argument("--output", type=Path, default=Path("data/exp111_width_profile"))
    parser.add_argument("--conf", type=float, default=0.35)
    parser.add_argument("--grow", type=int, default=-1)
    parser.add_argument("--min-area", type=int, default=100)
    args = parser.parse_args()
    cv2.setNumThreads(2)
    args.output.mkdir(parents=True, exist_ok=True)
    annotations = next(Path(".").rglob("MAGFiLO_1.0_Annotations_kaggle2026_train.json"))
    records, split = records_for(annotations)
    pool = json.loads(args.pool.read_text())

    merges, healthy = [], []
    for photograph, passes in records.items():
        emitted = emit(pool[photograph], args.conf, args.grow, args.min_area)
        if not emitted:
            continue
        decoded = [mu.decode(rle) for rle in emitted]
        for _, truths in passes:
            if not truths:
                continue
            ious = np.asarray(mu.iou(emitted, truths, [0] * len(truths)))
            for p in range(len(emitted)):
                covered = [g for g in range(len(truths)) if ious[p, g] > TOUCH]
                profile = width_profile(decoded[p])
                if profile is None:
                    continue
                features = profile_features(profile)
                if features is None:
                    continue
                if len(covered) >= 2:
                    rescued = 0
                    for g in covered:
                        piece = decoded[p] & mu.decode(truths[g])
                        if piece.sum() < 40:
                            continue
                        after = float(mu.iou([encode(piece)], [truths[g]], [0])[0][0])
                        if after >= 0.5 > ious[:, g].max():
                            rescued += 1
                    if rescued:
                        merges.append(features)
                elif len(covered) == 1 and ious[p, covered[0]] >= 0.5 and \
                        len([q for q in range(len(emitted)) if ious[q, covered[0]] > TOUCH]) == 1:
                    healthy.append(features)

    print(f"oracle-splittable merges {len(merges)}   healthy 1:1 {len(healthy)}\n")
    print(f"{'feature':14s} {'AUC':>6} {'AUCneg':>7} {'merge med':>11} {'healthy med':>12}"
          f"  best over both directions")
    report = dict(split=split.summary(), merges=len(merges), healthy=len(healthy), features={})
    for key in ("rho", "saddle_depth", "max_slope", "mean_slope", "peak_count",
                "peak_ratio", "width_cv"):
        positive = np.array([r[key] for r in merges], float)
        negative = np.array([r[key] for r in healthy], float)
        if not positive.size or not negative.size:
            continue
        area = rank_auc(positive, negative)
        best = best_operating_point(positive, negative)
        report["features"][key] = dict(auc=round(area, 4), best=best)
        print(f"{key:14s} {area:6.3f} {1 - area:7.3f} {np.median(positive):11.4f} "
              f"{np.median(negative):12.4f}  {best['side']} q{best['quantile']:.2f} "
              f"net {best['net']:+d} precision {best['precision']:.3f}")
    (args.output / "report.json").write_text(json.dumps(report, indent=2))
    print(f"\nwrote {args.output / 'report.json'}")


if __name__ == "__main__":
    main()
