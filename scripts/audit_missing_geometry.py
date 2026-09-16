"""Where is the missing area: beyond the mask's ends, or alongside its spine?

exp105b measured that near-threshold 1:1 masks lack a median 575 truth pixels.
That says *how much* is missing, not *where*, and the two possibilities imply
different algorithms: area missing beyond the ends is truncation, repaired by
endpoint continuation along a supported path; area missing alongside the body is
thinness, repaired by local boundary refinement. Implementing either before
measuring which dominates would be guesswork.

Two measures, both skeleton-free because skeletons are unstable under small
boundary perturbations:

- **axial fraction**: project the mask's pixels onto its principal axis to get its
  own extent, then project the missing pixels. Missing pixels falling outside that
  extent lie beyond an end; the rest lie alongside.
- **distance profile**: distance from each missing pixel to the nearest mask pixel.
  Thinness keeps the missing area within a few pixels of the boundary; truncation
  places it far beyond.

Healthy matched pairs are the control. "Near-threshold masks are truncated" has to
mean more truncated than ordinary matches, not merely truncated in absolute terms.
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


def decompose(mask: np.ndarray, truth: np.ndarray) -> dict | None:
    """Split the missing truth area into axial (beyond the ends) and lateral."""
    missing = (truth & ~mask).astype(bool)
    if missing.sum() < 20 or mask.sum() < 50:
        return None
    ys, xs = np.nonzero(mask)
    points = np.stack([xs, ys], 1).astype(np.float32)
    centre = points.mean(0)
    _, _, axes = np.linalg.svd(points - centre, full_matrices=False)
    along = (points - centre) @ axes[0]
    low, high = np.percentile(along, 1), np.percentile(along, 99)

    my, mx = np.nonzero(missing)
    missing_points = np.stack([mx, my], 1).astype(np.float32)
    projected = (missing_points - centre) @ axes[0]
    axial = (projected < low) | (projected > high)

    distance = cv2.distanceTransform((1 - mask).astype(np.uint8), cv2.DIST_L2, 5)[my, mx]
    half_width = float(np.median(cv2.distanceTransform(mask, cv2.DIST_L2, 5)[ys, xs]))
    return dict(missing=int(missing.sum()),
                axial_fraction=float(axial.mean()),
                axial_pixels=int(axial.sum()),
                median_distance=float(np.median(distance)),
                p90_distance=float(np.percentile(distance, 90)),
                within_3px=float((distance <= 3).mean()),
                beyond_10px=float((distance > 10).mean()),
                mask_half_width=half_width)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pool", type=Path,
                        default=Path("data/exp033_cached_tta/candidates_0.5_0.5.json"))
    parser.add_argument("--annotations", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=Path("data/exp108_missing_geometry"))
    parser.add_argument("--conf", type=float, default=0.35)
    parser.add_argument("--grow", type=int, default=-1)
    parser.add_argument("--min-area", type=int, default=100)
    args = parser.parse_args()
    cv2.setNumThreads(2)
    args.output.mkdir(parents=True, exist_ok=True)
    annotations = args.annotations or next(Path(".").rglob("MAGFiLO_1.0_Annotations_kaggle2026_train.json"))
    records, split = records_for(annotations)
    pool = json.loads(args.pool.read_text())

    near, healthy = [], []
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
                if len(covered) != 1:
                    continue
                g = covered[0]
                if len([q for q in range(len(emitted)) if ious[q, g] > TOUCH]) != 1:
                    continue
                iou = float(ious[p, g])
                if iou < 0.4:
                    continue
                row = decompose(decoded[p], mu.decode(truths[g]))
                if row is None:
                    continue
                row["iou"] = iou
                (near if iou < 0.5 else healthy).append(row)

    def summarise(rows, label):
        if not rows:
            print(f"{label}: none")
            return {}
        median = lambda key: float(np.median([r[key] for r in rows]))
        out = dict(n=len(rows), missing=median("missing"), axial_fraction=median("axial_fraction"),
                   axial_pixels=median("axial_pixels"), median_distance=median("median_distance"),
                   p90_distance=median("p90_distance"), within_3px=median("within_3px"),
                   beyond_10px=median("beyond_10px"), mask_half_width=median("mask_half_width"))
        print(f"{label:24s} n={out['n']:4d}  missing {out['missing']:6.0f} px  "
              f"axial {out['axial_fraction']:.3f} ({out['axial_pixels']:.0f} px)  "
              f"dist med {out['median_distance']:5.1f} p90 {out['p90_distance']:6.1f}  "
              f"within3px {out['within_3px']:.3f}  beyond10px {out['beyond_10px']:.3f}  "
              f"half-width {out['mask_half_width']:.1f}")
        return out

    print("missing-area geometry, strict 1:1 pairs\n")
    report = dict(split=split.summary(),
                  near=summarise(near, "near threshold 0.40-0.50"),
                  healthy=summarise(healthy, "matched >= 0.50"))
    (args.output / "report.json").write_text(json.dumps(report, indent=2))
    if near and healthy:
        a, b = report["near"], report["healthy"]
        print(f"\naxial fraction: near {a['axial_fraction']:.3f} vs healthy {b['axial_fraction']:.3f}")
        print(f"verdict: {'TRUNCATION dominates' if a['axial_fraction'] >= 0.5 else 'THINNESS dominates'}"
              f" for the near-threshold population")
        print(f"missing area within 3 px of the mask: near {a['within_3px']:.1%}, "
              f"healthy {b['within_3px']:.1%}")


if __name__ == "__main__":
    main()
