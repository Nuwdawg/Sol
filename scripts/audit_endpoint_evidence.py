"""Is there evidence ahead of a truncated mask's endpoint, where the defect actually is?

exp108 measured that near-threshold 1:1 masks are cut short: 59% of the missing
truth area lies beyond the mask's own extent, a median 10.9 px away with a p90 of
30.7 px, on bodies whose median half-width is 2.2 px. Seven trigger families have
now been rejected, and every one of them sampled a 4 px ring around the boundary -
the wrong neighbourhood entirely.

This audit samples where the defect is. For each mask, take the two extreme points
along its principal axis. At each endpoint, sweep a cone of half-angle `cone`
degrees around the outward axis direction, between `near` and `far` pixels ahead,
and measure inside it:

- `sem`: mean exp085 mask probability, the learned foreground field;
- `spine`: mean exp085 centreline probability;
- `orientation`: mean absolute cosine between the image structure tensor's ridge
  direction and the mask's own axis, which asks whether the image texture ahead
  continues in the same direction rather than merely being bright;
- `truth_ahead`: the fraction of the cone the ground truth occupies. This is a
  LABEL, not a feature. It verifies the populations are what they claim to be -
  truncated masks must have more truth ahead than healthy ones. If it fails to
  separate them, the audit is broken and no other number in it means anything.

The question is falsifiable: do truncated masks show more forward evidence than
healthy ones, in a place a geodesic could walk? If not, endpoint continuation has
nothing to follow and the deterministic route to these objects is closed.

Both orientations of every feature are scored, after the 2026-09-16 correction
that a one-sided search rejects features which rank the classes backwards.
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
FIELDS = Path("data/exp085_output/validation/probabilities")
IMAGES = Path("data/competition/unpacked/MAGFiLO_1.0_Kaggle_2026/train/train_images")


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


def endpoints(mask):
    """The two extreme points along the mask's principal axis, with outward directions."""
    ys, xs = np.nonzero(mask)
    points = np.stack([xs, ys], 1).astype(np.float32)
    centre = points.mean(0)
    _, _, axes = np.linalg.svd(points - centre, full_matrices=False)
    along = (points - centre) @ axes[0]
    return [(points[np.argmax(along * sign)], axes[0] * sign) for sign in (+1, -1)]


def cone_mask(shape, tip, direction, near, far, cone_degrees):
    """Pixels between `near` and `far` ahead of `tip`, within `cone_degrees` of `direction`."""
    height, width = shape
    x0, x1 = int(max(tip[0] - far, 0)), int(min(tip[0] + far + 1, width))
    y0, y1 = int(max(tip[1] - far, 0)), int(min(tip[1] + far + 1, height))
    if x1 <= x0 or y1 <= y0:
        return None, None
    ys, xs = np.mgrid[y0:y1, x0:x1]
    dx, dy = xs - tip[0], ys - tip[1]
    distance = np.sqrt(dx * dx + dy * dy)
    with np.errstate(invalid="ignore", divide="ignore"):
        cosine = (dx * direction[0] + dy * direction[1]) / np.maximum(distance, 1e-6)
    inside = (distance >= near) & (distance <= far) & (cosine >= np.cos(np.radians(cone_degrees)))
    return inside, (slice(y0, y1), slice(x0, x1))


def orientation_agreement(image, window, inside, direction, sigma=3.0):
    """Mean absolute cosine between the structure tensor's ridge direction and the axis."""
    patch = image[window].astype(np.float32)
    gx = cv2.Sobel(patch, cv2.CV_32F, 1, 0, ksize=5)
    gy = cv2.Sobel(patch, cv2.CV_32F, 0, 1, ksize=5)
    jxx = cv2.GaussianBlur(gx * gx, (0, 0), sigma)
    jyy = cv2.GaussianBlur(gy * gy, (0, 0), sigma)
    jxy = cv2.GaussianBlur(gx * gy, (0, 0), sigma)
    angle = 0.5 * np.arctan2(2 * jxy, jxx - jyy)
    ridge = np.stack([-np.sin(angle), np.cos(angle)], -1)
    cosine = np.abs(ridge[..., 0] * direction[0] + ridge[..., 1] * direction[1])
    return float(cosine[inside].mean()) if inside.any() else None


def rank_auc(positive, negative):
    positive, negative = np.asarray(positive, float), np.asarray(negative, float)
    if not positive.size or not negative.size:
        return float("nan")
    ranks = np.concatenate([positive, negative]).argsort().argsort() + 1
    return float((ranks[:positive.size].sum() - positive.size * (positive.size + 1) / 2)
                 / (positive.size * negative.size))


def best_operating_point(positive, negative):
    """Best net objects over both directions and several quantile cuts."""
    best = None
    for direction in (+1, -1):
        for quantile in (0.70, 0.80, 0.90, 0.95):
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
    parser.add_argument("--output", type=Path, default=Path("data/exp109_endpoint_evidence"))
    parser.add_argument("--conf", type=float, default=0.35)
    parser.add_argument("--grow", type=int, default=-1)
    parser.add_argument("--min-area", type=int, default=100)
    parser.add_argument("--near", type=float, default=4.0)
    parser.add_argument("--far", type=float, default=30.0)
    parser.add_argument("--cone", type=float, default=35.0)
    args = parser.parse_args()
    cv2.setNumThreads(2)
    args.output.mkdir(parents=True, exist_ok=True)
    annotations = next(Path(".").rglob("MAGFiLO_1.0_Annotations_kaggle2026_train.json"))
    records, split = records_for(annotations)
    pool = json.loads(args.pool.read_text())

    near_rows, healthy_rows = [], []
    for photograph, passes in records.items():
        stem = Path(photograph).stem
        field_path = FIELDS / f"{stem}_mask.png"
        if not field_path.is_file():
            continue
        sem = cv2.imread(str(field_path), cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255.0
        spine = cv2.imread(str(FIELDS / f"{stem}_spine.png"),
                           cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255.0
        image = cv2.imread(str(IMAGES / photograph), cv2.IMREAD_GRAYSCALE)
        if image is None:
            continue
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
                truth = mu.decode(truths[g])
                best = None
                for tip, direction in endpoints(decoded[p]):
                    inside, window = cone_mask(decoded[p].shape, tip, direction,
                                               args.near, args.far, args.cone)
                    if inside is None or not inside.any():
                        continue
                    row = dict(sem=float(sem[window][inside].mean()),
                               spine=float(spine[window][inside].mean()),
                               orientation=orientation_agreement(image, window, inside, direction),
                               truth_ahead=float(truth[window][inside].mean()))
                    if best is None or row["truth_ahead"] > best["truth_ahead"]:
                        best = row
                if best is None:
                    continue
                best["iou"] = iou
                (near_rows if iou < 0.5 else healthy_rows).append(best)

    print(f"cone {args.near}-{args.far} px ahead, half-angle {args.cone} deg, "
          f"emission conf {args.conf} grow {args.grow} min_area {args.min_area}")
    print(f"near-threshold {len(near_rows)}   healthy {len(healthy_rows)}\n")
    header = f"{'feature':14s} {'AUC':>6} {'AUCneg':>7} {'near med':>10} {'healthy med':>12}"
    print(header + "  best over both directions")
    report = dict(split=split.summary(), near=len(near_rows), healthy=len(healthy_rows),
                  cone=dict(near=args.near, far=args.far, half_angle=args.cone),
                  emission=dict(conf=args.conf, grow=args.grow, min_area=args.min_area),
                  features={})
    for key in ("truth_ahead", "sem", "spine", "orientation"):
        positive = np.array([r[key] for r in near_rows if r.get(key) is not None], float)
        negative = np.array([r[key] for r in healthy_rows if r.get(key) is not None], float)
        if not positive.size or not negative.size:
            continue
        area = rank_auc(positive, negative)
        best = best_operating_point(positive, negative)
        report["features"][key] = dict(auc=round(area, 4), best=best)
        note = "   <- LABEL, verifies the populations" if key == "truth_ahead" else ""
        print(f"{key:14s} {area:6.3f} {1 - area:7.3f} {np.median(positive):10.4f} "
              f"{np.median(negative):12.4f}  {best['side']} q{best['quantile']:.2f} "
              f"net {best['net']:+d} precision {best['precision']:.3f}{note}")
    (args.output / "report.json").write_text(json.dumps(report, indent=2))
    print(f"\nwrote {args.output / 'report.json'}")


if __name__ == "__main__":
    main()
