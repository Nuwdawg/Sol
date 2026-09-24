"""Does a *path* exist ahead of a truncated endpoint, even if the cone is mostly empty?

exp109 measured the mean learned-foreground probability in a 35-degree cone ahead
of each endpoint and found 0.0005. That rejected endpoint continuation, but the
statistic was wrong for the question: a geodesic follows the cheapest path, not the
average. A thin high-probability thread inside a mostly empty cone is exactly the
thing a shortest-path method exists to find, and a mean would hide it.

This audit asks the question a geodesic would ask. From each endpoint, cast rays at
one-degree steps across the cone. Along each ray, from `near` to `far` pixels,
sample the learned foreground field, the spine field and the raw image. Then report,
per instance:

- `best_ray_mean`: the highest mean support of any single ray - the closest scalar
  analogue of "there is a supported path in this direction";
- `best_ray_min`: for that ray, the weakest pixel along it, which is what actually
  limits a path: one zero-probability gap breaks a continuation;
- `cone_max`: the single highest-probability pixel anywhere in the cone, an upper
  bound on any evidence at all;
- `dark_best_ray`: the same best-ray statistic on inverted image intensity, since
  filaments absorb - this tests the image directly rather than the learned field,
  in case the field simply never learned these faint extensions.

Ground truth appears only as `truth_best_ray`, a label verifying the populations,
never as a feature. Both directions of every feature are scored with precision at
operating points, per the 2026-09-16 correction.
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
    ys, xs = np.nonzero(mask)
    points = np.stack([xs, ys], 1).astype(np.float32)
    centre = points.mean(0)
    _, _, axes = np.linalg.svd(points - centre, full_matrices=False)
    along = (points - centre) @ axes[0]
    return [(points[np.argmax(along * sign)], axes[0] * sign) for sign in (+1, -1)]


def ray_samples(field, tip, direction, near, far, cone_degrees, step=1.0):
    """Mean, minimum and maximum along every ray in the cone; returns per-ray arrays."""
    base = np.arctan2(direction[1], direction[0])
    angles = np.radians(np.arange(-cone_degrees, cone_degrees + 1e-6, 1.0)) + base
    distances = np.arange(near, far + 1e-6, step)
    height, width = field.shape
    means, minima, maxima = [], [], []
    for angle in angles:
        xs = np.clip((tip[0] + np.cos(angle) * distances).astype(np.int32), 0, width - 1)
        ys = np.clip((tip[1] + np.sin(angle) * distances).astype(np.int32), 0, height - 1)
        values = field[ys, xs]
        means.append(float(values.mean()))
        minima.append(float(values.min()))
        maxima.append(float(values.max()))
    return np.array(means), np.array(minima), np.array(maxima)


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
    parser.add_argument("--output", type=Path, default=Path("data/exp110_best_path"))
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
        raw = cv2.imread(str(IMAGES / photograph), cv2.IMREAD_GRAYSCALE)
        if raw is None:
            continue
        # Filaments absorb: darkness relative to a large-scale background is the
        # physical signal, independent of anything the network learned.
        background = cv2.GaussianBlur(raw.astype(np.float32), (0, 0), 25.0)
        dark = np.clip((background - raw.astype(np.float32)) / 40.0, 0, 1)

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
                truth = mu.decode(truths[g]).astype(np.float32)
                best = None
                for tip, direction in endpoints(decoded[p]):
                    sem_mean, sem_min, sem_max = ray_samples(
                        sem, tip, direction, args.near, args.far, args.cone)
                    spine_mean, _, _ = ray_samples(
                        spine, tip, direction, args.near, args.far, args.cone)
                    dark_mean, _, _ = ray_samples(
                        dark, tip, direction, args.near, args.far, args.cone)
                    truth_mean, _, _ = ray_samples(
                        truth, tip, direction, args.near, args.far, args.cone)
                    pick = int(np.argmax(sem_mean))
                    row = dict(best_ray_mean=float(sem_mean.max()),
                               best_ray_min=float(sem_min[pick]),
                               cone_max=float(sem_max.max()),
                               spine_best_ray=float(spine_mean.max()),
                               dark_best_ray=float(dark_mean.max()),
                               truth_best_ray=float(truth_mean.max()))
                    if best is None or row["truth_best_ray"] > best["truth_best_ray"]:
                        best = row
                if best is None:
                    continue
                best["iou"] = iou
                (near_rows if iou < 0.5 else healthy_rows).append(best)

    print(f"rays at 1 degree over +/-{args.cone} deg, {args.near}-{args.far} px ahead")
    print(f"near-threshold {len(near_rows)}   healthy {len(healthy_rows)}\n")
    print(f"{'feature':16s} {'AUC':>6} {'AUCneg':>7} {'near med':>10} {'healthy med':>12}"
          f"  best over both directions")
    report = dict(split=split.summary(), near=len(near_rows), healthy=len(healthy_rows),
                  cone=dict(near=args.near, far=args.far, half_angle=args.cone), features={})
    for key in ("truth_best_ray", "best_ray_mean", "best_ray_min", "cone_max",
                "spine_best_ray", "dark_best_ray"):
        positive = np.array([r[key] for r in near_rows], float)
        negative = np.array([r[key] for r in healthy_rows], float)
        if not positive.size or not negative.size:
            continue
        area = rank_auc(positive, negative)
        best = best_operating_point(positive, negative)
        report["features"][key] = dict(auc=round(area, 4), best=best)
        note = "   <- LABEL" if key == "truth_best_ray" else ""
        print(f"{key:16s} {area:6.3f} {1 - area:7.3f} {np.median(positive):10.4f} "
              f"{np.median(negative):12.4f}  {best['side']} q{best['quantile']:.2f} "
              f"net {best['net']:+d} precision {best['precision']:.3f}{note}")
    (args.output / "report.json").write_text(json.dumps(report, indent=2))
    print(f"\nwrote {args.output / 'report.json'}")


if __name__ == "__main__":
    main()
