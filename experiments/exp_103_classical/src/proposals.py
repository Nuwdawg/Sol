"""A non-ML filament proposal stream, and the measurement that justifies it.

Why this exists
---------------
Every model measured here draws good masks and misses filaments. exp098: the
2048-trained YOLOv8l-seg reaches SQ 0.6899 against the old model's 0.6843 while
finding 819 filaments against 860, and its best D4 operating point still leaves
506 ground-truth instances unmatched. Three test photographs receive no
prediction at all. A second detector trained the same way inherits the same blind
spot, because it inherits the same confidence floor and the same training signal.

So the question this module answers is not "can classical vision beat a network".
It is narrower and decisive for the pipeline: **of the filaments the detector
misses, how many are visible to a physics-based contrast method at all?** If the
answer is a large fraction, a proposal stream plus a verifier is a real route to
recall. If it is a small fraction, those filaments are not recoverable from
contrast, and the pipeline needs a different first stage. Either way the number
settles it, which is what CLAUDE.md section 6 demands of an idea like this.

The physics
-----------
A GONG H-alpha image is the solar disk seen in a chromospheric absorption line.
Filaments are cool, dense plasma suspended along magnetic polarity inversion
lines; they absorb the line and therefore read *darker* than the quiet disk. Two
properties of the instrument and the sun decide how to threshold them:

- **Limb darkening.** The disk is systematically dimmer toward its edge, because
  the line of sight there leaves the photosphere at a shallower angle. A single
  global threshold would find "filaments" all around the limb. The radial median
  profile is divided out first, so intensity means the same thing everywhere.
- **Faded boundaries.** Filament edges fade over several pixels, which is why
  annotators disagree about them (exp_005 measured masks systematically too
  large). A single threshold therefore either splits a filament into fragments or
  swallows its surroundings. Hysteresis handles this properly: a strict threshold
  marks seeds, a permissive one marks the region they may grow into, and only
  connected regions containing a seed survive.

What is deliberately not done
-----------------------------
No attempt is made to separate touching filaments, to place a final boundary, or
to score a proposal's quality. Those are later stages' jobs. This stage is
optimised for recall at a tolerable proposal count, and it is measured that way.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from pycocotools import mask as mu

NATIVE = 2048


def disk_mask(image: np.ndarray, floor: int = 25) -> tuple[np.ndarray, tuple[float, float, float]]:
    """The solar disk, as a mask plus (centre x, centre y, radius) in pixels."""
    rough = (image > floor).astype(np.uint8)
    rough = cv2.morphologyEx(rough, cv2.MORPH_CLOSE,
                             cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (31, 31)))
    contours, _ = cv2.findContours(rough, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        raise SystemExit("no solar disk found")
    (x, y), radius = cv2.minEnclosingCircle(max(contours, key=cv2.contourArea))
    mask = np.zeros(image.shape, np.uint8)
    cv2.circle(mask, (int(round(x)), int(round(y))), int(round(radius)), 1, -1)
    return mask, (float(x), float(y), float(radius))


def flatten_limb(image: np.ndarray, mask: np.ndarray, centre: tuple[float, float, float],
                 bins: int = 256, inset: float = 0.985) -> np.ndarray:
    """Divide out limb darkening using the disk's own radial median profile.

    The profile is the sun's, not a model's: for each radius bin the median of the
    pixels at that radius is the quiet-disk level there. Dividing by it leaves a
    flat field in which a filament is dark everywhere on the disk, including near
    the edge.
    """
    x0, y0, radius = centre
    ys, xs = np.nonzero(mask)
    values = image[ys, xs].astype(np.float32)
    distance = np.sqrt((xs - x0) ** 2 + (ys - y0) ** 2) / max(radius, 1.0)
    index = np.clip((distance * bins).astype(np.int32), 0, bins - 1)
    profile = np.ones(bins, np.float32)
    total = np.bincount(index, minlength=bins)
    for position in range(bins):
        if total[position] >= 32:
            profile[position] = float(np.median(values[index == position]))
    # Radii with too few pixels, and the outermost ring, copy their inward neighbour.
    for position in range(1, bins):
        if total[position] < 32 or profile[position] <= 1e-3:
            profile[position] = profile[position - 1]
    profile = cv2.GaussianBlur(profile.reshape(-1, 1), (1, 9), 0).reshape(-1)
    flat = np.zeros(image.shape, np.float32)
    flat[ys, xs] = values / np.maximum(profile[index], 1e-3)
    inner = np.zeros(image.shape, np.uint8)
    cv2.circle(inner, (int(round(x0)), int(round(y0))), int(round(radius * inset)), 1, -1)
    return flat * inner


def local_contrast(flat: np.ndarray, scales: tuple[int, ...] = (31, 101)) -> np.ndarray:
    """How much darker each pixel is than its surroundings, over several scales.

    A filament is a local depression, not an absolute intensity, so the background
    is estimated by heavy blurring and subtracted. Taking the maximum over scales
    keeps both narrow barbs and wide spines.
    """
    valid = (flat > 0).astype(np.float32)
    best = np.zeros(flat.shape, np.float32)
    for scale in scales:
        blurred = cv2.GaussianBlur(flat, (0, 0), scale / 3.0)
        weight = cv2.GaussianBlur(valid, (0, 0), scale / 3.0)
        background = blurred / np.maximum(weight, 1e-3)
        best = np.maximum(best, background - flat)
    return best * (flat > 0)


def hysteresis(contrast: np.ndarray, high: float, low: float, disk: np.ndarray) -> np.ndarray:
    """Seeds at `high`, grown through anything above `low` that touches them."""
    strong = ((contrast >= high) & (disk > 0)).astype(np.uint8)
    weak = ((contrast >= low) & (disk > 0)).astype(np.uint8)
    count, labels = cv2.connectedComponents(weak, connectivity=8)
    if count <= 1:
        return np.zeros_like(weak)
    keep = np.zeros(count, bool)
    keep[np.unique(labels[strong > 0])] = True
    keep[0] = False
    return keep[labels].astype(np.uint8)


def elongation(component: np.ndarray) -> float:
    """Length over width of the minimum-area rectangle; filaments are thin and long."""
    contours, _ = cv2.findContours(component, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return 0.0
    (_, _), (width, height), _ = cv2.minAreaRect(max(contours, key=cv2.contourArea))
    short, long = sorted((max(width, 1.0), max(height, 1.0)))
    return long / short


def propose(image: np.ndarray, high: float, low: float, min_area: int,
            min_elongation: float, close: int = 5) -> list[dict]:
    """Candidate filament masks for one photograph, as RLE dictionaries."""
    disk, centre = disk_mask(image)
    flat = flatten_limb(image, disk, centre)
    contrast = local_contrast(flat)
    grown = hysteresis(contrast, high, low, disk)
    if close:
        grown = cv2.morphologyEx(grown, cv2.MORPH_CLOSE,
                                 cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close, close)))
    count, labels, stats, _ = cv2.connectedComponentsWithStats(grown, connectivity=8)
    proposals = []
    for label in range(1, count):
        area = int(stats[label, cv2.CC_STAT_AREA])
        if area < min_area:
            continue
        component = (labels == label).astype(np.uint8)
        shape = elongation(component)
        if shape < min_elongation:
            continue
        rle = mu.encode(np.asfortranarray(component))
        proposals.append(dict(area=area, elongation=round(float(shape), 2),
                              mean_contrast=round(float(contrast[component > 0].mean()), 4),
                              rle=dict(size=rle["size"], counts=rle["counts"].decode("ascii"))))
    return proposals


def coverage(proposals: list[dict], truths: list[dict], thresholds=(0.5, 0.3, 0.1)) -> dict:
    """For each ground-truth instance, the best IoU any proposal reaches."""
    if not truths:
        return dict(truths=0, best=[], covered={f"{t}": 0 for t in thresholds})
    if not proposals:
        return dict(truths=len(truths), best=[0.0] * len(truths),
                    covered={f"{t}": 0 for t in thresholds})
    ious = np.asarray(mu.iou([p["rle"] for p in proposals], truths, [0] * len(truths)))
    best = ious.max(axis=0) if ious.size else np.zeros(len(truths))
    return dict(truths=len(truths), proposals=len(proposals),
                best=[round(float(value), 4) for value in best],
                covered={f"{t}": int((best >= t).sum()) for t in thresholds})


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--images", type=Path, required=True)
    parser.add_argument("--photographs", nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--high", type=float, default=0.035)
    parser.add_argument("--low", type=float, default=0.015)
    parser.add_argument("--min-area", type=int, default=300)
    parser.add_argument("--min-elongation", type=float, default=1.8)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    report = {}
    for name in args.photographs:
        image = cv2.imread(str(args.images / name), cv2.IMREAD_GRAYSCALE)
        if image is None or image.shape != (NATIVE, NATIVE):
            raise SystemExit(f"cannot read {name}")
        proposals = propose(image, args.high, args.low, args.min_area, args.min_elongation)
        report[name] = dict(count=len(proposals),
                            areas=[p["area"] for p in proposals],
                            elongations=[p["elongation"] for p in proposals])
        (args.output / f"{Path(name).stem}_proposals.json").write_text(json.dumps(proposals))
        print(f"{name}: {len(proposals)} proposals", flush=True)
    (args.output / "summary.json").write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
