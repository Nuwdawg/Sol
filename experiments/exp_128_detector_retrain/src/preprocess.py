"""Reche & Cid's preprocessing for MAGFiLO, which our detectors have never seen.

Every model trained in this repository reads the raw competition JPEG. The
published method for this exact dataset (Reche & Cid 2026, Acta Astronautica)
normalises first, and reports per-instance IoU 0.70 / Dice 0.81. Our matched SQ is
0.684, so their advantage is not boundary quality - it is recognition, and
normalisation is the part of their pipeline that plausibly drives it.

Three steps, in their order:

1. **Radial limb-darkening correction.** The disk is systematically dimmer toward
   its edge because the line of sight leaves the photosphere at a shallower angle.
   The correction divides by the disk's own radial median profile, so a filament
   reads equally dark at centre and near the limb. Taken from exp103, which already
   measured this profile correctly.
2. **Histogram matching to a fixed reference.** Exposure, seeing and instrument
   state move the whole intensity distribution between observations. Matching every
   disk interior to one reference distribution makes a given grey level mean the
   same thing across the archive. The reference is built once from the training
   photographs and frozen, so validation and test see the identical transform.
3. **Off-disk zeroed.** Sky pixels carry no filament information and their gradient
   at the limb is the strongest edge in the frame.

Why this is worth a retrain rather than a post-process: all three change what the
*detector* sees. exp104 and exp117 showed that transforming masks after the fact
cannot recover recall, and recall is where the 465 misses are.

The reference is a CDF, saved as JSON, and applied by lookup. Nothing here reads
ground truth.
"""
from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

NATIVE = 2048
LEVELS = 256


def disk_geometry(image: np.ndarray, floor: int = 25) -> tuple[np.ndarray, tuple[float, float, float]]:
    """The solar disk as a mask plus (centre x, centre y, radius) in pixels."""
    rough = (image > floor).astype(np.uint8)
    rough = cv2.morphologyEx(rough, cv2.MORPH_CLOSE,
                             cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (31, 31)))
    contours, _ = cv2.findContours(rough, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        raise ValueError("no solar disk found")
    (x, y), radius = cv2.minEnclosingCircle(max(contours, key=cv2.contourArea))
    mask = np.zeros(image.shape, np.uint8)
    cv2.circle(mask, (int(round(x)), int(round(y))), int(round(radius)), 1, -1)
    return mask, (float(x), float(y), float(radius))


def flatten_limb(image: np.ndarray, mask: np.ndarray, centre: tuple[float, float, float],
                 bins: int = 256, inset: float = 0.985) -> np.ndarray:
    """Divide out limb darkening using the disk's own radial median profile.

    The profile is the sun's, not a model's: for each radius bin the median of the
    pixels at that radius is the quiet-disk level there.
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
    for position in range(1, bins):
        if total[position] < 32 or profile[position] <= 1e-3:
            profile[position] = profile[position - 1]
    profile = cv2.GaussianBlur(profile.reshape(-1, 1), (1, 9), 0).reshape(-1)
    flat = np.zeros(image.shape, np.float32)
    flat[ys, xs] = values / np.maximum(profile[index], 1e-3)
    inner = np.zeros(image.shape, np.uint8)
    cv2.circle(inner, (int(round(x0)), int(round(y0))), int(round(radius * inset)), 1, -1)
    return flat * inner, inner


def to_levels(flat: np.ndarray, inner: np.ndarray, low: float = 0.55,
              high: float = 1.25) -> np.ndarray:
    """Map the flattened disk onto 0-255, clipping the tails that limb noise creates."""
    scaled = np.zeros(flat.shape, np.uint8)
    inside = inner > 0
    if not inside.any():
        return scaled
    values = np.clip((flat[inside] - low) / max(high - low, 1e-6), 0, 1)
    scaled[inside] = np.clip(values * (LEVELS - 1), 0, LEVELS - 1).astype(np.uint8)
    return scaled


def histogram_of(levels: np.ndarray, inner: np.ndarray) -> np.ndarray:
    counts = np.bincount(levels[inner > 0].ravel(), minlength=LEVELS).astype(np.float64)
    total = counts.sum()
    return counts / total if total > 0 else counts


def build_reference(histograms: list[np.ndarray]) -> np.ndarray:
    """The frozen reference CDF: the mean disk histogram over the training set.

    Averaging distributions rather than images keeps the three-peak structure of a
    typical H-alpha disk - quiet sun, network, and the dark filament tail - which is
    what makes a grey level comparable between observations.
    """
    mean = np.mean(np.stack(histograms), axis=0)
    total = mean.sum()
    return np.cumsum(mean / total) if total > 0 else np.cumsum(mean)


def match_to(levels: np.ndarray, inner: np.ndarray, reference_cdf: np.ndarray) -> np.ndarray:
    """Histogram-match the disk interior onto the reference CDF by lookup."""
    source = np.cumsum(histogram_of(levels, inner))
    lookup = np.searchsorted(reference_cdf, source).clip(0, LEVELS - 1).astype(np.uint8)
    matched = np.zeros(levels.shape, np.uint8)
    inside = inner > 0
    matched[inside] = lookup[levels[inside]]
    return matched


def prepare(image: np.ndarray):
    """Raw grayscale in; flattened levels and the inner-disk mask out."""
    mask, centre = disk_geometry(image)
    flat, inner = flatten_limb(image, mask, centre)
    return to_levels(flat, inner), inner


def preprocess(image: np.ndarray, reference_cdf: np.ndarray | None) -> np.ndarray:
    """The full transform: limb-flattened, histogram-matched, off-disk zeroed."""
    levels, inner = prepare(image)
    if reference_cdf is not None:
        levels = match_to(levels, inner, reference_cdf)
    return levels * (inner > 0)


def save_reference(cdf: np.ndarray, path: Path) -> None:
    path.write_text(json.dumps(dict(levels=LEVELS, cdf=[float(v) for v in cdf])))


def load_reference(path: Path) -> np.ndarray:
    payload = json.loads(Path(path).read_text())
    if payload["levels"] != LEVELS:
        raise ValueError(f"reference built for {payload['levels']} levels, not {LEVELS}")
    return np.asarray(payload["cdf"], dtype=np.float64)
