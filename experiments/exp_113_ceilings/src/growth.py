"""Oracle ceilings for the two growth repairs: endpoint continuation and lateral boundary.

exp108 decomposed the missing area of near-threshold 1:1 masks into roughly 59%
axial (beyond the mask's ends) and 41% lateral (alongside its body). Those two
populations are addressed by different method families in the testing manual -
endpoint geodesics and elastica for the first, active contours and graph cuts for
the second - and neither family has had its ceiling measured.

This module supplies both ceilings, using truth to choose what to add, which is
what makes them upper bounds rather than postprocessors:

- `apply_axial_ceiling`: add only the missing truth pixels that lie beyond the
  mask's own extent along its principal axis, and only within a corridor of
  `reach` pixels of the mask. This is the most a perfect endpoint continuation
  could recover, including perfect width reconstruction.
- `apply_lateral_ceiling`: add only the missing truth pixels within `band` pixels
  of the existing boundary, which is the most a perfect narrow-band boundary
  method could recover.

Both are deliberately generous to the method: a real operator must also decide
*whether* to act and *where* to stop, without truth. If a ceiling is small, the
manual's step 1 stops that family regardless of how good a trigger might be.
"""
from __future__ import annotations

import cv2
import numpy as np
from pycocotools import mask as mu

from experiments.exp_033_inference.src.sweep import encode

TOUCH = 0.05


def _axial_split(mask: np.ndarray, missing: np.ndarray):
    """Missing pixels beyond the mask's extent along its principal axis."""
    ys, xs = np.nonzero(mask)
    if len(xs) < 20:
        return np.zeros_like(missing, dtype=bool)
    points = np.stack([xs, ys], 1).astype(np.float32)
    centre = points.mean(0)
    _, _, axes = np.linalg.svd(points - centre, full_matrices=False)
    along = (points - centre) @ axes[0]
    low, high = np.percentile(along, 1), np.percentile(along, 99)
    my, mx = np.nonzero(missing)
    if not len(mx):
        return np.zeros_like(missing, dtype=bool)
    projected = (np.stack([mx, my], 1).astype(np.float32) - centre) @ axes[0]
    beyond = (projected < low) | (projected > high)
    out = np.zeros_like(missing, dtype=bool)
    out[my[beyond], mx[beyond]] = True
    return out


def _within(mask: np.ndarray, radius: int) -> np.ndarray:
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1))
    return cv2.dilate(mask, kernel).astype(bool)


def _grow_population(population, selector, band_or_reach: int, only_near: bool):
    """Shared driver: add truth pixels chosen by `selector` to each 1:1 prediction."""
    repaired, modified, added = [], 0, 0
    for record in population:
        predicted, truths = record["predicted"], record["truths"]
        if not predicted or not truths:
            repaired.append(dict(record, predicted=list(predicted)))
            continue
        ious = np.asarray(mu.iou(predicted, truths, [0] * len(truths)))
        output = list(predicted)
        for p in range(len(predicted)):
            covered = [g for g in range(len(truths)) if ious[p, g] > TOUCH]
            if len(covered) != 1:
                continue
            g = covered[0]
            if len([q for q in range(len(predicted)) if ious[q, g] > TOUCH]) != 1:
                continue
            iou = float(ious[p, g])
            if only_near and not (0.4 <= iou < 0.5):
                continue
            mask = mu.decode(predicted[p])
            truth = mu.decode(truths[g])
            missing = (truth.astype(bool) & ~mask.astype(bool))
            if missing.sum() < 20:
                continue
            chosen = selector(mask, missing, band_or_reach)
            if chosen.sum() < 20:
                continue
            grown = (mask.astype(bool) | chosen).astype(np.uint8)
            output[p] = encode(grown)
            modified += 1
            added += int(chosen.sum())
        repaired.append(dict(record, predicted=output))
    return repaired, dict(masks_modified=modified, pixels_added=added)


def _axial_selector(mask, missing, reach):
    return _axial_split(mask, missing) & _within(mask, reach)


def _lateral_selector(mask, missing, band):
    axial = _axial_split(mask, missing)
    return missing & ~axial & _within(mask, band)


def apply_axial_ceiling(population, reach: int = 30, only_near: bool = True):
    """ORACLE: perfect endpoint continuation within `reach` pixels."""
    return _grow_population(population, _axial_selector, reach, only_near)


def apply_lateral_ceiling(population, band: int = 4, only_near: bool = True):
    """ORACLE: perfect narrow-band boundary correction within `band` pixels."""
    return _grow_population(population, _lateral_selector, band, only_near)


def apply_full_growth_ceiling(population, only_near: bool = True):
    """ORACLE: add every missing truth pixel of near-threshold 1:1 masks.

    The absolute ceiling for any repair that only grows existing 1:1 instances,
    against which the axial and lateral ceilings can be read as fractions.
    """
    return _grow_population(population, lambda mask, missing, _: missing, 0, only_near)
