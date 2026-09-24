"""Growth operators: the first implementable attempts at the lateral and axial repairs.

exp113 measured the ceilings - lateral +0.018865, axial +0.018103 - and no operator
has been written for either. Splitting has now failed at placement across three
geometries (exp114-116), so growth is the remaining family with real headroom and
no attempt against it.

Two operators, both reading only inference-time inputs:

- `dark_band_grow`: the lateral repair. Widen the mask into the adjacent band, but
  only where the image is dark, so growth follows absorption instead of expanding
  uniformly. exp104 measured that uniform dilation costs 0.027 PQ, which is the
  control this must beat.
- `dark_ray_extend`: the axial repair. From each principal-axis endpoint, extend
  along the best-supported ray while absorption holds up, and rebuild width from
  the mask's own half-width. exp110 measured raw absorption ahead of truncations at
  AUC 0.605-0.611, the only image-derived signal above chance there.

**Thresholds are relative, not absolute.** A fixed absorption floor is not
comparable across photographs: exposure, limb position and seeing all move the raw
contrast. Both operators therefore threshold against the *mask's own interior
absorption*, so the test is "is this pixel nearly as dark as the filament I already
have". That is self-calibrating per mask and per photograph, and it needs no tuning
constant in image units.

Neither operator reads ground truth. Only the gate does, and only in the oracle
driver, which is never deployable.
"""
from __future__ import annotations

import cv2
import numpy as np
from pycocotools import mask as mu

from experiments.exp_033_inference.src.sweep import encode

TOUCH = 0.05


def dark_field(image: np.ndarray, sigma: float = 25.0) -> np.ndarray:
    """Absorption against a local background: positive where the image is darker.

    A large-sigma blur is a local quiet-disk estimate, so this also removes limb
    darkening without needing the radial profile fit that exp103 used. Units are
    raw intensity counts, and callers must compare them to a per-mask reference
    rather than to a constant.
    """
    floated = image.astype(np.float32)
    return cv2.GaussianBlur(floated, (0, 0), sigma) - floated


def _reference(mask: np.ndarray, dark: np.ndarray) -> float:
    """The mask's own interior absorption: the scale every threshold is relative to."""
    inside = dark[mask > 0]
    return float(np.median(inside)) if inside.size else 0.0


def dark_band_grow(mask: np.ndarray, dark: np.ndarray, band: int = 3,
                   relative: float = 0.55, max_growth: float = 0.5) -> np.ndarray:
    """Lateral repair: add adjacent pixels at least `relative` times as dark as the mask.

    `max_growth` caps added area as a fraction of the original, so a mask lying on a
    broad dark region cannot bloat without limit.
    """
    reference = _reference(mask, dark)
    if reference <= 0:
        return mask
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * band + 1, 2 * band + 1))
    ring = (cv2.dilate(mask, kernel) > 0) & (mask == 0)
    candidate = ring & (dark >= relative * reference)
    if not candidate.any():
        return mask
    limit = int(max_growth * int(mask.sum()))
    if limit > 0 and int(candidate.sum()) > limit:
        values = dark[candidate]
        threshold = float(np.partition(values, -limit)[-limit])
        candidate = candidate & (dark >= threshold)
    return ((mask > 0) | candidate).astype(np.uint8)


def _endpoints(mask: np.ndarray):
    """The two extreme points along the principal axis, with the outward direction."""
    ys, xs = np.nonzero(mask)
    points = np.stack([xs, ys], 1).astype(np.float32)
    centre = points.mean(0)
    _, _, axes = np.linalg.svd(points - centre, full_matrices=False)
    along = (points - centre) @ axes[0]
    return [(points[int(np.argmax(along * sign))], axes[0] * sign) for sign in (+1, -1)]


def dark_ray_extend(mask: np.ndarray, dark: np.ndarray, reach: int = 24,
                    cone: float = 40.0, relative: float = 0.5, step: float = 1.0,
                    probe: int = 6) -> np.ndarray:
    """Axial repair: extend each endpoint along its best-supported ray.

    Direction is chosen by mean absorption over the first `probe` pixels only, not
    over the whole reach. Averaging over the full reach conflates two decisions and
    makes the operator decline almost everywhere: a ray that correctly leaves the
    filament end scores below the floor because most of its length is background,
    so a correct direction is rejected for being long. The sanity check caught this
    as 0 pixels added on 3 of 6 masks.

    The walk then stops at the first sample below the relative floor, so the
    extension ends where the evidence ends rather than at a fixed length. Width
    comes from the mask's own median half-width.
    """
    if int(mask.sum()) < 50:
        return mask
    reference = _reference(mask, dark)
    if reference <= 0:
        return mask
    floor = relative * reference
    distance = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
    half_width = max(float(np.median(distance[mask > 0])), 1.0)
    grown = mask.copy()
    height, width = mask.shape
    distances = np.arange(1.0, reach + 1e-6, step)
    probed = distances[:max(probe, 1)]
    for tip, direction in _endpoints(mask):
        base = np.arctan2(float(direction[1]), float(direction[0]))
        best_angle, best_support = None, floor
        for angle in np.radians(np.arange(-cone, cone + 1e-6, 5.0)) + base:
            xs = np.clip((tip[0] + np.cos(angle) * probed).astype(np.int32), 0, width - 1)
            ys = np.clip((tip[1] + np.sin(angle) * probed).astype(np.int32), 0, height - 1)
            support = float(dark[ys, xs].mean())
            if support > best_support:
                best_angle, best_support = angle, support
        if best_angle is None:
            continue
        for offset in distances:
            x = int(round(float(tip[0]) + np.cos(best_angle) * offset))
            y = int(round(float(tip[1]) + np.sin(best_angle) * offset))
            if not (0 <= x < width and 0 <= y < height) or dark[y, x] < floor:
                break
            cv2.circle(grown, (x, y), int(round(half_width)), 1, -1)
    return grown


def dark_grow_then_extend(mask: np.ndarray, dark: np.ndarray, band: int = 3,
                          relative: float = 0.55, reach: int = 24, cone: float = 40.0,
                          ray_relative: float = 0.5) -> np.ndarray:
    """Both repairs in sequence, since exp108 split the deficit 59% axial, 41% lateral."""
    grown = dark_band_grow(mask, dark, band=band, relative=relative)
    return dark_ray_extend(grown, dark, reach=reach, cone=cone, relative=ray_relative)


def uniform_dilate(mask: np.ndarray, dark: np.ndarray, iterations: int = 1) -> np.ndarray:
    """Matched control: grow the same masks without reading the image at all.

    Any credit claimed for absorption-guided growth must be credit *over this*.
    exp104 measured uniform dilation applied to every prediction at -0.027 PQ, but
    that is not the comparison being made here: under the same oracle gate the naive
    operator may well be positive, and a guided operator has to beat it to have
    earned anything.
    """
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    return cv2.dilate(mask, kernel, iterations=iterations)


def apply_growth_operator(population, load_image, operator, oracle_gate: bool = True,
                          near_band=(0.4, 0.5), **kwargs):
    """Run a growth operator, gated by truth (analysis) or on every strict 1:1 mask.

    `load_image` maps a photograph name to its grayscale array. Absorption fields are
    held one at a time: `build_population` emits every record of a photograph
    consecutively, so a single-entry cache gives the same hit rate as keeping all 106
    at 16 MB instead of 1.7 GB.
    """
    repaired, attempted, changed, added = [], 0, 0, 0
    cached_name, dark = None, None
    for record in population:
        predicted, truths = record["predicted"], record["truths"]
        photograph = record["photograph"]
        if not predicted or not truths:
            repaired.append(dict(record, predicted=list(predicted)))
            continue
        ious = np.asarray(mu.iou(predicted, truths, [0] * len(truths)))
        candidates = []
        for p in range(len(predicted)):
            covered = [g for g in range(len(truths)) if ious[p, g] > TOUCH]
            if len(covered) != 1:
                continue
            g = covered[0]
            if len([q for q in range(len(predicted)) if ious[q, g] > TOUCH]) != 1:
                continue
            if oracle_gate and not (near_band[0] <= float(ious[p, g]) < near_band[1]):
                continue
            candidates.append(p)
        if not candidates:
            repaired.append(dict(record, predicted=list(predicted)))
            continue
        # Decide the gate before paying for the image. A 2048x2048 read plus a
        # sigma-25 blur costs far more than the RLE overlaps above, and under the
        # oracle gate most photographs hold nothing to repair at all.
        if photograph != cached_name:
            image = load_image(photograph)
            cached_name = photograph
            dark = None if image is None else dark_field(image)
        if dark is None:
            repaired.append(dict(record, predicted=list(predicted)))
            continue
        output = list(predicted)
        for p in candidates:
            attempted += 1
            mask = mu.decode(predicted[p])
            grown = operator(mask, dark, **kwargs)
            extra = int(grown.sum()) - int(mask.sum())
            if extra <= 0:
                continue
            output[p] = encode(grown)
            changed += 1
            added += extra
        repaired.append(dict(record, predicted=output))
    return repaired, dict(masks_attempted=attempted, masks_changed=changed,
                          pixels_added=added, oracle_gate=oracle_gate)
