"""Faithful split operators, tested under an oracle gate before any gate is built.

exp113 measured that perfect truth-assignment splitting of merged predictions is
worth +0.022342 PQ. That is a ceiling on the repair family, not evidence that any
implementable splitter reaches it. The manual's method 11 asks for the operator to
be run and rescored, and separates two questions this log had been conflating:

    gate:   should this mask be split?
    repair: if split, does the operator produce better instances?

The second is answerable now, with an oracle gate: apply the splitter only to the
masks that truly cover two or more truths, and measure how much of the ceiling a
real operator recovers. A weak result means the operator is weak; a strong result
means only the gate remains.

Two operators, both dependency-free:

- `watershed_split`: markers from persistent maxima of the distance transform,
  then a priority flood over the negated distance transform, restricted to the
  mask. This is marker-controlled watershed without requiring skimage.
- `neck_cut_split`: the manual's comparison baseline. Cut perpendicular to the
  principal axis at the narrowest interior cross-section.

Neither reads ground truth. Only the *gate* does, and only in the oracle driver.
"""
from __future__ import annotations

import heapq

import cv2
import numpy as np
from pycocotools import mask as mu

from experiments.exp_033_inference.src.sweep import encode

TOUCH = 0.05
NEIGHBOURS = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]


def distance_markers(mask: np.ndarray, min_separation: int = 6, floor_fraction: float = 0.45,
                     limit: int = 3) -> list[tuple[int, int]]:
    """Persistent maxima of the smoothed distance transform, greedily separated.

    A merge of two filaments leaves two interior width maxima with a dip between
    them. Requiring separation and a floor relative to the global maximum drops the
    ripple that raw local maxima would produce along a single filament.
    """
    distance = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
    smooth = cv2.GaussianBlur(distance, (0, 0), 1.0)
    peak = float(smooth.max())
    if peak <= 0:
        return []
    ys, xs = np.nonzero(mask)
    order = np.argsort(-smooth[ys, xs])
    chosen: list[tuple[int, int]] = []
    for index in order:
        y, x = int(ys[index]), int(xs[index])
        value = float(smooth[y, x])
        if value < floor_fraction * peak:
            break
        if all((y - cy) ** 2 + (x - cx) ** 2 >= min_separation ** 2 for cy, cx in chosen):
            chosen.append((y, x))
        if len(chosen) >= limit:
            break
    return chosen


def watershed_split(mask: np.ndarray, min_piece: int = 40, min_separation: int = 6,
                    floor_fraction: float = 0.45, limit: int = 3) -> list[np.ndarray]:
    """Marker-controlled watershed on the negated distance transform, inside the mask."""
    markers = distance_markers(mask, min_separation, floor_fraction, limit)
    if len(markers) < 2:
        return [mask]
    distance = cv2.GaussianBlur(cv2.distanceTransform(mask, cv2.DIST_L2, 5), (0, 0), 1.0)
    labels = np.zeros(mask.shape, np.int32)
    queue: list[tuple[float, int, int, int]] = []
    for index, (y, x) in enumerate(markers, start=1):
        labels[y, x] = index
        heapq.heappush(queue, (-float(distance[y, x]), y, x, index))
    while queue:
        _, y, x, label = heapq.heappop(queue)
        for dy, dx in NEIGHBOURS:
            ny, nx = y + dy, x + dx
            if not (0 <= ny < mask.shape[0] and 0 <= nx < mask.shape[1]):
                continue
            if not mask[ny, nx] or labels[ny, nx]:
                continue
            labels[ny, nx] = label
            heapq.heappush(queue, (-float(distance[ny, nx]), ny, nx, label))
    pieces = []
    for index in range(1, len(markers) + 1):
        piece = ((labels == index) & (mask > 0)).astype(np.uint8)
        if int(piece.sum()) >= min_piece:
            pieces.append(piece)
    return pieces if len(pieces) >= 2 else [mask]


def neck_cut_split(mask: np.ndarray, min_piece: int = 40, bins: int = 32,
                   edge: int = 4) -> list[np.ndarray]:
    """Cut perpendicular to the principal axis at the narrowest interior section."""
    ys, xs = np.nonzero(mask)
    if len(xs) < 80:
        return [mask]
    points = np.stack([xs, ys], 1).astype(np.float32)
    centre = points.mean(0)
    _, _, axes = np.linalg.svd(points - centre, full_matrices=False)
    along = (points - centre) @ axes[0]
    distance = cv2.distanceTransform(mask, cv2.DIST_L2, 5)[ys, xs]
    edges = np.linspace(along.min(), along.max(), bins + 1)
    widths, centres = [], []
    for index in range(bins):
        selected = (along >= edges[index]) & (along < edges[index + 1])
        if selected.sum() >= 5:
            widths.append(float(distance[selected].max()))
            centres.append(0.5 * (edges[index] + edges[index + 1]))
    if len(widths) < 2 * edge + 3:
        return [mask]
    interior = np.asarray(widths[edge:-edge])
    position = int(np.argmin(interior)) + edge
    cut = centres[position]
    side = (along < cut)
    first = np.zeros(mask.shape, np.uint8)
    second = np.zeros(mask.shape, np.uint8)
    first[ys[side], xs[side]] = 1
    second[ys[~side], xs[~side]] = 1
    pieces = [piece for piece in (first, second) if int(piece.sum()) >= min_piece]
    return pieces if len(pieces) == 2 else [mask]


def apply_split_operator(population, operator, oracle_gate: bool = True, **kwargs):
    """Run a splitter, gated either by truth (ceiling analysis) or on every mask.

    `oracle_gate=True` restricts the operator to predictions that genuinely cover
    two or more truths. That isolates operator quality from gate quality and is
    never a deployable configuration.
    """
    repaired, attempted, split, pieces_made = [], 0, 0, 0
    for record in population:
        predicted, truths = record["predicted"], record["truths"]
        if not predicted:
            repaired.append(dict(record, predicted=list(predicted)))
            continue
        ious = (np.asarray(mu.iou(predicted, truths, [0] * len(truths)))
                if truths else np.zeros((len(predicted), 0)))
        output = []
        for index, rle in enumerate(predicted):
            if oracle_gate:
                covered = [g for g in range(len(truths)) if ious[index, g] > TOUCH]
                if len(covered) < 2:
                    output.append(rle)
                    continue
            attempted += 1
            parts = operator(mu.decode(rle), **kwargs)
            if len(parts) < 2:
                output.append(rle)
                continue
            split += 1
            pieces_made += len(parts)
            output.extend(encode(part) for part in parts)
        repaired.append(dict(record, predicted=output))
    return repaired, dict(masks_attempted=attempted, masks_split=split,
                          pieces_produced=pieces_made, oracle_gate=oracle_gate)
