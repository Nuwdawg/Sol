"""Cut a merged mask along the interface between its two width maxima.

exp115 showed that neither existing splitter fails for want of tuning. Across four
marker settings the rescue count never passed 17 of an available 40, while pieces
per mask moved from 2.9 to 1.6. The position of the cut is wrong, not its
frequency:

- `neck_cut_split` severs perpendicular to the mask's global principal axis, which
  cannot follow the interface of a curved merge;
- `watershed_split` floods outward from distance-transform maxima, which sit on the
  thickest part of each filament rather than on the boundary between them.

Truth assignment wins 40 crossings because it cuts along the actual interface. This
operator approximates that without truth: take the two dominant width maxima as
terminals, then find the cheapest path *across* the mask that separates them, where
crossing a wide region is expensive and crossing a narrow one is cheap. The cut
therefore follows the local width minimum wherever it runs, curved or straight.

Cost of entering a pixel is the distance transform itself, so the path prefers thin
places; a constant floor keeps it from meandering through background-adjacent
pixels for free. The path is grown with Dijkstra between the two mask boundary
points that lie closest to the perpendicular bisector of the two maxima, which is
where a separating cut must begin and end.
"""
from __future__ import annotations

import heapq

import cv2
import numpy as np

from experiments.exp_113_ceilings.src.splitters import distance_markers

NEIGHBOURS = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]


def _boundary_pixels(mask: np.ndarray) -> np.ndarray:
    eroded = cv2.erode(mask, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)))
    ys, xs = np.nonzero(mask & ~eroded.astype(bool) if mask.dtype == bool
                        else (mask - eroded))
    return np.stack([ys, xs], 1)


def _cheapest_separating_path(mask: np.ndarray, distance: np.ndarray,
                              start: tuple[int, int], goal_side: np.ndarray,
                              floor: float = 0.35):
    """Dijkstra from one boundary pixel to any pixel on the far side of the mask.

    Entering a pixel costs its distance-to-background plus a floor, so the path is
    drawn through the narrowest available crossing.
    """
    rows, columns = mask.shape
    best = np.full(mask.shape, np.inf, dtype=np.float32)
    parent: dict[tuple[int, int], tuple[int, int]] = {}
    best[start] = 0.0
    queue = [(0.0, start)]
    while queue:
        accumulated, node = heapq.heappop(queue)
        if accumulated > best[node]:
            continue
        if goal_side[node]:
            path = [node]
            while path[-1] in parent:
                path.append(parent[path[-1]])
            return path
        y, x = node
        for dy, dx in NEIGHBOURS:
            ny, nx = y + dy, x + dx
            if not (0 <= ny < rows and 0 <= nx < columns) or not mask[ny, nx]:
                continue
            step = float(distance[ny, nx]) + floor
            candidate = accumulated + step
            if candidate < best[ny, nx]:
                best[ny, nx] = candidate
                parent[(ny, nx)] = node
                heapq.heappush(queue, (candidate, (ny, nx)))
    return None


def interface_cut_split(mask: np.ndarray, min_piece: int = 40, min_separation: int = 12,
                        floor_fraction: float = 0.55, thickness: int = 1) -> list[np.ndarray]:
    """Split by carving the cheapest separating path between the two width maxima."""
    markers = distance_markers(mask, min_separation, floor_fraction, limit=2)
    if len(markers) < 2:
        return [mask]
    (ay, ax), (by, bx) = markers[0], markers[1]
    distance = cv2.distanceTransform(mask, cv2.DIST_L2, 5)

    # The cut must run across the mask, so start from the boundary pixel nearest the
    # midpoint of the two maxima and end anywhere on the opposite boundary half.
    boundary = _boundary_pixels(mask)
    if len(boundary) < 4:
        return [mask]
    midpoint = np.array([(ay + by) / 2.0, (ax + bx) / 2.0])
    axis = np.array([by - ay, bx - ax], dtype=np.float64)
    norm = np.linalg.norm(axis)
    if norm < 1e-6:
        return [mask]
    axis /= norm
    offsets = (boundary - midpoint) @ np.array([-axis[1], axis[0]])
    start = tuple(int(v) for v in boundary[int(np.argmax(offsets))])
    far_side = np.zeros(mask.shape, dtype=bool)
    opposite = boundary[offsets < 0]
    if not len(opposite):
        return [mask]
    far_side[opposite[:, 0], opposite[:, 1]] = True

    path = _cheapest_separating_path(mask, distance, start, far_side)
    if path is None:
        # A boundary-only target set strands paths that arrive at an interior pixel,
        # which was 3 of 6 declines in the first diagnosis. Terminating on the far
        # half-plane instead lets any crossing of the midline finish the cut.
        half_plane = np.zeros(mask.shape, dtype=bool)
        ys, xs = np.nonzero(mask)
        offsets_all = (np.stack([ys, xs], 1) - midpoint) @ np.array([-axis[1], axis[0]])
        half_plane[ys[offsets_all < 0], xs[offsets_all < 0]] = True
        path = _cheapest_separating_path(mask, distance, start, half_plane)
    if path is None:
        return [mask]

    # A 1 px carve along an 8-connected path does not sever an 8-connected mask,
    # because diagonal neighbours stay linked: that was the other 2 declines.
    carved = mask.copy()
    for y, x in path:
        cv2.circle(carved, (int(x), int(y)), max(thickness, 2), 0, -1)
    count, labels = cv2.connectedComponents(carved, connectivity=4)
    pieces = []
    for index in range(1, count):
        piece = (labels == index).astype(np.uint8)
        if int(piece.sum()) >= min_piece:
            pieces.append(piece)
    if len(pieces) < 2:
        return [mask]
    # Give the carved pixels back to their nearest piece so the split conserves area.
    restored = []
    for piece in pieces:
        grown = cv2.dilate(piece, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)))
        restored.append(((grown > 0) & (mask > 0)).astype(np.uint8))
    # Keep only the two largest pieces. The first firing of this operator produced
    # three pieces from a two-truth merge, so an unbounded piece count over-splits
    # even when the carve lands correctly; truth assignment yields two.
    claimed = np.zeros(mask.shape, np.uint8)
    final = []
    for piece in sorted(restored, key=lambda p: -int(p.sum()))[:2]:
        piece = ((piece > 0) & (claimed == 0)).astype(np.uint8)
        if int(piece.sum()) >= min_piece:
            claimed |= piece
            final.append(piece)
    if len(final) < 2:
        return [mask]
    # Any area the two largest pieces did not claim returns to the nearer of them,
    # so the split conserves the original mask rather than discarding a third lobe.
    leftover = ((mask > 0) & (claimed == 0)).astype(np.uint8)
    if int(leftover.sum()):
        first, second = final
        near_first = cv2.distanceTransform((1 - first).astype(np.uint8), cv2.DIST_L2, 5)
        near_second = cv2.distanceTransform((1 - second).astype(np.uint8), cv2.DIST_L2, 5)
        final = [((first > 0) | ((leftover > 0) & (near_first <= near_second))).astype(np.uint8),
                 ((second > 0) | ((leftover > 0) & (near_first > near_second))).astype(np.uint8)]
    return final
