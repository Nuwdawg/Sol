"""A real minimum-path-cost audit for endpoint continuation, built on the raw image.

Why this supersedes exp109 and exp110
-------------------------------------
Both earlier audits rejected endpoint continuation on summary statistics inside a
cone: the mean support (exp109) and per-ray means, minima and maxima (exp110). The
second concluded that a geodesic is blocked because every ray crosses a pixel of
zero learned support. That argument is invalid. With a finite cost such as

    c(x) = alpha * (1 - D(x)) + beta * (1 - O(x)) + step

a zero-support pixel is merely expensive, and Dijkstra crosses it whenever every
competing path costs more. Only an infinite cost or a forbidden traversal blocks a
path. So the correct quantity is the actual minimum cost of reaching a point
`reach` pixels ahead:

    C*(p) = min over paths gamma from the endpoint of sum_{x in gamma} c(x)

Why the cost field uses the image, not the network
--------------------------------------------------
exp110 measured that the learned foreground field is quieter ahead of truncated
masks than ahead of healthy ones (best-ray mean 0.0082 against 0.0137), while raw
dark absorption separated them at AUC 0.611 - the only image-derived signal above
0.6 all session. The network never learned these faint distal extensions, so a cost
field built from its output cannot find them. This cost field is therefore built
from absorption against a large-scale background, plus orientation consistency from
the structure tensor, both computed directly from the photograph.

What is reported
----------------
Per instance, for the better of its two endpoints:

- `path_cost`: minimum normalised cost per pixel of reaching `reach` px ahead;
- `path_dark`: mean absorption along that optimal path;
- `path_orientation`: mean orientation consistency along it;
- `path_truth`: fraction of the optimal path covered by ground truth. LABEL only,
  used to verify the populations, never as a feature.

If truncated masks have cheaper, darker, better-aligned continuations than healthy
masks, endpoint continuation is viable and the earlier rejections were artefacts of
the statistic. If they do not, the rejection stands on a correct argument this time.
"""
from __future__ import annotations

import argparse
import heapq
import json
from pathlib import Path

import cv2
import numpy as np
from pycocotools import mask as mu

from experiments.exp_033_inference.src.sweep import encode, records_for
from shared.utils import paint_panoptic

KERNEL = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
TOUCH = 0.05
IMAGES = Path("data/competition/unpacked/MAGFiLO_1.0_Kaggle_2026/train/train_images")
NEIGHBOURS = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]


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


def orientation_field(patch, sigma=3.0):
    """Ridge direction per pixel from the structure tensor, as unit vectors."""
    gx = cv2.Sobel(patch, cv2.CV_32F, 1, 0, ksize=5)
    gy = cv2.Sobel(patch, cv2.CV_32F, 0, 1, ksize=5)
    jxx = cv2.GaussianBlur(gx * gx, (0, 0), sigma)
    jyy = cv2.GaussianBlur(gy * gy, (0, 0), sigma)
    jxy = cv2.GaussianBlur(gx * gy, (0, 0), sigma)
    angle = 0.5 * np.arctan2(2 * jxy, jxx - jyy)
    return np.stack([-np.sin(angle), np.cos(angle)], -1)


def cheapest_continuation(dark, ridge, tip, direction, reach, cone, alpha, beta, step):
    """Dijkstra from the endpoint; return the best path reaching `reach` px ahead.

    The cost of entering a pixel is finite everywhere, so a low-evidence pixel is
    expensive rather than impassable - the correction that motivated this audit.
    """
    height, width = dark.shape
    x0 = int(max(tip[0] - reach - 2, 0)); x1 = int(min(tip[0] + reach + 3, width))
    y0 = int(max(tip[1] - reach - 2, 0)); y1 = int(min(tip[1] + reach + 3, height))
    if x1 - x0 < 8 or y1 - y0 < 8:
        return None
    local_dark = dark[y0:y1, x0:x1]
    local_ridge = ridge[y0:y1, x0:x1]
    alignment = np.abs(local_ridge[..., 0] * direction[0] + local_ridge[..., 1] * direction[1])
    cost = alpha * (1.0 - local_dark) + beta * (1.0 - alignment) + step

    start = (int(round(tip[1])) - y0, int(round(tip[0])) - x0)
    rows, columns = cost.shape
    if not (0 <= start[0] < rows and 0 <= start[1] < columns):
        return None
    best = np.full(cost.shape, np.inf)
    parent = {}
    best[start] = 0.0
    queue = [(0.0, start)]
    target, target_cost, target_steps = None, np.inf, 0
    while queue:
        accumulated, node = heapq.heappop(queue)
        if accumulated > best[node]:
            continue
        dy = (node[0] + y0) - tip[1]
        dx = (node[1] + x0) - tip[0]
        distance = np.hypot(dx, dy)
        forward = (dx * direction[0] + dy * direction[1]) / max(distance, 1e-6)
        if distance >= reach and forward >= np.cos(np.radians(cone)):
            target, target_cost = node, accumulated
            break
        if distance > reach * 1.6:          # never wander far beyond the goal ring
            continue
        for oy, ox in NEIGHBOURS:
            neighbour = (node[0] + oy, node[1] + ox)
            if not (0 <= neighbour[0] < rows and 0 <= neighbour[1] < columns):
                continue
            stride = 1.414 if oy and ox else 1.0
            candidate = accumulated + cost[neighbour] * stride
            if candidate < best[neighbour]:
                best[neighbour] = candidate
                parent[neighbour] = node
                heapq.heappush(queue, (candidate, neighbour))
    if target is None:
        return None
    path = [target]
    while path[-1] in parent:
        path.append(parent[path[-1]])
    path = np.array(path)
    target_steps = len(path)
    rows_index, columns_index = path[:, 0], path[:, 1]
    return dict(cost=float(target_cost / max(target_steps, 1)),
                dark=float(local_dark[rows_index, columns_index].mean()),
                orientation=float(alignment[rows_index, columns_index].mean()),
                pixels=[(int(c + x0), int(r + y0)) for r, c in path])


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
    parser.add_argument("--output", type=Path, default=Path("data/exp112_geodesic_cost"))
    parser.add_argument("--conf", type=float, default=0.35)
    parser.add_argument("--grow", type=int, default=-1)
    parser.add_argument("--min-area", type=int, default=100)
    parser.add_argument("--reach", type=float, default=20.0)
    parser.add_argument("--cone", type=float, default=45.0)
    parser.add_argument("--alpha", type=float, default=1.0, help="weight on absorption")
    parser.add_argument("--beta", type=float, default=1.0, help="weight on orientation")
    parser.add_argument("--step", type=float, default=0.05, help="cost of a step regardless")
    parser.add_argument("--limit", type=int, default=0, help="stop after N instances, 0 = all")
    args = parser.parse_args()
    cv2.setNumThreads(2)
    args.output.mkdir(parents=True, exist_ok=True)
    annotations = next(Path(".").rglob("MAGFiLO_1.0_Annotations_kaggle2026_train.json"))
    records, split = records_for(annotations)
    pool = json.loads(args.pool.read_text())

    near_rows, healthy_rows, seen = [], [], 0
    for photograph, passes in records.items():
        raw = cv2.imread(str(IMAGES / photograph), cv2.IMREAD_GRAYSCALE)
        if raw is None:
            continue
        floated = raw.astype(np.float32)
        background = cv2.GaussianBlur(floated, (0, 0), 25.0)
        dark = np.clip((background - floated) / 40.0, 0, 1)
        ridge = orientation_field(floated)

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
                    found = cheapest_continuation(dark, ridge, tip, direction, args.reach,
                                                  args.cone, args.alpha, args.beta, args.step)
                    if found is None:
                        continue
                    pixels = found.pop("pixels")
                    found["truth"] = float(np.mean([truth[y, x] for x, y in pixels]))
                    if best is None or found["truth"] > best["truth"]:
                        best = found
                if best is None:
                    continue
                best["iou"] = iou
                (near_rows if iou < 0.5 else healthy_rows).append(best)
                seen += 1
        if args.limit and seen >= args.limit:
            break

    print(f"minimum-cost continuation, reach {args.reach} px, cone {args.cone} deg, "
          f"alpha {args.alpha} beta {args.beta} step {args.step}")
    print(f"near-threshold {len(near_rows)}   healthy {len(healthy_rows)}\n")
    print(f"{'feature':14s} {'AUC':>6} {'AUCneg':>7} {'near med':>10} {'healthy med':>12}"
          f"  best over both directions")
    report = dict(split=split.summary(), near=len(near_rows), healthy=len(healthy_rows),
                  settings=vars(args) | dict(pool=str(args.pool), output=str(args.output)),
                  features={})
    for key in ("truth", "cost", "dark", "orientation"):
        positive = np.array([r[key] for r in near_rows], float)
        negative = np.array([r[key] for r in healthy_rows], float)
        if not positive.size or not negative.size:
            continue
        area = rank_auc(positive, negative)
        best = best_operating_point(positive, negative)
        report["features"][key] = dict(auc=round(area, 4), best=best)
        note = "   <- LABEL" if key == "truth" else ""
        print(f"{key:14s} {area:6.3f} {1 - area:7.3f} {np.median(positive):10.4f} "
              f"{np.median(negative):12.4f}  {best['side']} q{best['quantile']:.2f} "
              f"net {best['net']:+d} precision {best['precision']:.3f}{note}")
    (args.output / "report.json").write_text(json.dumps(report, indent=2, default=str))
    print(f"\nwrote {args.output / 'report.json'}")


if __name__ == "__main__":
    main()
