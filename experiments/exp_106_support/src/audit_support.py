"""Does independent foreground evidence mark the masks that stopped too early?

The hypothesis, stated narrowly enough to be wrong
--------------------------------------------------
exp105 measured that near-threshold failures are 83% strict 1:1 - one prediction,
one truth, wrong extent - and that those masks are *under*-segmented: median 575
truth pixels missing against 139 spurious. exp104 then measured that uniform
dilation costs 0.027 PQ. So the extent error is real but the repair must be
triggered per instance.

exp105c killed the geometric triggers: elongation, solidity and spine width
variation all sit within 0.08 AUC of chance, individually and combined, and the
narrow-neck statistic reaches AUC 0.712 yet only 3-7% precision, which nets -207
objects at its best operating point.

What survives is evidence that is *causally* connected to the error rather than
correlated with it. If a mask stopped early, then immediately outside its boundary
there should still be filament. Two independent sources can say so without ground
truth:

1. **Semantic support.** exp085's field reaches mask Dice 0.703 and spine Dice
   0.635. Foreground persisting in an exterior annulus is direct evidence that the
   instance continues past its boundary. This is the *directional* signal.
2. **Cross-architecture disagreement.** exp002 and exp097 are separately trained at
   different resolutions; exp102 showed their pooled proposals beat either alone.
   Where they disagree about a boundary the edge is uncertain. This is a *gating*
   signal: it says the boundary is unreliable, not which way to move it.

Measured per instance, following the user's formulation:

    R_r    = dilate(M, r) \\ M                    exterior annulus
    B_r    = M \\ erode(M, r)                     inner boundary band
    s(M)   = mean(P_sem | R_r) - mean(P_sem | B_r)

An under-grown instance should show unusually high exterior support relative to
its own boundary band. The audit reports rank AUC separating the near-threshold
1:1 population from the healthy 1:1 population, and precision at operating points,
because exp105c showed AUC alone can hide a useless trigger.

Nothing here fits anything. Ground truth is used only to label the two populations
for scoring the discriminator, never inside the score itself.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from pycocotools import mask as mu

from experiments.exp_033_inference.src.sweep import encode, records_for
from experiments.exp_097_runpod_yolo.src.events import emit as emit_event
from shared.utils import paint_panoptic

KERNEL = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
TOUCH = 0.05
BAND = 0.4, 0.5


def emit_masks(entries: list[dict], conf: float, grow: int, min_area: int) -> list[dict]:
    candidates = []
    for entry in entries:
        if entry["score"] < conf:
            continue
        mask = mu.decode(entry["rle"])
        if grow < 0:
            mask = cv2.erode(mask, KERNEL, iterations=-grow)
        candidates.append((entry["score"], mask))
    return [encode(mask) for _, mask, _ in paint_panoptic(candidates, min_area=min_area)]


def support(mask: np.ndarray, field: np.ndarray, radius: int) -> dict:
    """Exterior annulus support minus inner boundary support, per the user's formula."""
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1))
    outside = cv2.dilate(mask, kernel) & ~mask
    inside = mask & ~cv2.erode(mask, kernel)
    if outside.sum() < 20 or inside.sum() < 20:
        return dict(exterior=None, interior=None, delta=None)
    exterior = float(field[outside > 0].mean())
    interior = float(field[inside > 0].mean())
    return dict(exterior=exterior, interior=interior, delta=exterior - interior)


def disagreement(mask: np.ndarray, other_pool: list[dict], radius: int) -> float | None:
    """How much another model's foreground differs around this boundary.

    The other model's instances are rasterised into one foreground map, so this is
    a boundary-uncertainty measure, not an instance correspondence.

    Passing the same pool as `--pool` does NOT give a null control: it measures how
    a mask's boundary ring differs from its own pool's foreground, which is
    self-consistency. A 2026-09-16 run made that mistake and read AUC 0.597, which
    is a real number about the wrong quantity. The caller labels the feature
    accordingly.
    """
    if not other_pool:
        return None
    other = np.zeros(mask.shape, np.uint8)
    for rle in other_pool:
        other |= mu.decode(rle)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1))
    ring = cv2.dilate(mask, kernel) & ~cv2.erode(mask, kernel)
    if ring.sum() < 20:
        return None
    return float((mask[ring > 0] != other[ring > 0]).mean())


def rank_auc(positive: list[float], negative: list[float]) -> float:
    positive, negative = np.asarray(positive, float), np.asarray(negative, float)
    if not positive.size or not negative.size:
        return float("nan")
    ranks = np.concatenate([positive, negative]).argsort().argsort() + 1
    return float((ranks[:positive.size].sum() - positive.size * (positive.size + 1) / 2)
                 / (positive.size * negative.size))


def operating_points(positive: list[float], negative: list[float], quantiles) -> list[dict]:
    """Precision at thresholds taken from the healthy population, where a trigger must live."""
    positive, negative = np.asarray(positive, float), np.asarray(negative, float)
    rows = []
    for q in quantiles:
        cut = float(np.quantile(negative, q))
        caught = int((positive >= cut).sum())
        damaged = int((negative >= cut).sum())
        rows.append(dict(quantile=q, cut=round(cut, 4), caught=caught,
                         of=int(positive.size), damaged=damaged,
                         precision=round(caught / max(caught + damaged, 1), 4),
                         net=caught - damaged))
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pool", type=Path, required=True, help="instance pool to audit")
    parser.add_argument("--other-pool", type=Path, default=None,
                        help="a differently trained model's pool, for disagreement")
    parser.add_argument("--fields", type=Path, default=None,
                        help="directory of semantic probability maps, <stem>.png, 0-255")
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--conf", type=float, default=0.35)
    parser.add_argument("--grow", type=int, default=-1)
    parser.add_argument("--min-area", type=int, default=100)
    parser.add_argument("--radius", type=int, default=4)
    args = parser.parse_args()
    cv2.setNumThreads(2)
    args.output.mkdir(parents=True, exist_ok=True)
    records, split = records_for(args.annotations)
    pool = json.loads(args.pool.read_text())
    other = json.loads(args.other_pool.read_text()) if args.other_pool else {}

    near, healthy = [], []
    for photograph, passes in records.items():
        emitted = emit_masks(pool.get(photograph, []), args.conf, args.grow, args.min_area)
        if not emitted:
            continue
        decoded = [mu.decode(rle) for rle in emitted]
        field = None
        if args.fields:
            path = args.fields / f"{Path(photograph).stem}.png"
            if path.is_file():
                field = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255.0
        other_rles = [entry["rle"] for entry in other.get(photograph, [])
                      if entry["score"] >= args.conf] if other else []
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
                row = dict(photograph=photograph, iou=round(iou, 4))
                if field is not None:
                    row.update({f"sem_{k}": v for k, v in
                                support(decoded[p], field, args.radius).items()})
                row["disagreement"] = disagreement(decoded[p], other_rles, args.radius)
                if BAND[0] <= iou < BAND[1]:
                    near.append(row)
                elif iou >= BAND[1]:
                    healthy.append(row)

    report = dict(split=split.summary(), near=len(near), healthy=len(healthy),
                  rule=dict(conf=args.conf, grow=args.grow, min_area=args.min_area,
                            radius=args.radius), features={})
    for key in ("sem_delta", "sem_exterior", "disagreement"):
        positive = [r[key] for r in near if r.get(key) is not None]
        negative = [r[key] for r in healthy if r.get(key) is not None]
        if not positive or not negative:
            continue
        report["features"][key] = dict(
            auc=round(rank_auc(positive, negative), 4),
            near_median=round(float(np.median(positive)), 4),
            healthy_median=round(float(np.median(negative)), 4),
            operating_points=operating_points(positive, negative, (0.80, 0.90, 0.95, 0.98)))
        emit_event("FEATURE", dict(name=key, **{k: v for k, v in report["features"][key].items()
                                                if k != "operating_points"}))
    (args.output / "report.json").write_text(json.dumps(report, indent=2))
    (args.output / "rows.json").write_text(json.dumps(dict(near=near, healthy=healthy)))
    emit_event("SUMMARY", dict(near=len(near), healthy=len(healthy),
                               features=sorted(report["features"])))


if __name__ == "__main__":
    main()
