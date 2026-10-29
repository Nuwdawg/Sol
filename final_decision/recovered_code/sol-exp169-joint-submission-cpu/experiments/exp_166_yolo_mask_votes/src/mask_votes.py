"""exp166: do the YOLO masks improve the boundary when they join the pixel vote?

The final rule's mask is the pixel majority over a Mask2Former candidate and its partners in the seven
other dihedral views: eight estimates from one network. The two YOLO detectors are trained separately,
so their masks are independent estimates of the same boundary. Their convention is looser (exp104: a
1 px erosion is their best emission), so each YOLO partner mask is eroded by `erode` pixels before it
votes. Scores stay the final rule's (geometric mean with YOLO votes, w 2); only masks change.

    python -m experiments.exp_166_yolo_mask_votes.src.mask_votes --cache ... --yolo yolov8l=... --yolo yolo11m=...
        --annotations ... --output ...
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import cv2
import numpy as np
from pycocotools import mask as mu

from experiments.exp_033_inference.src.cached_tta import vote_rle
from experiments.exp_033_inference.src.sweep import records_for
from experiments.exp_149_evidence_fusion.src import evidence as E
from experiments.exp_157_shift_views.src.views import CONFS, as_text, partners
from experiments.exp_159_yolo_votes.src.votes import load_pool, yolo_support

KERNEL = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))


def yolo_partners(entry: dict, pool_entries: list[dict], erode: int) -> list[dict | None]:
    """Each full candidate's best-IoU (> 0.5) YOLO instance mask, eroded, as an RLE (or None)."""
    rles = [E.as_rle(c["rle"]) for c in entry["full"]]
    if not rles or not pool_entries:
        return [None] * len(rles)
    others = [E.as_rle(p["rle"]) for p in pool_entries]
    ious = np.asarray(mu.iou(rles, others, [0] * len(others))).reshape(len(rles), len(others))
    out = []
    for i in range(len(rles)):
        j = int(ious[i].argmax())
        if ious[i, j] <= 0.5:
            out.append(None)
            continue
        mask = mu.decode(others[j])
        if erode:
            mask = cv2.erode(mask, KERNEL, iterations=erode)
        out.append(mu.encode(np.asfortranarray(mask)))
    return out


def pool_for(entry: dict, pools: list[list[dict]], erode: int | None) -> list[tuple[float, "E.Crop"]]:
    groups, total, views = partners(entry, ("d4",))
    s = np.array([c["score"] for c in entry["full"]], np.float64)
    for pool_entries in pools:
        total = total + 2.0 * yolo_support(entry, pool_entries)
        views += 2.0
    scores = np.sqrt(s * total / views)
    if erode is not None:
        for pool_entries in pools:
            for i, extra in enumerate(yolo_partners(entry, pool_entries, erode)):
                if extra is not None:
                    groups[i].append(extra)
    out = []
    for i, group in enumerate(groups):
        mask = entry["full"][i]["rle"] if len(group) == 1 else as_text(vote_rle(group, max(1, math.ceil(0.5 * len(group)))))
        out.append((float(scores[i]), E.Crop(mask)))
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--yolo", action="append", default=[])
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    cache = E.load_cache(args.cache)
    records, _ = records_for(args.annotations)
    pools = [load_pool(Path(p)) for _, p in (s.split("=", 1) for s in sorted(args.yolo))]
    report, final = {}, None
    for tag, erode in (("final", None), ("yolo_masks_raw", 0), ("yolo_masks_erode1", 1)):
        pool = {n: pool_for(cache[n], [p.get(n, []) for p in pools], erode) for n in records}
        rows, per = E.score_rule(pool, records, CONFS, min_area=100)
        best = max(rows, key=lambda r: r["pq"])
        if tag == "final":
            final = per[best["conf"]]
        report[tag] = dict(best=best, vs_final=E.bootstrap_difference(final, per[best["conf"]], records))
        print("VARIANT", tag, json.dumps(report[tag]), flush=True)
    (args.output / "exp166_report.json").write_text(json.dumps(report, indent=1, default=float))


if __name__ == "__main__":
    main()
