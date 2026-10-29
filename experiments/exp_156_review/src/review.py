"""exp156: what does a candidate submission change against the current best, photograph by photograph?

HOST_GUIDANCE asks for native-resolution review of the instances a change adds or removes before
a variant is promoted. This script matches the two CSVs instance by instance (IoU > 0.5) on the
test photographs and reports:
- counts of kept, added and removed instances, and the IoU of kept pairs;
- the added and removed instances by area and by distance from the disk centre;
- a review sheet: 512 px native crops around the largest changes, with the photograph in grey,
  instances present in both in green, only in the candidate in red, only in the base in blue.
No labels are involved: the test set has none here.

    python -m experiments.exp_156_review.src.review --base base.csv --candidate cand.csv
        --images <test_images> --output /kaggle/working/review
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
from pycocotools import mask as mu

from shared.solar_disk import find_disk

SIZE = 2048
TILE = 512


def read_csv(path: Path) -> dict:
    out = defaultdict(list)
    with Path(path).open() as handle:
        for row in csv.DictReader(handle):
            stem = row["filament_id"].rsplit("_", 1)[0]
            out[stem].append(dict(size=[SIZE, SIZE], counts=row["segmentation_rle"].encode("ascii")))
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--images", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tiles", type=int, default=16)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    base, cand = read_csv(args.base), read_csv(args.candidate)
    stems = sorted(set(base) | set(cand))
    kept_ious, added, removed, changes = [], [], [], []
    for stem in stems:
        b, c = base.get(stem, []), cand.get(stem, [])
        ious = np.asarray(mu.iou(c, b, [0] * len(b))).reshape(len(c), len(b)) if b and c else np.zeros((len(c), len(b)))
        matched_c, matched_b = set(), set()
        for i, j in sorted(zip(*np.nonzero(ious > 0.5)), key=lambda p: -ious[p]):
            if i in matched_c or j in matched_b:
                continue
            matched_c.add(i); matched_b.add(j); kept_ious.append(float(ious[i, j]))
        gray = None
        for kind, masks, matched, sink in (("added", c, matched_c, added), ("removed", b, matched_b, removed)):
            for k, rle in enumerate(masks):
                if k in matched:
                    continue
                if gray is None:
                    gray = cv2.imread(str(args.images / f"{stem}.jpeg"), cv2.IMREAD_GRAYSCALE)
                    disk = find_disk(gray)
                x, y, w, h = mu.toBbox(rle)
                area = float(mu.area(rle))
                radius = float(np.hypot(x + w / 2 - disk[0], y + h / 2 - disk[1]) / disk[2])
                sink.append(dict(stem=stem, area=area, radius=radius))
                changes.append((area, stem, kind, (x + w / 2, y + h / 2)))

    def summary(items):
        areas = np.array([i["area"] for i in items]) if items else np.zeros(0)
        radii = np.array([i["radius"] for i in items]) if items else np.zeros(0)
        return dict(n=len(items), area_median=float(np.median(areas)) if len(areas) else None,
                    area_bins={f"<{t}": int((areas < t).sum()) for t in (300, 1000, 3000)},
                    limb_share=float((radii > 0.9).mean()) if len(radii) else None)

    report = dict(photographs=len(stems), base_instances=sum(len(v) for v in base.values()),
                  candidate_instances=sum(len(v) for v in cand.values()), kept=len(kept_ious),
                  kept_iou_median=float(np.median(kept_ious)) if kept_ious else None,
                  kept_iou_p10=float(np.percentile(kept_ious, 10)) if kept_ious else None,
                  added=summary(added), removed=summary(removed))
    (args.output / "review.json").write_text(json.dumps(report, indent=1))
    print("REVIEW", json.dumps(report), flush=True)

    tiles = []
    for area, stem, kind, (cx, cy) in sorted(changes, reverse=True)[:args.tiles]:
        gray = cv2.imread(str(args.images / f"{stem}.jpeg"), cv2.IMREAD_GRAYSCALE)
        x0 = int(np.clip(cx - TILE // 2, 0, SIZE - TILE)); y0 = int(np.clip(cy - TILE // 2, 0, SIZE - TILE))
        canvas = cv2.cvtColor(gray[y0:y0 + TILE, x0:x0 + TILE], cv2.COLOR_GRAY2BGR)
        b_masks = [mu.decode(r)[y0:y0 + TILE, x0:x0 + TILE] for r in base.get(stem, [])]
        c_masks = [mu.decode(r)[y0:y0 + TILE, x0:x0 + TILE] for r in cand.get(stem, [])]
        for masks, colour in ((b_masks, (255, 128, 0)), (c_masks, (0, 0, 255))):
            for m in masks:
                contours, _ = cv2.findContours(np.ascontiguousarray(m), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
                cv2.drawContours(canvas, contours, -1, colour, 1)
        cv2.putText(canvas, f"{stem} {kind} {int(area)}px", (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1,
                    cv2.LINE_AA)
        tiles.append(canvas)
    if tiles:
        while len(tiles) % 4:
            tiles.append(np.zeros_like(tiles[0]))
        sheet = np.vstack([np.hstack(tiles[i:i + 4]) for i in range(0, len(tiles), 4)])
        cv2.imwrite(str(args.output / "review_sheet.jpg"), sheet, [cv2.IMWRITE_JPEG_QUALITY, 75])


if __name__ == "__main__":
    main()
