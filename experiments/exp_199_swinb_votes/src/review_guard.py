"""Native-resolution review of the instances the guard re-adds to a test file (CPU).

Each panel: the re-added instance (yellow, thick) and the final rule's instances (red). Images
come from the competition archive by HTTP range reads (final_decision/analysis/httpzip.py).

    python -m experiments.exp_199_swinb_votes.src.review_guard --archive-url URL
        --final exp169.csv --guarded submission_guarded.csv --output review.jpg
"""
from __future__ import annotations

import argparse
import csv
import io
import sys
import zipfile
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
from pycocotools import mask as mu

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "final_decision" / "analysis"))
from httpzip import RangeFile  # noqa: E402


def read(path):
    by = defaultdict(list)
    for row in csv.DictReader(open(path, newline="")):
        by[row["filament_id"].rsplit("_", 1)[0]].append(row["segmentation_rle"])
    return by


def outline(canvas, mask, colour, thickness):
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cv2.drawContours(canvas, contours, -1, colour, thickness)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--archive-url", required=True)
    parser.add_argument("--final", type=Path, required=True)
    parser.add_argument("--guarded", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    final, guarded = read(args.final), read(args.guarded)
    z = zipfile.ZipFile(io.BufferedReader(RangeFile(args.archive_url), buffer_size=1 << 20))
    panels = []
    for stem in sorted(guarded):
        added = guarded[stem][len(final.get(stem, [])):]          # the guard appends after the final rows
        if not added:
            continue
        if guarded[stem][:len(final.get(stem, []))] != final.get(stem, []):
            raise SystemExit(f"{stem}: the guarded file changed the final rule's own rows")
        data = z.read(f"MAGFiLO_1.0_Kaggle_2026/test/test_images/{stem}.jpeg")
        image = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_GRAYSCALE)
        for counts in added:
            target = mu.decode({"size": [2048, 2048], "counts": counts.encode()})
            canvas = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
            for other in final.get(stem, []):
                outline(canvas, mu.decode({"size": [2048, 2048], "counts": other.encode()}), (0, 0, 255), 2)
            outline(canvas, target, (0, 255, 255), 3)
            ys, xs = np.nonzero(target)
            cy, cx = (ys.min() + ys.max()) // 2, (xs.min() + xs.max()) // 2
            half = max(ys.max() - ys.min(), xs.max() - xs.min()) // 2 + 160
            y0, y1, x0, x1 = max(0, cy - half), min(2048, cy + half), max(0, cx - half), min(2048, cx + half)
            panel = cv2.resize(canvas[y0:y1, x0:x1], (520, 520), interpolation=cv2.INTER_AREA)
            cv2.putText(panel, f"{stem} +{int(target.sum())} px", (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            panels.append(panel)
    cols = 4
    rows = [np.hstack(panels[i:i + cols] + [np.zeros_like(panels[0])] * (cols - len(panels[i:i + cols])))
            for i in range(0, len(panels), cols)]
    cv2.imwrite(str(args.output), np.vstack(rows), [cv2.IMWRITE_JPEG_QUALITY, 85])
    print("REVIEW", args.output, len(panels))


if __name__ == "__main__":
    main()
