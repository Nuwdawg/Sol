"""exp200b: would a different polygon rasteriser for the test truth explain the leaderboard gap?

The training truth arrives as polygons and this repository rasterises them with pycocotools. The
hidden test truth arrives as RLE, made by the organisers with a tool nobody has named. Filaments
are a few pixels wide, so a rasteriser that includes the pixels an edge passes through, or that
rounds vertices, moves every boundary by up to a pixel, and would cost every model about the same.
That is the shape of the gap: 0.06-0.08 for every model family.

This rasterises the validation polygons several ways, scores the cached validation predictions of
known submissions against each (the organisers' metric, one record per annotator), and prints the
results beside the public scores. Dilated and eroded pycocotools truth bound the effect.

    python -m experiments.exp_200_test_truth.src.rasterisers --annotations ... --predictions ...
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
from matplotlib.path import Path as Polygon
from PIL import Image, ImageDraw
from pycocotools import mask as mu

from experiments.exp_200_test_truth.src.hypotheses import PUBLIC, as_bytes, match, total
from shared.data_split import assert_disjoint, make_split

SIZE = 2048


def polygons(segmentation) -> list[np.ndarray]:
    return [np.asarray(p, np.float64).reshape(-1, 2) for p in segmentation if len(p) >= 6]


def pycoco(segmentation) -> np.ndarray:
    return mu.decode(mu.merge(mu.frPyObjects(segmentation, SIZE, SIZE)))


def cv2_fill(segmentation, how: str) -> np.ndarray:
    mask = np.zeros((SIZE, SIZE), np.uint8)
    for p in polygons(segmentation):
        if how == "round":
            cv2.fillPoly(mask, [np.round(p).astype(np.int32)], 1)
        elif how == "floor":
            cv2.fillPoly(mask, [np.floor(p).astype(np.int32)], 1)
        elif how == "subpixel":
            cv2.fillPoly(mask, [np.round(p * 16).astype(np.int32)], 1, lineType=cv2.LINE_8, shift=4)
        elif how == "round_half":                        # vertices moved to the pixel-centre convention
            cv2.fillPoly(mask, [np.round(p - 0.5).astype(np.int32)], 1)
    return mask


def pil_fill(segmentation, outline: bool) -> np.ndarray:
    image = Image.new("L", (SIZE, SIZE), 0)
    draw = ImageDraw.Draw(image)
    for p in polygons(segmentation):
        draw.polygon([tuple(v) for v in p], fill=1, outline=1 if outline else None)
    return np.asarray(image, np.uint8)


def mpl_fill(segmentation, offset: float) -> np.ndarray:
    """Pixels whose point (x + offset, y + offset) lies inside the polygon."""
    mask = np.zeros(SIZE * SIZE, bool)
    for p in polygons(segmentation):
        x0, y0 = np.floor(p.min(0)).astype(int) - 1
        x1, y1 = np.ceil(p.max(0)).astype(int) + 1
        x0, y0, x1, y1 = max(x0, 0), max(y0, 0), min(x1, SIZE - 1), min(y1, SIZE - 1)
        ys, xs = np.mgrid[y0:y1 + 1, x0:x1 + 1]
        inside = Polygon(p).contains_points(np.stack([xs.ravel() + offset, ys.ravel() + offset], 1))
        mask[(ys.ravel()[inside]) * SIZE + xs.ravel()[inside]] = True
    return mask.reshape(SIZE, SIZE).astype(np.uint8)


KERNEL = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
RASTERISERS = {
    "pycocotools": pycoco,
    "pycoco_dilate1": lambda s: cv2.dilate(pycoco(s), KERNEL),
    "pycoco_erode1": lambda s: cv2.erode(pycoco(s), KERNEL),
    "cv2_round": lambda s: cv2_fill(s, "round"),
    "cv2_floor": lambda s: cv2_fill(s, "floor"),
    "cv2_subpixel": lambda s: cv2_fill(s, "subpixel"),
    "cv2_round_half": lambda s: cv2_fill(s, "round_half"),
    "pil_fill": lambda s: pil_fill(s, False),
    "pil_outline": lambda s: pil_fill(s, True),
    "mpl_centre": lambda s: mpl_fill(s, 0.5),
    "mpl_corner": lambda s: mpl_fill(s, 0.0),
}


def val_truth(annotations: Path) -> dict:
    """{file_name: [(image_id, [segmentation, ...])]} for the canonical validation split."""
    coco = json.loads(annotations.read_text())
    split = make_split(str(annotations)); assert_disjoint(split)
    val_ids = set(split.val_image_ids)
    record = {r["id"]: r for r in coco["images"]}
    by_id = defaultdict(list)
    for a in coco["annotations"]:
        if a["image_id"] in val_ids:
            by_id[a["image_id"]].append(a["segmentation"])
    out = defaultdict(list)
    for image_id in split.val_image_ids:
        out[record[image_id]["file_name"]].append((image_id, by_id[image_id]))
    return dict(out)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True, help="hypotheses.py's prediction cache")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    truth = val_truth(args.annotations)
    preds = json.loads(args.predictions.read_text())
    rows = {r: {m: [] for m in preds} for r in RASTERISERS}
    areas = {r: [] for r in RASTERISERS}
    for name in sorted(truth):
        mine = {m: [as_bytes(x) for x in preds[m][name]] for m in preds}
        for image_id, segmentations in sorted(truth[name]):
            for r, fn in RASTERISERS.items():
                masks = [fn(s) if isinstance(s, list) else mu.decode(mu.frPyObjects(s, SIZE, SIZE)) for s in segmentations]
                areas[r].extend(int(m.sum()) for m in masks)
                gt = [mu.encode(np.asfortranarray(m)) for m in masks if m.any()]
                for m in preds:
                    rows[r][m].append(match(mine[m], gt))
    table = {r: {m: total(v) for m, v in by.items()} for r, by in rows.items()}
    base = np.asarray(areas["pycocotools"], float)
    print("rasteriser".ljust(16), *(f"{m:>9}" for m in preds), "  area vs pycocotools", " public:", PUBLIC)
    for r in RASTERISERS:
        ratio = float(np.median(np.asarray(areas[r], float) / np.maximum(base, 1)))
        print(r.ljust(16), *(f"{table[r][m]['pq']:9.4f}" for m in preds), f"  x{ratio:.3f}")
    if args.output:
        args.output.write_text(json.dumps(dict(public=PUBLIC, table=table), indent=1))


if __name__ == "__main__":
    main()
