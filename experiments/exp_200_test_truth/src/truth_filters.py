"""exp200c: would a filtered or coarser test truth explain the leaderboard gap?

The training annotations carry four categories (1 Left, 2 Right, 3 Unidentifiable, 4 Ambiguous),
an area and a polygon. The test truth is an RLE per filament with no category. If the organisers
built it from a subset of the annotations (some categories, or only filaments above an area), or
rasterised it at a lower resolution, every model would lose about the same amount on test. That is
the gap's signature. This scores cached validation predictions against each such truth.

    python -m experiments.exp_200_test_truth.src.truth_filters --annotations ... --predictions ...
"""
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np
from pycocotools import mask as mu

from experiments.exp_200_test_truth.src.hypotheses import PUBLIC, as_bytes, match, total
from shared.data_split import assert_disjoint, make_split

SIZE = 2048


def coarse(segmentation, factor: int) -> np.ndarray:
    """The polygon rasterised at 2048/factor and enlarged back by nearest neighbour."""
    small = SIZE // factor
    scaled = [[v / factor for v in p] for p in segmentation]
    mask = mu.decode(mu.merge(mu.frPyObjects(scaled, small, small)))
    return cv2.resize(mask, (SIZE, SIZE), interpolation=cv2.INTER_NEAREST)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    coco = json.loads(args.annotations.read_text())
    split = make_split(str(args.annotations)); assert_disjoint(split)
    val_ids = set(split.val_image_ids)
    record = {r["id"]: r for r in coco["images"]}
    print("categories, all annotations:", sorted(Counter(a["category_id"] for a in coco["annotations"]).items()))
    by_id = defaultdict(list)
    for a in coco["annotations"]:
        if a["image_id"] in val_ids:
            by_id[a["image_id"]].append(a)
    preds = json.loads(args.predictions.read_text())
    variants = {
        "all (validation as scored)": lambda a: True,
        "Left/Right only (1, 2)": lambda a: a["category_id"] in (1, 2),
        "without Ambiguous (4)": lambda a: a["category_id"] != 4,
        "without Unidentifiable (3)": lambda a: a["category_id"] != 3,
        "area >= 300": lambda a: a["area"] >= 300,
        "area >= 1000": lambda a: a["area"] >= 1000,
    }
    rows = {v: {m: [] for m in preds} for v in variants}
    rows.update({f"rasterised at {SIZE // f}": {m: [] for m in preds} for f in (2, 4)})
    for image_id in split.val_image_ids:
        name = record[image_id]["file_name"]
        mine = {m: [as_bytes(x) for x in preds[m][name]] for m in preds}
        rles = {id(a): mu.merge(mu.frPyObjects(a["segmentation"], SIZE, SIZE)) for a in by_id[image_id]}
        for v, keep in variants.items():
            truth = [rles[id(a)] for a in by_id[image_id] if keep(a)]
            for m in preds:
                rows[v][m].append(match(mine[m], truth))
        for f in (2, 4):
            truth = [mu.encode(np.asfortranarray(coarse(a["segmentation"], f))) for a in by_id[image_id]]
            for m in preds:
                rows[f"rasterised at {SIZE // f}"][m].append(match(mine[m], truth))
    table = {v: {m: total(r) for m, r in by.items()} for v, by in rows.items()}
    print("truth".ljust(28), *(f"{m:>9}" for m in preds), " public:", PUBLIC)
    for v in table:
        print(v.ljust(28), *(f"{table[v][m]['pq']:9.4f}" for m in preds))
    if args.output:
        args.output.write_text(json.dumps(dict(public=PUBLIC, table=table), indent=1))


if __name__ == "__main__":
    main()
