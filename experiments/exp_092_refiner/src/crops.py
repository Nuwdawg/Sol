"""Build (proposal, annotation record) crop pairs for boundary refinement.

The competition scores every prediction against one annotation record at a time,
so one training example is created per (proposal, record) pair rather than per
proposal. A model fitted to that set regresses toward the mask that maximises
expected IoU over annotators, which the consensus measurement in
`docs/ACTIVE_EXPERIMENT.md` put at IoU 0.809 against every record, well above
the 0.684 the current emission achieves.

Nothing here perturbs ground truth. The proposals are real detections from the
frozen exp002 detector under the same eight dihedral views used at inference.
"""
from __future__ import annotations

from collections import defaultdict
import json
from pathlib import Path

import cv2
import numpy as np
from pycocotools import mask as mu

from experiments.exp_085_morden_spine.src.prepare import image_channels

NATIVE = 2048


def window(mask: np.ndarray, size: int, context: float) -> tuple[int, int, int]:
    """A square window around the proposal, sized by its longest side."""
    ys, xs = np.nonzero(mask)
    if not len(xs):
        return 0, 0, NATIVE
    height = int(ys.max() - ys.min()) + 1
    width = int(xs.max() - xs.min()) + 1
    side = int(np.clip(round(max(height, width) * context), size // 4, NATIVE))
    x0 = int(np.clip((int(xs.max()) + int(xs.min()) - side) // 2, 0, NATIVE - side))
    y0 = int(np.clip((int(ys.max()) + int(ys.min()) - side) // 2, 0, NATIVE - side))
    return x0, y0, side


def resample(plane: np.ndarray, box: tuple[int, int, int], size: int,
             interpolation: int) -> np.ndarray:
    x0, y0, side = box
    return cv2.resize(plane[y0:y0 + side, x0:x0 + side], (size, size),
                      interpolation=interpolation)


def channels_for(image: np.ndarray) -> np.ndarray:
    """The exp085 solar channels, computed once at native resolution."""
    return image_channels(image, NATIVE)


def pair_records(proposals: list[dict], truths: list[list[dict]], floor: float):
    """Yield (proposal index, truth rle, iou) for every record that matches."""
    rles = [entry["rle"] for entry in proposals]
    for record in truths:
        if not rles or not record:
            continue
        table = np.asarray(mu.iou(rles, record, [0] * len(record)))
        for index in range(len(rles)):
            best = int(table[index].argmax())
            score = float(table[index, best])
            if score >= floor:
                yield index, record[best], score


def build_examples(image: np.ndarray, proposals: list[dict], truths: list[list[dict]],
                   size: int, context: float, floor: float) -> list[dict]:
    planes = channels_for(image)
    decoded = [mu.decode(entry["rle"]) for entry in proposals]
    examples = []
    for index, truth, score in pair_records(proposals, truths, floor):
        box = window(decoded[index], size, context)
        features = np.stack([resample(plane, box, size, cv2.INTER_AREA) for plane in planes]
                            + [resample(decoded[index] * 255, box, size, cv2.INTER_NEAREST)])
        target = resample(mu.decode(truth) * 255, box, size, cv2.INTER_NEAREST)
        examples.append(dict(features=features.astype(np.uint8),
                             target=(target > 127).astype(np.uint8),
                             seed_iou=score, box=box,
                             score=float(proposals[index].get("score", 0.))))
    return examples


def truths_by_photo(annotations: Path, stems: set[str]) -> dict[str, list[list[dict]]]:
    coco = json.loads(Path(annotations).read_text(encoding="utf-8"))
    record = {row["id"]: row for row in coco["images"]}
    grouped = defaultdict(list)
    for annotation in coco["annotations"]:
        row = record[annotation["image_id"]]
        if row["file_name"] not in stems:
            continue
        segmentation = annotation["segmentation"]
        if isinstance(segmentation, list):
            rle = mu.merge(mu.frPyObjects(segmentation, row["height"], row["width"]))
        elif isinstance(segmentation["counts"], list):
            rle = mu.frPyObjects(segmentation, row["height"], row["width"])
        else:
            rle = segmentation
        grouped[(row["file_name"], annotation["image_id"])].append(rle)
    by_photo = defaultdict(list)
    for (name, _), rles in grouped.items():
        by_photo[name].append(rles)
    return dict(by_photo)
