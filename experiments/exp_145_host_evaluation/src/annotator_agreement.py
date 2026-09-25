"""How well do two expert annotators agree, under the host's own matching rule?

296 photographs carry two or three independent annotation records. Each ordered pair
(a, b) with a != b scores annotator a's masks as the prediction against annotator b's
record as the ground truth, at native 2048 resolution, and all pairs are pooled into
one PQ, exactly as the host pools annotator-images:

    TP = every (gt, prediction) pair with IoU > 0.5
    FP = predictions that hit no ground truth, FN = ground truth hit by no prediction
    PQ = sum(IoU of TP) / (TP + FP/2 + FN/2)

IoU comes from pycocotools on the RLEs, the same pixel counts the notebook's dense
matrices produce, so the numbers are the notebook's without its memory cost.
This is a reference for how variable the labels are, not a ceiling on the score.

    python -m experiments.exp_145_host_evaluation.src.annotator_agreement \
        --annotations data/MAGFiLO_1.0_Annotations_kaggle2026_train.json --output agreement.json
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from pycocotools import mask as mu

from shared.data_split import SPLIT_SEED
from experiments.exp_033_inference.src.sweep import records_for


def all_records(annotations: Path) -> dict:
    """{file_name: [(image_id, [rle, ...]), ...]} over the whole file, every record."""
    coco = json.loads(annotations.read_text())
    images = {image["id"]: image for image in coco["images"]}
    by_image = defaultdict(list)
    for annotation in coco["annotations"]:
        image = images[annotation["image_id"]]
        segmentation = annotation["segmentation"]
        if isinstance(segmentation, list):
            rle = mu.merge(mu.frPyObjects(segmentation, image["height"], image["width"]))
        elif isinstance(segmentation["counts"], list):
            rle = mu.frPyObjects(segmentation, image["height"], image["width"])
        else:
            rle = segmentation
        by_image[annotation["image_id"]].append(rle)
    by_photo = defaultdict(list)
    for image_id, image in images.items():
        by_photo[image["file_name"]].append((image_id, by_image.get(image_id, [])))
    return dict(by_photo)


def pooled_agreement(records: dict) -> dict:
    tp_iou, fp, fn, pairs = [], 0, 0, 0
    for name, annotators in records.items():
        if len(annotators) < 2:
            continue
        for a, (_, predicted) in enumerate(annotators):
            for b, (_, truth) in enumerate(annotators):
                if a == b:
                    continue
                pairs += 1
                if not truth:
                    fp += len(predicted)
                    continue
                if not predicted:
                    fn += len(truth)
                    continue
                iou = np.asarray(mu.iou(list(predicted), list(truth), [0] * len(truth))).T
                hit = iou > 0.5
                tp_iou.extend(iou[hit].tolist())
                fp += int((hit.sum(axis=0) == 0).sum())
                fn += int((hit.sum(axis=1) == 0).sum())
    tp = len(tp_iou)
    denominator = tp + 0.5 * fp + 0.5 * fn
    return dict(pairs=pairs, tp=tp, fp=fp, fn=fn,
                pq=float(np.sum(tp_iou)) / denominator if denominator else 0.0,
                sq=float(np.mean(tp_iou)) if tp_iou else 0.0,
                rq=tp / denominator if denominator else 0.0)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    everything = all_records(args.annotations)
    validation, _ = records_for(args.annotations)
    counts = defaultdict(int)
    for annotators in everything.values():
        counts[len(annotators)] += 1
    report = dict(split_seed=SPLIT_SEED,
                  photographs_by_annotator_count={str(k): v for k, v in sorted(counts.items())},
                  all_photographs=pooled_agreement(everything),
                  validation_photographs=pooled_agreement(validation))
    args.output.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
