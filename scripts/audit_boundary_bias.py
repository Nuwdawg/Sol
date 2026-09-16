"""Ask whether the retained masks are systematically too small or too large.

Segmentation quality is the open lever, so the direction of the error matters.
For every matched (prediction, annotation record) pair this reports precision,
recall and the area ratio, then prices symmetric morphological corrections
applied to the emitted masks only. Instances are never added or removed.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from pycocotools import mask as mu

from experiments.exp_033_inference.src.sweep import encode, pq_rle, records_for
from shared.utils import aggregate_pq

BASELINE_PQ = 0.4558638687292126
KERNEL = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))


def morph(rle: dict, steps: int) -> dict:
    if steps == 0:
        return rle
    mask = mu.decode(rle)
    mask = (cv2.dilate if steps > 0 else cv2.erode)(mask, KERNEL, iterations=abs(steps))
    return encode(mask)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rich", type=Path, required=True)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    cv2.setNumThreads(1)
    rich = json.loads(args.rich.read_text())
    records, split = records_for(args.annotations)
    emitted = {name: [row["rle"] for row in rows] for name, rows in rich.items()}

    precision, recall, ratio = [], [], []
    for name, passes in records.items():
        predicted = emitted[name]
        areas = [float(mu.area(rle)) for rle in predicted]
        for _, truth in passes:
            if not predicted or not truth:
                continue
            table = np.asarray(mu.iou(predicted, truth, [0] * len(truth)))
            used_p, used_g = set(), set()
            for p, g in sorted(np.argwhere(table > .5), key=lambda k: -table[k[0], k[1]]):
                if p in used_p or g in used_g:
                    continue
                used_p.add(int(p)); used_g.add(int(g))
                intersection = float(mu.area(mu.merge([predicted[p], truth[g]], intersect=True)))
                truth_area = float(mu.area(truth[g]))
                precision.append(intersection / areas[p])
                recall.append(intersection / truth_area)
                ratio.append(areas[p] / truth_area)

    # Price uniform morphological corrections with the emission otherwise fixed.
    corrections = []
    for steps in (-1, 0, 1, 2, 3):
        metrics = []
        for name, passes in records.items():
            adjusted = [morph(rle, steps) for rle in emitted[name]]
            metrics.extend(pq_rle(adjusted, truth) for _, truth in passes)
        corrections.append(dict(steps=steps, **aggregate_pq(metrics)))

    report = dict(
        status="complete", split=split.summary(), matched=len(precision),
        precision=dict(mean=float(np.mean(precision)), median=float(np.median(precision))),
        recall=dict(mean=float(np.mean(recall)), median=float(np.median(recall))),
        area_ratio=dict(mean=float(np.mean(ratio)), median=float(np.median(ratio)),
                        below_one=int((np.asarray(ratio) < 1).sum())),
        corrections=corrections,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
