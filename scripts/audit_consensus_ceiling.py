"""Price the best single mask a per-record metric can reward.

Filaments drawn by two or more annotators are grouped by mask overlap. For each
group the script builds candidate single masks (each annotator's own mask, the
pixel majority, the union and the intersection) and reports the mean IoU each
scores against every record in that group. The winner bounds SQ for any model
that must emit one mask and be scored against each annotator separately.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from pycocotools import mask as mu

from experiments.exp_033_inference.src.sweep import encode, records_for


def group_filaments(passes):
    """Union-find over annotator masks that overlap above 0.5 IoU."""
    entries = [(index, rle) for index, (_, truth) in enumerate(passes) for rle in truth]
    if not entries:
        return []
    rles = [rle for _, rle in entries]
    table = np.asarray(mu.iou(rles, rles, [0] * len(rles)))
    parent = list(range(len(entries)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]; i = parent[i]
        return i

    for i in range(len(entries)):
        for j in range(i + 1, len(entries)):
            if entries[i][0] != entries[j][0] and table[i, j] > .5:
                parent[find(i)] = find(j)
    groups = {}
    for index in range(len(entries)):
        groups.setdefault(find(index), []).append(index)
    return [[entries[i] for i in members] for members in groups.values()
            if len({entries[i][0] for i in members}) > 1]


def mean_iou(candidate, members):
    others = [rle for _, rle in members]
    return float(np.mean(np.asarray(mu.iou([candidate], others, [0] * len(others)))))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    records, split = records_for(args.annotations)
    rows = []
    for name, passes in records.items():
        if len(passes) < 2:
            continue
        for members in group_filaments(passes):
            masks = np.stack([mu.decode(rle) for _, rle in members])
            majority = (masks.sum(axis=0) * 2 >= masks.shape[0]).astype(np.uint8)
            union = (masks.sum(axis=0) > 0).astype(np.uint8)
            intersection = (masks.sum(axis=0) == masks.shape[0]).astype(np.uint8)
            best_own = max(mean_iou(rle, members) for _, rle in members)
            rows.append(dict(
                annotators=len({index for index, _ in members}), masks=len(members),
                best_own=best_own,
                majority=mean_iou(encode(majority), members),
                union=mean_iou(encode(union), members),
                intersection=mean_iou(encode(intersection), members) if intersection.any() else 0.,
            ))
    summary = {key: float(np.mean([row[key] for row in rows]))
               for key in ("best_own", "majority", "union", "intersection")}
    report = dict(status="complete", split=split.summary(), groups=len(rows),
                  mean_iou_against_every_record=summary,
                  best_single_choice=max(summary, key=summary.get))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
