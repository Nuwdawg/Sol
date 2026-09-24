"""Audit precision-oriented pruning of accepted D4 masks without retraining."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from pycocotools import mask as mu

from experiments.exp_033_inference.src.cached_tta import vote_rle
from experiments.exp_033_inference.src.sweep import encode, pq_rle, records_for
from shared.utils import aggregate_pq


def fuse_rich(entries: list, link: float = .5, vote: float = .5) -> list[dict]:
    order = sorted(range(len(entries)), key=lambda i: -entries[i][0])
    if not order:
        return []
    ious = mu.iou([entry[1] for entry in entries], [entry[1] for entry in entries],
                  [0] * len(entries))
    used, output = set(), []
    for seed in order:
        if seed in used:
            continue
        members = [seed]
        used.add(seed)
        for index in order:
            if index not in used and ious[seed, index] >= link:
                members.append(index)
                used.add(index)
        rle = vote_rle([entries[index][1] for index in members],
                       max(1, round(vote * len(members))))
        if mu.area(rle) < 40:
            continue
        confidences = [float(entries[index][0]) for index in members]
        output.append(dict(
            score=float(np.mean(confidences)) * (.5 + .5 * len(members) / 8),
            rle=rle,
            views=len(members),
            mean_confidence=float(np.mean(confidences)),
            max_confidence=max(confidences),
            min_confidence=min(confidences),
        ))
    return output


def emit_rich(entries: list[dict]) -> list[dict]:
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    claimed = np.zeros((2048, 2048), np.uint8)
    output = []
    for entry in sorted((row for row in entries if row["score"] >= .35),
                        key=lambda row: -row["score"]):
        mask = cv2.erode(mu.decode(entry["rle"]), kernel)
        remaining = (mask.astype(np.uint8) & (1 - claimed)).astype(np.uint8)
        area = int(remaining.sum())
        if area < 300:
            continue
        claimed |= remaining
        output.append(dict(entry, rle=encode(remaining), area=area))
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--views", type=Path, required=True)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    cv2.setNumThreads(1)
    views = json.loads(args.views.read_text())
    expected = json.loads(args.baseline.read_text())
    records, split = records_for(args.annotations)
    assert set(views) == set(records) == set(expected)

    rich = {name: emit_rich(fuse_rich(entries)) for name, entries in views.items()}
    for name in rich:
        assert [row["rle"] for row in rich[name]] == expected[name], name

    def score(rule) -> dict:
        metrics, retained = [], 0
        for name, passes in records.items():
            predicted = [row["rle"] for row in rich[name]
                         if row["views"] >= rule["views"]
                         and row["mean_confidence"] >= rule["mean_confidence"]]
            retained += len(predicted)
            metrics.extend(pq_rle(predicted, truth) for _, truth in passes)
        return dict(retained=retained, **aggregate_pq(metrics))

    rules = []
    for minimum_views in range(1, 9):
        for mean_confidence in (.25, .30, .35, .40, .45, .50, .55, .60):
            rule = dict(views=minimum_views, mean_confidence=mean_confidence)
            rules.append(dict(rule=rule, **score(rule)))
    rules.sort(key=lambda row: row["pq"], reverse=True)

    groups = []
    for view_count in range(1, 9):
        rows = [row for entries in rich.values() for row in entries
                if row["views"] == view_count]
        groups.append(dict(views=view_count, count=len(rows),
                           mean_score=float(np.mean([row["score"] for row in rows])) if rows else None,
                           mean_area=float(np.mean([row["area"] for row in rows])) if rows else None))
    report = dict(status="complete", split=split.summary(), best=rules[0],
                  top=rules[:20], groups=groups)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2))
    args.output.with_name("rich_candidates.json").write_text(json.dumps(rich))
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
