"""Measure inter-annotator mask IoU on the canonical validation photographs.

This calibrates how much boundary refinement is reachable at all. It compares
annotation records of the same photograph against each other. It never merges
annotators and never changes the official per-record metric.
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path

import numpy as np
from pycocotools import mask as mu

from experiments.exp_033_inference.src.sweep import records_for


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    records, split = records_for(args.annotations)
    paired, photos = [], 0
    for name, passes in records.items():
        if len(passes) < 2:
            continue
        photos += 1
        for i in range(len(passes)):
            for j in range(i + 1, len(passes)):
                a, b = passes[i][1], passes[j][1]
                if not a or not b:
                    continue
                table = np.asarray(mu.iou(a, b, [0] * len(b)))
                used_a, used_b = set(), set()
                for p, g in sorted(np.argwhere(table > .5), key=lambda k: -table[k[0], k[1]]):
                    if p not in used_a and g not in used_b:
                        used_a.add(int(p)); used_b.add(int(g))
                        paired.append(float(table[p, g]))
    counts = Counter()
    for value in paired:
        counts[f"{np.floor(value * 10) / 10:.1f}"] += 1
    report = dict(status="complete", split=split.summary(), photographs=photos,
                  matched_pairs=len(paired),
                  mean=float(np.mean(paired)), median=float(np.median(paired)),
                  p10=float(np.percentile(paired, 10)), p90=float(np.percentile(paired, 90)),
                  histogram=dict(sorted(counts.items())))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
