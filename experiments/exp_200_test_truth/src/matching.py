"""exp200d: would the scorer's matching or averaging explain the leaderboard gap?

hypotheses.py assumed the organisers' notebook: every pair above IoU 0.5 is a true positive, counts
pooled over images. Kaggle's own metric code is private. Two plausible differences:

  pooled_1to1     all annotators' filaments in one image (overlapping), but each truth and each
                  prediction matched at most once (greedy by IoU, as a standard PQ does). Every
                  extra annotator's version of a filament the prediction already matched is a miss.
  photo_mean      PQ per photograph (records pooled within it), averaged over photographs.
  pooled_mean     the same average with pooled truth.
  record_mean     PQ per annotation record, averaged.

    python -m experiments.exp_200_test_truth.src.matching --annotations ... --predictions ...
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from pycocotools import mask as mu

from experiments.exp_033_inference.src.sweep import records_for
from experiments.exp_200_test_truth.src.hypotheses import PUBLIC, as_bytes, match, total


def one_to_one(predictions: list[dict], truth: list[dict]) -> dict:
    if not truth or not predictions:
        return dict(tp=0, fp=len(predictions), fn=len(truth), iou_sum=0.0)
    iou = np.asarray(mu.iou(predictions, truth, [0] * len(truth))).reshape(len(predictions), len(truth))
    pairs = sorted(((iou[i, j], i, j) for i, j in zip(*np.nonzero(iou > 0.5))), reverse=True)
    used_p, used_t, tp, iou_sum = set(), set(), 0, 0.0
    for value, i, j in pairs:
        if i in used_p or j in used_t:
            continue
        used_p.add(i); used_t.add(j); tp += 1; iou_sum += float(value)
    return dict(tp=tp, iou_sum=iou_sum, fp=len(predictions) - tp, fn=len(truth) - tp)


def mean_pq(groups: list[list[dict]]) -> float:
    values = []
    for rows in groups:
        t = total(rows)
        if t["tp"] + t["fp"] + t["fn"]:
            values.append(t["pq"])
    return float(np.mean(values))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    records, _ = records_for(args.annotations)
    preds = json.loads(args.predictions.read_text())
    table = {}
    for m in preds:
        record_rows, pooled_rows, pooled_1to1, photo_groups, pooled_groups = [], [], [], [], []
        for name in sorted(records):
            mine = [as_bytes(x) for x in preds[m][name]]
            passes = [[as_bytes(t) for t in truth] for _, truth in sorted(records[name])]
            rows = [match(mine, truth) for truth in passes]
            record_rows.extend(rows)
            photo_groups.append(rows)
            pooled = [t for truth in passes for t in truth]
            pooled_rows.append(match(mine, pooled))
            pooled_groups.append([pooled_rows[-1]])
            pooled_1to1.append(one_to_one(mine, pooled))
        table[m] = dict(record=total(record_rows)["pq"], pooled_all_pairs=total(pooled_rows)["pq"],
                        pooled_1to1=total(pooled_1to1)["pq"], photo_mean=mean_pq(photo_groups),
                        pooled_mean=mean_pq(pooled_groups), record_mean=mean_pq([[r] for r in record_rows]),
                        pooled_1to1_counts={k: v for k, v in total(pooled_1to1).items() if k in ("tp", "fp", "fn")})
    keys = ["record", "pooled_all_pairs", "pooled_1to1", "photo_mean", "pooled_mean", "record_mean"]
    print("scoring".ljust(18), *(f"{m:>9}" for m in preds), " public:", PUBLIC)
    for k in keys:
        print(k.ljust(18), *(f"{table[m][k]:9.4f}" for m in preds))
    for m in preds:
        print(m, "pooled_1to1 counts", table[m]["pooled_1to1_counts"])
    if args.output:
        args.output.write_text(json.dumps(dict(public=PUBLIC, table=table), indent=1))


if __name__ == "__main__":
    main()
