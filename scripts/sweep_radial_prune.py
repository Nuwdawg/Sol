"""Sweep a limb-aware pruning threshold on the exp085 semantic evidence.

The native review of the flat rule showed its errors are spatial: it correctly
deletes limb-edge artifacts, but it also deletes genuine low-contrast filaments
near the limb, where limb darkening depresses the semantic probability. So the
threshold is allowed to relax with radius instead of staying flat.

Pruning never repaints a mask, so the instance-versus-record IoU table is
computed once and every rule is scored by re-running the greedy match over the
retained rows. That makes an exact canonical sweep cheap.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from pycocotools import mask as mu

from experiments.exp_033_inference.src.sweep import records_for

BASELINE_PQ = 0.4558638687292126
CENTRE = 1024.0


def radius_of(rle: dict) -> float:
    ys, xs = np.nonzero(mu.decode(rle))
    return float(np.hypot(xs.mean() - CENTRE, ys.mean() - CENTRE) / CENTRE)


def score(tables, keep) -> dict:
    tp = fp = fn = 0
    iou_sum = 0.
    for photo, records in tables:
        rows = [i for i in range(len(keep[photo])) if keep[photo][i]]
        for table in records:
            truth_count = table.shape[1]
            if not rows or not truth_count:
                fp += len(rows); fn += truth_count
                continue
            sub = table[rows]
            used_p, used_g = set(), set()
            for p, g in sorted(np.argwhere(sub > .5), key=lambda k: -sub[k[0], k[1]]):
                if p in used_p or g in used_g:
                    continue
                used_p.add(int(p)); used_g.add(int(g)); iou_sum += float(sub[p, g])
            tp += len(used_p); fp += len(rows) - len(used_p); fn += truth_count - len(used_g)
    sq = iou_sum / tp if tp else 0.
    rq = tp / (tp + .5 * fp + .5 * fn) if (tp + fp + fn) else 0.
    return dict(pq=sq * rq, sq=sq, rq=rq, tp=tp, fp=fp, fn=fn)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--feature", default="semantic_p75")
    args = parser.parse_args()
    features = json.loads(args.features.read_text())
    records, split = records_for(args.annotations)

    tables, radii, values, counts = [], {}, {}, 0
    for name in sorted(features):
        rows = features[name]
        rles = [row["rle"] for row in rows]
        radii[name] = np.asarray([radius_of(rle) for rle in rles])
        values[name] = np.asarray([row[args.feature] for row in rows])
        counts += len(rows)
        per_record = []
        for _, truth in records[name]:
            if rles and truth:
                per_record.append(np.asarray(mu.iou(rles, truth, [0] * len(truth))))
            else:
                per_record.append(np.zeros((len(rles), len(truth))))
        tables.append((name, per_record))
    print(f"cached {counts} instances over {len(tables)} photographs", flush=True)

    keep_all = {name: np.ones(len(features[name]), bool) for name in features}
    baseline = score(tables, keep_all)
    assert abs(baseline["pq"] - BASELINE_PQ) < 1e-9, baseline
    print("baseline reproduced", json.dumps(baseline), flush=True)

    results = []
    for split_radius in (.55, .65, .75, .85, 1.01):
        for inner in (.10, .15, .20, .25, .30, .40, .50):
            for outer in (0., .02, .05, .08, .10, .15):
                if outer > inner:
                    continue
                keep = {name: np.where(radii[name] < split_radius,
                                       values[name] >= inner, values[name] >= outer)
                        for name in features}
                retained = int(sum(k.sum() for k in keep.values()))
                results.append(dict(split_radius=split_radius, inner=inner, outer=outer,
                                    retained=retained, **score(tables, keep)))
    results.sort(key=lambda row: -row["pq"])
    flat = [row for row in results if row["split_radius"] >= 1.0]
    report = dict(status="complete", split=split.summary(), feature=args.feature,
                  instances=counts, baseline=baseline, best=results[0],
                  best_flat=flat[0], top=results[:15])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2))
    print(json.dumps(dict(best=results[0], best_flat=flat[0]), indent=2), flush=True)


if __name__ == "__main__":
    main()
