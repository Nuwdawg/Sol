"""Apply the exp086 limb-aware semantic pruning rule to an existing submission.

The rule removes instances only. It never adds a mask, never edits a boundary
and never repaints, so the surviving rows are byte-identical to the submission
that scored 0.39 in public.

The same rule function is first replayed over the stored validation features and
must reproduce the tuned canonical PQ exactly. A deployment that cannot
reproduce its own validation number is refused before it writes a CSV.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
from pycocotools import mask as mu

from experiments.exp_033_inference.src.sweep import records_for

NATIVE = 2048
CENTRE = 1024.0
BASELINE_PQ = 0.4558638687292126
VALIDATED_PQ = 0.4606350484956118


def keeps(radius: float, value: float, split_radius: float, inner: float,
          outer: float) -> bool:
    """One rule, used by both the validation replay and the deployment."""
    return value >= (inner if radius < split_radius else outer)


def radius_of(rle: dict) -> float:
    ys, xs = np.nonzero(mu.decode(rle))
    return float(np.hypot(xs.mean() - CENTRE, ys.mean() - CENTRE) / CENTRE)


def replay_validation(features: dict, annotations: Path, rule: dict) -> dict:
    records, _ = records_for(annotations)
    tp = fp = fn = 0
    iou_sum = 0.0
    removed = 0
    for name, rows in features.items():
        retained, retained_rles = [], []
        for row in rows:
            radius = row.get("radius")
            if radius is None:
                radius = radius_of(row["rle"])
            if keeps(radius, row["semantic_p75"], **rule):
                retained.append(row); retained_rles.append(row["rle"])
            else:
                removed += 1
        for _, truth in records[name]:
            if not retained_rles or not truth:
                fp += len(retained_rles); fn += len(truth); continue
            table = np.asarray(mu.iou(retained_rles, truth, [0] * len(truth)))
            used_p, used_g = set(), set()
            for p, g in sorted(np.argwhere(table > .5), key=lambda k: -table[k[0], k[1]]):
                if p in used_p or g in used_g:
                    continue
                used_p.add(int(p)); used_g.add(int(g)); iou_sum += float(table[p, g])
            tp += len(used_p); fp += len(retained_rles) - len(used_p)
            fn += len(truth) - len(used_g)
    sq = iou_sum / tp if tp else 0.
    rq = tp / (tp + .5 * fp + .5 * fn) if (tp + fp + fn) else 0.
    return dict(pq=sq * rq, sq=sq, rq=rq, tp=tp, fp=fp, fn=fn, removed=removed)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--submission", type=Path, required=True)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--validation-features", type=Path, required=True)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--split-radius", type=float, default=0.65)
    parser.add_argument("--inner", type=float, default=0.30)
    parser.add_argument("--outer", type=float, default=0.12)
    args = parser.parse_args()
    rule = dict(split_radius=args.split_radius, inner=args.inner, outer=args.outer)

    replay = replay_validation(json.loads(args.validation_features.read_text()),
                               args.annotations, rule)
    print("validation replay", json.dumps(replay), flush=True)
    assert abs(replay["pq"] - VALIDATED_PQ) < 1e-9, (
        f"rule does not reproduce its tuned PQ: {replay['pq']} vs {VALIDATED_PQ}")

    features = json.loads(args.features.read_text())
    by_id = {row["filament_id"]: row for rows in features.values() for row in rows}
    rows = list(csv.DictReader(args.submission.open(newline="")))
    missing = [row["filament_id"] for row in rows if row["filament_id"] not in by_id]
    assert not missing, f"{len(missing)} submission rows have no features: {missing[:5]}"

    kept, dropped = [], []
    for row in rows:
        feature = by_id[row["filament_id"]]
        (kept if keeps(feature["radius"], feature["semantic_p75"], **rule)
         else dropped).append(row)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["filament_id", "segmentation_rle"])
        writer.writeheader()
        for row in kept:
            writer.writerow(row)

    images = defaultdict(int)
    for row in kept:
        images[row["filament_id"].rsplit("_", 1)[0]] += 1
    digest = hashlib.sha256(args.output.read_bytes()).hexdigest().upper()
    report = dict(status="complete", rule=rule, validation_replay=replay,
                  baseline_pq=BASELINE_PQ, validated_pq=VALIDATED_PQ,
                  input_rows=len(rows), kept_rows=len(kept), dropped_rows=len(dropped),
                  contributing_images=len(images), sha256=digest,
                  dropped=[dict(filament_id=row["filament_id"],
                                area=by_id[row["filament_id"]]["area"],
                                radius=by_id[row["filament_id"]]["radius"],
                                semantic_p75=by_id[row["filament_id"]]["semantic_p75"])
                           for row in dropped])
    args.output.with_suffix(".report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps({k: v for k, v in report.items() if k != "dropped"}, indent=2), flush=True)


if __name__ == "__main__":
    main()
