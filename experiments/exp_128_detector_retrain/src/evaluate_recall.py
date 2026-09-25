"""The pre-registered measurement harness: proposal recall first, PQ last.

`docs/PREREGISTRATION.md` fixes the metrics that decide whether a retrain arm
continues or is abandoned, and PQ is deliberately not the first of them. PQ
conflates localization, instance decomposition and boundary estimation, which is
how a whole line of work came to be aimed at the wrong stage: exp129 concluded the
error was "entirely detection" from a figure produced downstream of the very rule
that was discarding the evidence.

So this reports, in this order:

1. **proposal recall at IoU 0.5** - the share of validation truths with *any*
   candidate above 0.5, before confidence, erosion, area floor or painting. This is
   the detector's localization ability with the emission rule taken out of the way.
2. **the bucket histogram** in the pipeline's own representation - matchable /
   lost to the confidence floor / boundary failure / weak proposal / never proposed.
3. **matched IoU at native 2048 resolution**. exp094's crop refiner appeared to gain
   +0.013 and actually lost 0.002 because it was scored at crop resolution against a
   native baseline; that mistake is not repeated.
4. **PQ**, as the summary.

Input is a D4 pool JSON, `{photograph: [{score, rle}]}`, exactly what
`evaluate_checkpoints.py` already writes. Reusing that seam means no fresh inference
code, so the thing under test is the detector rather than a new harness.

Frozen comparison points, from the current detector:

| quantity | value |
|---|---|
| proposal recall, raw | 76.7% |
| proposal recall, eroded | 79.4% |
| proposal recall, eroded + conf 0.35 | 66.4% |
| matchable / conf-lost / boundary / weak / absent | 880 / 172 / 165 / 45 / 63 |
| PQ | 0.4558638687292126 |

Arm A must land within 2 points of the recall figures or the harness is measuring
something other than the change under test.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from pycocotools import mask as mu

from experiments.exp_033_inference.src.sweep import encode, pq_rle, records_for
from shared.utils import aggregate_pq, paint_panoptic

KERNEL = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
BASELINE = dict(raw=0.767, eroded=0.794, eroded_conf=0.664, pq=0.4558638687292126,
                buckets=dict(matchable=880, confidence_floor=172, boundary=165,
                             weak=45, absent=63))
TOLERANCE = 0.02


def load_pool(path: Path) -> dict[str, list[dict]]:
    pool = json.loads(Path(path).read_text())
    cleaned = {}
    for photograph, entries in pool.items():
        rows = []
        for entry in entries:
            rle = entry["rle"]
            counts = rle["counts"]
            rows.append(dict(score=float(entry["score"]),
                             rle=dict(size=rle["size"],
                                      counts=counts.decode("ascii")
                                      if isinstance(counts, bytes) else counts)))
        cleaned[Path(photograph).name] = rows
    return cleaned


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pool", type=Path, required=True)
    parser.add_argument("--annotations", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--label", default="arm")
    parser.add_argument("--conf", type=float, default=0.35)
    parser.add_argument("--grow", type=int, default=-1)
    parser.add_argument("--min-area", type=int, default=300)
    args = parser.parse_args()
    cv2.setNumThreads(2)
    args.output.mkdir(parents=True, exist_ok=True)
    records, split = records_for(args.annotations)
    pool = load_pool(args.pool)

    missing = [name for name in records if name not in pool]
    if missing:
        raise SystemExit(f"pool misses {len(missing)} validation photographs: {missing[:3]}")

    raw_best, ero_best, conf_best, rows = [], [], [], []
    for photograph, passes in records.items():
        entries = pool[photograph]
        scores = np.array([e["score"] for e in entries]) if entries else np.zeros(0)
        raw = [e["rle"] for e in entries]
        eroded, kept = [], []
        for entry in entries:
            mask = mu.decode(entry["rle"])
            if args.grow < 0:
                mask = cv2.erode(mask, KERNEL, iterations=-args.grow)
            eroded.append(encode(mask))
            if entry["score"] >= args.conf and int(mask.sum()) >= args.min_area:
                kept.append((entry["score"], mask))
        emitted = [encode(m) for _, m, _ in paint_panoptic(kept, min_area=args.min_area)]
        above = scores >= args.conf
        for _, truths in passes:
            if not truths:
                continue
            n = len(truths)
            if not entries:
                raw_best += [0.0] * n
                ero_best += [0.0] * n
                conf_best += [0.0] * n
            else:
                ir = np.asarray(mu.iou(raw, truths, [0] * n))
                ie = np.asarray(mu.iou(eroded, truths, [0] * n))
                raw_best += ir.max(axis=0).tolist()
                ero_best += ie.max(axis=0).tolist()
                conf_best += (ie[above].max(axis=0).tolist() if above.any()
                              else [0.0] * n)
            rows.append(pq_rle(emitted, truths))

    raw_best = np.array(raw_best)
    ero_best = np.array(ero_best)
    conf_best = np.array(conf_best)
    aggregate = aggregate_pq(rows)

    recall = dict(raw=float(np.mean(raw_best >= 0.5)),
                  eroded=float(np.mean(ero_best >= 0.5)),
                  eroded_conf=float(np.mean(conf_best >= 0.5)))
    matchable = conf_best >= 0.5
    confidence_floor = (ero_best >= 0.5) & ~matchable
    rest = ~matchable & ~confidence_floor
    buckets = dict(matchable=int(matchable.sum()),
                   confidence_floor=int(confidence_floor.sum()),
                   boundary=int((rest & (ero_best >= 0.3)).sum()),
                   weak=int((rest & (ero_best >= 0.1) & (ero_best < 0.3)).sum()),
                   absent=int((rest & (ero_best < 0.1)).sum()))

    print(f"=== {args.label} ===  truths {len(raw_best)}   split: {split.summary()}\n")
    print("1. PROPOSAL RECALL at IoU 0.5 (the primary metric)")
    for key in ("raw", "eroded", "eroded_conf"):
        delta = recall[key] - BASELINE[key]
        flag = "OK" if abs(delta) <= TOLERANCE else ("UP" if delta > 0 else "DOWN")
        print(f"   {key:12} {recall[key]:6.1%}   baseline {BASELINE[key]:6.1%}   "
              f"{delta:+.1%}  {flag}")
    print("\n2. BUCKETS, in the pipeline's representation")
    for key, value in buckets.items():
        print(f"   {key:18} {value:5d}   baseline {BASELINE['buckets'][key]:5d}   "
              f"{value - BASELINE['buckets'][key]:+5d}")
    print("\n3. MATCHED IoU at native 2048 resolution")
    print(f"   SQ {aggregate['sq']:.4f}   mean matched IoU over {aggregate['tp']} TPs")
    print("\n4. PQ (summary only)")
    print(f"   PQ {aggregate['pq']:.6f}   baseline {BASELINE['pq']:.6f}   "
          f"{aggregate['pq'] - BASELINE['pq']:+.6f}")
    print(f"   TP/FP/FN {aggregate['tp']}/{aggregate['fp']}/{aggregate['fn']}")

    report = dict(label=args.label, recall=recall, buckets=buckets,
                  rule=dict(conf=args.conf, grow=args.grow, min_area=args.min_area),
                  official={k: aggregate[k] for k in ("pq", "sq", "rq", "tp", "fp", "fn")},
                  baseline=BASELINE, split=split.summary())
    (args.output / f"{args.label}_recall.json").write_text(json.dumps(report, indent=2))
    print(f"\nwrote {args.output / f'{args.label}_recall.json'}")


if __name__ == "__main__":
    main()
