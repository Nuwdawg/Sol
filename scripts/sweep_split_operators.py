"""Why do real splitters recover only an eighth of the split ceiling?

exp114 established that gating is not the binding problem: with a perfect gate the
neck cut recovers +0.002846 of a +0.022342 ceiling, and watershed is net-negative
because it over-segments. Truth assignment makes 57 pieces from the same 28 masks
and wins 40 crossings with none lost; the neck cut makes 56 pieces and wins 22
while losing 6. The cut lands in the wrong place.

This script does two things the previous run could not:

1. **Diagnoses placement per mask.** For every oracle merge it reports how many
   truths the oracle split rescues, how many each operator rescues, and the best
   IoU each piece reaches against its truth. That separates "cut in the wrong
   place" from "cut in the right place but the pieces are still too poor".
2. **Sweeps the operator constants cheaply.** `build_population` costs about 229
   seconds and every earlier run repeated it, so the population is built once here
   and reused across every variant. Only the 28 oracle merges are touched, so each
   variant costs seconds.

Nothing here is deployable: the gate is the oracle throughout, by design, so that
operator quality is measured in isolation.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from pycocotools import mask as mu

from experiments.exp_033_inference.src.sweep import encode, records_for
from experiments.exp_113_ceilings.src.harness import (
    BASELINE_PQ, build_population, harm_report, score)
from experiments.exp_113_ceilings.src.interface_cut import interface_cut_split
from experiments.exp_113_ceilings.src.splitters import (
    apply_split_operator, neck_cut_split, watershed_split)

TOUCH = 0.05
CEILING = 0.022342


def oracle_merges(population):
    """Every prediction that truly covers two or more truths, with its record index."""
    found = []
    for index, record in enumerate(population):
        predicted, truths = record["predicted"], record["truths"]
        if not predicted or not truths:
            continue
        ious = np.asarray(mu.iou(predicted, truths, [0] * len(truths)))
        for p in range(len(predicted)):
            covered = [g for g in range(len(truths)) if ious[p, g] > TOUCH]
            if len(covered) >= 2:
                found.append(dict(record=index, prediction=p, covered=covered,
                                  best_before=[float(ious[:, g].max()) for g in covered]))
    return found


def diagnose(population, merges, operator, **kwargs):
    """Per-mask: how many truths the oracle rescues versus how many the operator does."""
    oracle_total, operator_total, rows = 0, 0, []
    for merge in merges:
        record = population[merge["record"]]
        mask = mu.decode(record["predicted"][merge["prediction"]])
        truths = record["truths"]
        oracle_hits = 0
        for g, before in zip(merge["covered"], merge["best_before"]):
            piece = mask & mu.decode(truths[g])
            if piece.sum() < 40:
                continue
            after = float(mu.iou([encode(piece)], [truths[g]], [0])[0][0])
            if after >= 0.5 > before:
                oracle_hits += 1
        parts = operator(mask, **kwargs)
        operator_hits, best_ious = 0, []
        if len(parts) >= 2:
            encoded = [encode(part) for part in parts]
            for g, before in zip(merge["covered"], merge["best_before"]):
                reached = max(float(mu.iou([part], [truths[g]], [0])[0][0]) for part in encoded)
                best_ious.append(round(reached, 3))
                if reached >= 0.5 > before:
                    operator_hits += 1
        oracle_total += oracle_hits
        operator_total += operator_hits
        rows.append(dict(pieces=len(parts), oracle_hits=oracle_hits,
                         operator_hits=operator_hits, best_ious=best_ious))
    return dict(oracle_rescues=oracle_total, operator_rescues=operator_total, rows=rows)


def evaluate_variant(population, baseline, operator, label, **kwargs):
    repaired, stats = apply_split_operator(population, operator, oracle_gate=True, **kwargs)
    after = score(repaired)
    harm = harm_report(population, repaired)
    delta = after["pq"] - baseline["pq"]
    print(f"  {label:38s} {delta:+.6f} ({100 * delta / CEILING:+6.1f}% of ceiling)  "
          f"pieces {stats['pieces_produced']:3d}  up {harm['crossings_up']:2d} "
          f"down {harm['crossings_down']:2d}")
    return dict(label=label, delta_pq=round(delta, 6), pieces=stats["pieces_produced"], **harm)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pool", type=Path,
                        default=Path("data/exp033_cached_tta/candidates_0.5_0.5.json"))
    parser.add_argument("--output", type=Path, default=Path("data/exp115_split_sweep"))
    args = parser.parse_args()
    cv2.setNumThreads(2)
    args.output.mkdir(parents=True, exist_ok=True)
    annotations = next(Path(".").rglob("MAGFiLO_1.0_Annotations_kaggle2026_train.json"))
    records, _ = records_for(annotations)
    pool = json.loads(args.pool.read_text())

    population = build_population(pool, records, 0.35, -1, 300)
    baseline = score(population)
    assert abs(baseline["pq"] - BASELINE_PQ) < 1e-9, baseline["pq"]
    merges = oracle_merges(population)
    print(f"baseline PQ {baseline['pq']:.10f}, oracle merges {len(merges)}, "
          f"ceiling +{CEILING:.6f}\n")

    print("placement diagnosis (oracle gate, per mask):")
    report = dict(baseline=baseline["pq"], merges=len(merges), variants=[], diagnosis={})
    for label, operator, kwargs in (("neck_cut", neck_cut_split, {}),
                                    ("watershed limit 2", watershed_split,
                                     dict(min_separation=8, floor_fraction=0.55)),
                                    ("interface_cut", interface_cut_split, {})):
        found = diagnose(population, merges, operator, **kwargs)
        report["diagnosis"][label] = {k: v for k, v in found.items() if k != "rows"}
        print(f"  {label:20s} oracle rescues {found['oracle_rescues']:2d}, "
              f"operator rescues {found['operator_rescues']:2d}")

    print("\nvariants (oracle gate throughout):")
    variants = [
        ("neck_cut (baseline)", neck_cut_split, {}),
        ("watershed limit 3 floor 0.45", watershed_split, {}),
        ("watershed limit 2 sep 8 floor 0.55", watershed_split,
         dict(limit=2, min_separation=8, floor_fraction=0.55)),
        ("watershed limit 2 sep 12 floor 0.65", watershed_split,
         dict(limit=2, min_separation=12, floor_fraction=0.65)),
        ("watershed limit 2 sep 16 floor 0.75", watershed_split,
         dict(limit=2, min_separation=16, floor_fraction=0.75)),
        ("interface_cut sep 12 floor 0.55", interface_cut_split, {}),
        ("interface_cut sep 8 floor 0.45", interface_cut_split,
         dict(min_separation=8, floor_fraction=0.45)),
        ("interface_cut sep 16 floor 0.65 thick 2", interface_cut_split,
         dict(min_separation=16, floor_fraction=0.65, thickness=2)),
    ]
    for label, operator, kwargs in variants:
        report["variants"].append(
            evaluate_variant(population, baseline, operator, label, **kwargs))

    (args.output / "report.json").write_text(json.dumps(report, indent=2))
    best = max(report["variants"], key=lambda row: row["delta_pq"])
    print(f"\nbest variant: {best['label']} at {best['delta_pq']:+.6f}, "
          f"{100 * best['delta_pq'] / CEILING:.1f}% of the ceiling")
    print("A variant that splits in the right place should approach the oracle's 40 crossings.")


if __name__ == "__main__":
    main()
