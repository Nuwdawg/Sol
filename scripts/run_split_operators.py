"""Do real splitters recover the +0.022342 split ceiling, given a perfect gate?

Appendix B step 4: rasterize the repair and score it officially. exp113 measured
the ceiling by assigning pixels with ground truth; this measures what implementable
operators achieve when told exactly which masks to split.

Three configurations per operator:

1. oracle gate - only masks that truly cover two or more truths. Isolates operator
   quality. Never deployable.
2. every mask - the same operator with no gate, which is where an absent gate does
   its damage.

The comparison that matters is operator-under-oracle-gate against the +0.022342
truth-assignment ceiling. A low number there means the operator is weak and no
gate rescues it. A high number means only the gate is missing.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2

from experiments.exp_033_inference.src.sweep import records_for
from experiments.exp_113_ceilings.src.harness import (
    BASELINE_PQ, build_population, print_report, run_ceiling, score)
from experiments.exp_113_ceilings.src.splitters import (
    apply_split_operator, neck_cut_split, watershed_split)

CEILING = 0.022342      # exp113 split_by_truth, the oracle upper bound


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pool", type=Path,
                        default=Path("data/exp033_cached_tta/candidates_0.5_0.5.json"))
    parser.add_argument("--output", type=Path, default=Path("data/exp114_split_operators"))
    parser.add_argument("--conf", type=float, default=0.35)
    parser.add_argument("--grow", type=int, default=-1)
    parser.add_argument("--min-area", type=int, default=300)
    args = parser.parse_args()
    cv2.setNumThreads(2)
    annotations = next(Path(".").rglob("MAGFiLO_1.0_Annotations_kaggle2026_train.json"))
    records, split = records_for(annotations)
    pool = json.loads(args.pool.read_text())

    population = build_population(pool, records, args.conf, args.grow, args.min_area)
    baseline = score(population)
    print(f"baseline PQ : {baseline['pq']:.10f}  "
          f"TP/FP/FN {baseline['tp']}/{baseline['fp']}/{baseline['fn']}")
    assert abs(baseline["pq"] - BASELINE_PQ) < 1e-9, baseline["pq"]
    print(f"ceiling     : +{CEILING:.6f} (truth-assignment split)\n")

    results = []
    for name, operator, gated in (
            ("watershed_oracle_gate", watershed_split, True),
            ("neck_cut_oracle_gate", neck_cut_split, True),
            ("watershed_every_mask", watershed_split, False),
            ("neck_cut_every_mask", neck_cut_split, False)):
        report = run_ceiling(
            name, population,
            lambda p, op=operator, g=gated: apply_split_operator(p, op, oracle_gate=g),
            args.output,
            notes=("oracle gate: applied only to masks truly covering 2+ truths"
                   if gated else "no gate: applied to every prediction"))
        print_report(report)
        delta = report["official_metric"]["delta_pq"]
        results.append((name, delta, report["operator"], report["harm"]))
        print(f"  share of ceiling: {100 * delta / CEILING:+.1f}%")

    print("\nsummary")
    for name, delta, operator, harm in results:
        print(f"  {name:24s} {delta:+.6f}  "
              f"attempted {operator['masks_attempted']:4d} split {operator['masks_split']:3d} "
              f"pieces {operator['pieces_produced']:4d}  "
              f"crossings up {harm['crossings_up']:2d} down {harm['crossings_down']:2d}")
    print("\nattempted-but-not-split means the operator declined; split with a low delta")
    print("means it split badly. Those require different fixes.")


if __name__ == "__main__":
    main()
