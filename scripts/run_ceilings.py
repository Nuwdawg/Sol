"""Run the oracle ceilings for the split and union repair families.

Appendix B, step 1: a repair family is worth a deployable trigger only once its
oracle ceiling is known to be material. This script answers that for the two
families whose operators need nothing but the masks already on disk.

Both operators read ground truth. That is deliberate and is what makes them
ceilings rather than postprocessors. Neither may ever be presented as deployable.

Before any ceiling is believed, the manual's reproducibility assertions run:
the baseline PQ is recomputed from source artifacts and compared with the frozen
value, and population counts are rebuilt rather than copied from earlier prose.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2

from experiments.exp_033_inference.src.sweep import records_for
from experiments.exp_113_ceilings.src.harness import (
    BASELINE_PQ, apply_split_ceiling, apply_union_ceiling, build_population,
    print_report, run_ceiling, score)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pool", type=Path,
                        default=Path("data/exp033_cached_tta/candidates_0.5_0.5.json"))
    parser.add_argument("--output", type=Path, default=Path("data/exp113_ceilings"))
    parser.add_argument("--conf", type=float, default=0.35)
    parser.add_argument("--grow", type=int, default=-1)
    parser.add_argument("--min-area", type=int, default=300)
    parser.add_argument("--tolerance", type=float, default=1e-9)
    args = parser.parse_args()
    cv2.setNumThreads(2)
    annotations = next(Path(".").rglob("MAGFiLO_1.0_Annotations_kaggle2026_train.json"))
    records, split = records_for(annotations)
    pool = json.loads(args.pool.read_text())

    population = build_population(pool, records, args.conf, args.grow, args.min_area)
    baseline = score(population)
    print(f"split           : {split.summary()}")
    print(f"records rebuilt : {len(population)}")
    print(f"baseline PQ     : {baseline['pq']:.10f}  "
          f"TP/FP/FN {baseline['tp']}/{baseline['fp']}/{baseline['fn']}")
    if args.grow == -1 and args.min_area == 300 and args.conf == 0.35:
        assert abs(baseline["pq"] - BASELINE_PQ) < args.tolerance, (
            f"baseline not reproduced: {baseline['pq']} against {BASELINE_PQ}")
        print("baseline assertion: PASS")

    reports = []
    for name, operator, notes in (
            ("split_by_truth", apply_split_ceiling,
             "Every prediction covering two or more truths is divided by truth assignment."),
            ("union_fragments", apply_union_ceiling,
             "Fragments overlapping one unmatched truth are replaced by their union.")):
        report = run_ceiling(name, population, operator, args.output, notes)
        print_report(report)
        reports.append(report)

    ranked = sorted(reports, key=lambda r: -r["official_metric"]["delta_pq"])
    print("\nceilings ranked by official delta PQ")
    for report in ranked:
        metric = report["official_metric"]
        print(f"  {report['method']:16s} {metric['delta_pq']:+.6f}  "
              f"(crossings up {report['harm']['crossings_up']}, "
              f"down {report['harm']['crossings_down']})")
    print("\nAppendix B step 1: a trivial ceiling stops that family on opportunity cost.")


if __name__ == "__main__":
    main()
