"""Oracle ceilings for the growth repairs: axial continuation and lateral boundary.

Completes Appendix B step 1 across the four repair families. exp113 measured the
split (+0.022342) and union (+0.011579) ceilings; this measures what perfect
growth of existing strict 1:1 instances could recover, separated into the two
populations exp108 identified - about 59% of the missing area axial, 41% lateral.

Three operators, all reading ground truth and therefore upper bounds only:

- `growth_axial`: add missing truth beyond the mask's own extent, within 30 px.
  The most a perfect endpoint continuation could recover, width included.
- `growth_lateral`: add missing truth within 4 px of the existing boundary. The
  most a perfect narrow-band method (active contour, graph cut) could recover.
- `growth_full`: add every missing truth pixel of near-threshold 1:1 masks, the
  absolute bound the other two are read against.

None of these is a postprocessor. A real operator must also decide whether to act
and where to stop, without truth.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2

from experiments.exp_033_inference.src.sweep import records_for
from experiments.exp_113_ceilings.src.growth import (
    apply_axial_ceiling, apply_full_growth_ceiling, apply_lateral_ceiling)
from experiments.exp_113_ceilings.src.harness import (
    BASELINE_PQ, build_population, print_report, run_ceiling, score)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pool", type=Path,
                        default=Path("data/exp033_cached_tta/candidates_0.5_0.5.json"))
    parser.add_argument("--output", type=Path, default=Path("data/exp113_ceilings"))
    parser.add_argument("--conf", type=float, default=0.35)
    parser.add_argument("--grow", type=int, default=-1)
    parser.add_argument("--min-area", type=int, default=300)
    parser.add_argument("--reach", type=int, default=30)
    parser.add_argument("--band", type=int, default=4)
    args = parser.parse_args()
    cv2.setNumThreads(2)
    annotations = next(Path(".").rglob("MAGFiLO_1.0_Annotations_kaggle2026_train.json"))
    records, split = records_for(annotations)
    pool = json.loads(args.pool.read_text())

    population = build_population(pool, records, args.conf, args.grow, args.min_area)
    baseline = score(population)
    print(f"split           : {split.summary()}")
    print(f"baseline PQ     : {baseline['pq']:.10f}  "
          f"TP/FP/FN {baseline['tp']}/{baseline['fp']}/{baseline['fn']}")
    if args.grow == -1 and args.min_area == 300 and args.conf == 0.35:
        assert abs(baseline["pq"] - BASELINE_PQ) < 1e-9, baseline["pq"]
        print("baseline assertion: PASS")

    for name, operator, notes in (
            ("growth_axial", lambda p: apply_axial_ceiling(p, reach=args.reach),
             f"Perfect endpoint continuation: missing truth beyond the mask extent, "
             f"within {args.reach} px."),
            ("growth_lateral", lambda p: apply_lateral_ceiling(p, band=args.band),
             f"Perfect narrow-band boundary correction: missing truth within "
             f"{args.band} px of the boundary."),
            ("growth_full", apply_full_growth_ceiling,
             "Absolute bound: every missing truth pixel of near-threshold 1:1 masks.")):
        print_report(run_ceiling(name, population, operator, args.output, notes))

    print("\nall measured ceilings, ranked by official delta PQ")
    rows = []
    for path in sorted(args.output.glob("*_ceiling.json")):
        data = json.loads(path.read_text())
        rows.append((data["official_metric"]["delta_pq"], data["method"], data["harm"]))
    for delta, method, harm in sorted(rows, reverse=True):
        print(f"  {method:16s} {delta:+.6f}  crossings up {harm['crossings_up']}, "
              f"down {harm['crossings_down']}")
    print("\nA ceiling is an upper bound on a repair family, never a submission candidate.")


if __name__ == "__main__":
    main()
