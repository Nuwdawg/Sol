"""Do real growth operators recover the growth ceilings, given a perfect gate?

exp116 closed the split family at the operator level: three cut geometries, each
under a perfect gate, recovered at best +0.002846 of a +0.022342 ceiling. The two
growth families still have ceilings around +0.018 each and, until now, no operator
written against them at all.

This runs the same protocol the splitters were held to:

1. recompute the oracle ceilings in-process, so the comparison uses numbers from
   this run rather than remembered constants;
2. run each implementable operator under the *same* oracle gate the ceiling used
   (strict 1:1 masks with truth IoU in [0.4, 0.5)), which isolates operator quality
   from gate quality and is never deployable;
3. include a matched control - uniform dilation, which reads no image evidence - so
   that any credit for absorption-guided growth is credit over the naive operator;
4. score every variant with the frozen official metric after global rematching, and
   report crossings up and down.

A variant that wins crossings while losing as many is worth nothing, which is why
harm is printed beside every delta.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import cv2

from experiments.exp_033_inference.src.sweep import records_for
from experiments.exp_113_ceilings.src.growth import (
    apply_axial_ceiling, apply_full_growth_ceiling, apply_lateral_ceiling)
from experiments.exp_113_ceilings.src.growth_operators import (
    apply_growth_operator, dark_band_grow, dark_grow_then_extend, dark_ray_extend,
    uniform_dilate)
from experiments.exp_113_ceilings.src.harness import (
    BASELINE_PQ, build_population, harm_report, score)

IMAGES = Path("data/competition/unpacked/MAGFiLO_1.0_Kaggle_2026/train/train_images")


def loader(directory: Path):
    def load(name: str):
        return cv2.imread(str(directory / name), cv2.IMREAD_GRAYSCALE)
    return load


def report_line(label: str, baseline: dict, after: dict, harm: dict, stats: dict,
                ceiling: float) -> dict:
    delta = after["pq"] - baseline["pq"]
    share = f"{100 * delta / ceiling:+6.1f}%" if ceiling else "     -"
    print(f"  {label:42s} {delta:+.6f} ({share} of ceiling)  "
          f"changed {stats.get('masks_changed', 0):3d}/{stats.get('masks_attempted', 0):3d}  "
          f"up {harm['crossings_up']:2d} down {harm['crossings_down']:2d}")
    return dict(label=label, delta_pq=round(delta, 6),
                share_of_ceiling=round(delta / ceiling, 4) if ceiling else None,
                **{k: v for k, v in stats.items() if k != "oracle_gate"}, **harm)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pool", type=Path,
                        default=Path("data/exp033_cached_tta/candidates_0.5_0.5.json"))
    parser.add_argument("--images", type=Path, default=IMAGES)
    parser.add_argument("--output", type=Path, default=Path("data/exp117_growth_operators"))
    parser.add_argument("--conf", type=float, default=0.35)
    parser.add_argument("--grow", type=int, default=-1)
    parser.add_argument("--min-area", type=int, default=300)
    args = parser.parse_args()
    cv2.setNumThreads(2)
    args.output.mkdir(parents=True, exist_ok=True)
    load = loader(args.images)

    annotations = next(Path(".").rglob("MAGFiLO_1.0_Annotations_kaggle2026_train.json"))
    records, _ = records_for(annotations)
    pool = json.loads(args.pool.read_text())

    started = time.time()
    population = build_population(pool, records, args.conf, args.grow, args.min_area)
    baseline = score(population)
    assert abs(baseline["pq"] - BASELINE_PQ) < 1e-9, baseline["pq"]
    print(f"baseline PQ {baseline['pq']:.10f}  TP/FP/FN "
          f"{baseline['tp']}/{baseline['fp']}/{baseline['fn']}  "
          f"({time.time() - started:.0f}s)\n")

    print("oracle ceilings, recomputed in this run:")
    ceilings = {}
    for name, operator in (("lateral", apply_lateral_ceiling),
                           ("axial", apply_axial_ceiling),
                           ("full_growth", apply_full_growth_ceiling)):
        repaired, stats = operator(population)
        after = score(repaired)
        harm = harm_report(population, repaired)
        ceilings[name] = round(after["pq"] - baseline["pq"], 6)
        print(f"  {name:12s} {ceilings[name]:+.6f}  modified {stats['masks_modified']:3d}  "
              f"up {harm['crossings_up']:2d} down {harm['crossings_down']:2d}")

    variants = [
        ("control: dilate 1 iteration", uniform_dilate, dict(iterations=1), "lateral"),
        ("control: dilate 2 iterations", uniform_dilate, dict(iterations=2), "lateral"),
        ("band grow b3 rel 0.55", dark_band_grow, dict(band=3, relative=0.55), "lateral"),
        ("band grow b2 rel 0.70", dark_band_grow, dict(band=2, relative=0.70), "lateral"),
        ("band grow b4 rel 0.40", dark_band_grow, dict(band=4, relative=0.40), "lateral"),
        ("band grow b6 rel 0.55 cap 1.0", dark_band_grow,
         dict(band=6, relative=0.55, max_growth=1.0), "lateral"),
        ("ray extend r24 rel 0.50", dark_ray_extend, dict(reach=24, relative=0.50), "axial"),
        ("ray extend r40 rel 0.35", dark_ray_extend, dict(reach=40, relative=0.35), "axial"),
        ("ray extend r40 rel 0.50 cone 25", dark_ray_extend,
         dict(reach=40, relative=0.50, cone=25.0), "axial"),
        # The probe length decides whether a correct direction is accepted at all,
        # so it is swept rather than left at the first value that worked.
        ("ray extend r40 rel 0.50 probe 10", dark_ray_extend,
         dict(reach=40, relative=0.50, probe=10), "axial"),
        ("ray extend r24 rel 0.50 probe 3", dark_ray_extend,
         dict(reach=24, relative=0.50, probe=3), "axial"),
        ("band+ray combined", dark_grow_then_extend, {}, "full_growth"),
    ]

    print("\noperators under the oracle gate (never deployable):")
    results = []
    for label, operator, kwargs, family in variants:
        repaired, stats = apply_growth_operator(
            population, load, operator, oracle_gate=True, **kwargs)
        row = report_line(label, baseline, score(repaired),
                          harm_report(population, repaired), stats, ceilings[family])
        results.append(dict(row, family=family, gate="oracle"))

    best = max(results, key=lambda row: row["delta_pq"])
    print(f"\nbest gated operator: {best['label']} at {best['delta_pq']:+.6f}")

    print("\nsame operators with no gate, on every strict 1:1 mask:")
    ungated = []
    for label, operator, kwargs, family in variants:
        if label != best["label"] and not label.startswith("control: dilate 1"):
            continue
        repaired, stats = apply_growth_operator(
            population, load, operator, oracle_gate=False, **kwargs)
        row = report_line(f"ungated: {label}", baseline, score(repaired),
                          harm_report(population, repaired), stats, ceilings[family])
        ungated.append(dict(row, family=family, gate="none"))

    payload = dict(experiment_id="exp117", baseline_pq=baseline["pq"], ceilings=ceilings,
                   oracle_gated=results, ungated=ungated,
                   note=("Oracle gate = strict 1:1 masks with truth IoU in [0.4, 0.5). "
                         "It reads ground truth and is an analysis device only."))
    (args.output / "report.json").write_text(json.dumps(payload, indent=2))
    print(f"\nwrote {args.output / 'report.json'} in {time.time() - started:.0f}s total")


if __name__ == "__main__":
    main()
