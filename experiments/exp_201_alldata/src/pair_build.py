"""exp201c: the two-model file (exp139 + a second Swin-S voting through all 11 views), test photographs.

transfer.py's `pair22` rule on validation, applied unchanged to the test photographs: exp139 e40's
full-view candidates, voted on by exp139's own 11 views and the second model's 11 views (scores and
partner masks), plus the two YOLO pools; then the remnant + rescue rule (threshold, rescue band,
remnant filter 0.7, guard on exp139's single-view file). The YOLO weight and the threshold are
arguments: they are whatever validation chose (transfer.json), never re-tuned here.

Checks: with --other set to exp139's own cache and --weight 4 --threshold 0.68 --self-check, the
22-view path must differ from the deployed file (it double-counts), and with --no-pair it must
rebuild submission_exp169_remnant_rescue.csv byte for byte.

    python -m experiments.exp_201_alldata.src.pair_build --cache e040_test_cache.json
        --other SUB0_test_cache.json SUB1_test_cache.json --swinb shard0.jsonl shard1.jsonl
        --yolo yolov8l=... --yolo yolo11m=... --weight 8 --threshold 0.68 --output dir
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from experiments.exp_152_rule_and_emit.src.emit import write_csv
from experiments.exp_157_shift_views.src.views import merged_cache, rule_pool
from experiments.exp_159_yolo_votes.src.votes import load_pool
from experiments.exp_168_joint_tuning.src.joint import prepared, scores
from experiments.exp_199_swinb_votes.src.guard import guard, read_csv
from experiments.exp_200_test_truth.src.combo_test import write
from experiments.exp_200_test_truth.src.hypotheses import as_bytes
from experiments.exp_200_test_truth.src.paint_order import ORDERS
from experiments.exp_200_test_truth.src.remnants import paint_fraction
from experiments.exp_200_test_truth.src.swinb_gate import support
from experiments.exp_201_alldata.src.build import load_swinb
from experiments.exp_201_alldata.src.transfer import pair_ingredients

BAND, TAU, FRACTION = 0.08, 0.95, 0.7


def rows_for(ingredients: dict, s_b: dict, base: dict, weight: float, threshold: float) -> tuple[list, dict]:
    """combo_test.build with the YOLO weight and threshold as parameters (same row order)."""
    from pycocotools import mask as mu
    stems, finals = [], {}
    for name in sorted(ingredients):
        p = ingredients[name]
        s = scores(p, weight, 0.3)
        kept = [(float(v), c) for v, b, c in zip(s, s_b[name], p["crops"])
                if v >= threshold or (threshold - BAND <= v < threshold and b >= TAU)]
        final = [as_bytes(r) for r in paint_fraction(kept, ORDERS["score_desc"], FRACTION)]
        stem = Path(name).stem
        if final:
            stems.append(stem)
        finals[stem] = final
    stems += [st for st in base if not finals.get(st)]
    rows, added = [], 0
    for stem in stems:
        kept = finals.get(stem, [])
        extra = guard(base.get(stem, []), kept)
        added += len(extra)
        masks = kept + extra
        if len(masks) > 1 and int(mu.area(mu.merge(masks, intersect=False))) != sum(int(mu.area(m)) for m in masks):
            raise SystemExit(f"{stem}: overlapping masks")
        for k, rle in enumerate(masks, start=1):
            rows.append((f"{stem}_{k}", rle["counts"].decode("ascii") if isinstance(rle["counts"], bytes) else rle["counts"]))
    return rows, dict(rows=len(rows), guard_added=added, photographs_with_rows=len({r[0].rsplit("_", 1)[0] for r in rows}))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--cache", type=Path, required=True, help="exp139 e40 test cache (full, d4, shift)")
    parser.add_argument("--other", type=Path, nargs="+", required=True, help="second model's test caches")
    parser.add_argument("--swinb", type=Path, nargs="+", required=True)
    parser.add_argument("--yolo", action="append", default=[])
    parser.add_argument("--weight", type=float, required=True)
    parser.add_argument("--threshold", type=float, required=True)
    parser.add_argument("--no-pair", action="store_true", help="exp139 alone (the deployed rule): parity check")
    parser.add_argument("--expect-sha", default="")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    test = merged_cache(args.cache, args.cache)
    if len(test) != 180:
        raise SystemExit("expected 180 test photographs")
    other = {}
    for path in args.other:
        for name, entry in json.loads(path.read_text())["photographs"].items():
            if name in other:
                raise SystemExit(f"{name} twice in --other")
            other[name] = entry
    if sorted(other) != sorted(test):
        raise SystemExit(f"--other covers {len(other)} of 180 photographs")
    pools = {t: load_pool(Path(p)) for t, p in (s.split("=", 1) for s in args.yolo)}
    tags = sorted(pools)
    if tags != ["yolo11m", "yolov8l"]:
        raise SystemExit(f"expected the yolo11m and yolov8l pools, got {tags}")
    swinb = load_swinb(args.swinb)
    names = sorted(test)
    if args.no_pair:
        ingredients = {n: prepared(test[n], [pools[t].get(n, []) for t in tags], ("d4", "shift")) for n in names}
    else:
        ingredients = {n: pair_ingredients(test[n], other[n], [pools[t].get(n, []) for t in tags]) for n in names}
    write_csv({n: rule_pool(test[n], (), None, False) for n in test}, 0.85, 100, args.output / "single_view.csv")
    base, _ = read_csv(args.output / "single_view.csv")
    s_b = {n: support(test[n]["full"], [c for c in swinb[n] if c["p"] >= 0.5]) for n in names}
    rows, report = rows_for(ingredients, s_b, base, args.weight, args.threshold)
    tag = "exp139_alone" if args.no_pair else f"pair22_w{args.weight:g}_t{args.threshold:.2f}"
    report["sha256"] = write(rows, args.output / f"submission_{tag}.csv")
    report.update(rule=dict(weight=args.weight, threshold=args.threshold, pair=not args.no_pair,
                            band=BAND, tau=TAU, fraction=FRACTION))
    print("BUILT", json.dumps(report), flush=True)
    (args.output / f"{tag}.json").write_text(json.dumps(report, indent=1))
    if args.expect_sha and report["sha256"] != args.expect_sha.upper():
        raise SystemExit("the build does not rebuild the expected file")


if __name__ == "__main__":
    main()
