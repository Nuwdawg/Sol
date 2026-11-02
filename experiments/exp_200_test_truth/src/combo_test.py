"""exp200ad: the stacked file for the test photographs (combo.py's rule).

exp169's instances (score >= 0.68), plus the Mask2Former instances scoring in [0.60, 0.68) whose
Swin-B full-view partner scores >= 0.95, painted in descending score; an instance that keeps less
than 70% of its own vote mask is dropped; then exp199's guard re-adds large exp139 instances the
vote deleted. The CSV is written exactly as guard.py's test mode writes the deployed file.

Checks before the file is written:
1. with neither fix (fraction 0, no rescue) the test path rebuilds the submitted
   submission_exp169_guarded.csv byte for byte;
2. (optional) the same functions on validation reproduce combo.py's per-record PQ for this rule.

    python -m experiments.exp_200_test_truth.src.combo_test --ingredients test_ingredients.pkl
        --test-cache e040_test_cache.json --test-swinb shard0.jsonl shard1.jsonl --exp139-csv exp139_submission.csv
        --expect-guarded submission_exp169_guarded.csv --output dir
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import pickle
from pathlib import Path

from pycocotools import mask as mu

from experiments.exp_157_shift_views.src.views import merged_cache
from experiments.exp_168_joint_tuning.src.joint import scores
from experiments.exp_199_swinb_votes.src.guard import guard, read_csv
from experiments.exp_200_test_truth.src.hypotheses import as_bytes
from experiments.exp_200_test_truth.src.paint_order import ORDERS
from experiments.exp_200_test_truth.src.remnants import paint_fraction
from experiments.exp_200_test_truth.src.swinb_gate import support

FRACTION, RESCUE_LOW, RESCUE_TAU = 0.7, 0.60, 0.95


def build(ingredients: dict, s_b: dict, base: dict, fraction: float, rescue: bool) -> tuple[list, dict]:
    """Rows in guard.py's test-mode order and a small report."""
    stems, finals = [], {}
    for name in sorted(ingredients):
        p = ingredients[name]
        s = scores(p, 4.0, 0.3)
        kept = [(float(v), c) for v, b, c in zip(s, s_b.get(name, [0.0] * len(s)), p["crops"])
                if v >= 0.68 or (rescue and RESCUE_LOW <= v < 0.68 and b >= RESCUE_TAU)]
        final = [as_bytes(r) for r in paint_fraction(kept, ORDERS["score_desc"], fraction)]
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


def write(rows: list, path: Path) -> str:
    with open(path, "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["filament_id", "segmentation_rle"])
        writer.writerows(rows)
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--ingredients", type=Path, required=True, help="emit_test.py's cached exp169 test ingredients")
    parser.add_argument("--test-cache", type=Path, required=True)
    parser.add_argument("--test-swinb", type=Path, nargs="+", required=True)
    parser.add_argument("--exp139-csv", type=Path, required=True, help="exp139's submitted file, the guard's base")
    parser.add_argument("--expect-guarded", type=Path, required=True, help="the submitted exp169 + guard file")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    ingredients = pickle.loads(args.ingredients.read_bytes())
    if len(ingredients) != 180:
        raise SystemExit("expected 180 test photographs")
    base, _ = read_csv(args.exp139_csv)
    test = merged_cache(args.test_cache, args.test_cache)
    swinb = {}
    for path in args.test_swinb:
        for line in path.read_text().splitlines():
            if line.strip():
                item = json.loads(line)
                swinb[item["name"]] = item["full"]
    if sorted(swinb) != sorted(ingredients):
        raise SystemExit("Swin-B does not cover the test photographs")
    s_b = {n: support(test[n]["full"], [c for c in swinb[n] if c["p"] >= 0.5]) for n in sorted(ingredients)}

    rows, report = build(ingredients, s_b, base, 0.0, False)
    parity = write(rows, args.output / "exp169_guarded_rebuilt.csv")
    expected = hashlib.sha256(args.expect_guarded.read_bytes()).hexdigest().upper()
    print("PARITY", json.dumps(dict(report, sha256=parity, expected=expected)), flush=True)
    if parity != expected:
        raise SystemExit("the test path does not rebuild submission_exp169_guarded.csv")
    rows, report = build(ingredients, s_b, base, FRACTION, True)
    digest = write(rows, args.output / "submission_exp169_remnant_rescue.csv")
    print("WROTE", json.dumps(dict(report, sha256=digest)), flush=True)
    (args.output / "combo_test.json").write_text(json.dumps(dict(report, sha256=digest, parity=parity,
                                                                 rule=dict(fraction=FRACTION, rescue_low=RESCUE_LOW,
                                                                           rescue_tau=RESCUE_TAU)), indent=1))


if __name__ == "__main__":
    main()
