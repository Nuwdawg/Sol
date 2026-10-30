"""exp200g: exp169's rule family on the test photographs at other settings, from one cached pass.

The rule's ingredients (exp168's `prepared`: vote masks, view scores, YOLO support) are computed once,
exactly as exp169's builder does, and cached. A setting is YOLO weight : exponent : threshold : area.
The first file written is exp169's own setting (4 : 0.3 : 0.68 : 100), and it must be byte-identical
to the submitted file (SHA256 CC15EA41...), so every other file is the same rule with only its
constants moved (CLAUDE.md section 5). Each setting is written plain and with exp199's guard.

    python -m experiments.exp_200_test_truth.src.emit_test --test-cache e040_test_cache.json
        --yolo yolov8l=... --yolo yolo11m=... --base-csv exp139_submission.csv
        --ingredients test_ingredients.pkl --point 0:0.3:0.76:100 --output dir
"""
from __future__ import annotations

import argparse
import json
import pickle
import subprocess
import sys
from pathlib import Path

from experiments.exp_152_rule_and_emit.src.emit import write_csv
from experiments.exp_157_shift_views.src.views import merged_cache
from experiments.exp_159_yolo_votes.src.votes import load_pool
from experiments.exp_168_joint_tuning.src.joint import prepared, scores

EXP169_SHA256 = "CC15EA4124E56F8AA01DA9270FE140B0830EA754833722696244C60073008106"


def test_ingredients(args) -> dict:
    if args.ingredients.exists():
        return pickle.loads(args.ingredients.read_bytes())
    test = merged_cache(args.test_cache, args.test_cache)
    pools = {t: load_pool(Path(p)) for t, p in (s.split("=", 1) for s in args.yolo)}
    tags = sorted(pools)
    if tags != ["yolo11m", "yolov8l"]:
        raise SystemExit(f"expected the yolo11m and yolov8l pools, got {tags}")
    out = {n: prepared(test[n], [pools[t].get(n, []) for t in tags], ("d4", "shift")) for n in sorted(test)}
    args.ingredients.write_bytes(pickle.dumps(out))
    return out


def entries(ingredients: dict, weight: float, exponent: float) -> dict:
    return {n: list(zip(map(float, scores(p, weight, exponent)), p["crops"])) for n, p in ingredients.items()}


def guarded(base: Path, final: Path, output: Path) -> dict:
    subprocess.run([sys.executable, "-m", "experiments.exp_199_swinb_votes.src.guard", "test", "--base", str(base),
                    "--final", str(final), "--output", str(output)], check=True, stdout=subprocess.DEVNULL)
    return json.loads((output / "guard_test_report.json").read_text())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--test-cache", type=Path, required=True, help="Mask2Former test cache with full, d4 and shift")
    parser.add_argument("--yolo", action="append", default=[])
    parser.add_argument("--base-csv", type=Path, required=True, help="exp139's submitted file, the guard's base")
    parser.add_argument("--ingredients", type=Path, required=True)
    parser.add_argument("--point", action="append", default=[], help="weight:exponent:conf:min_area")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    ingredients = test_ingredients(args)
    if len(ingredients) != 180:
        raise SystemExit(f"expected 180 test photographs, got {len(ingredients)}")
    parity = write_csv(entries(ingredients, 4.0, 0.3), 0.68, 100, args.output / "exp169_rebuilt.csv")
    print("PARITY", json.dumps(parity), flush=True)
    if parity["sha256"] != EXP169_SHA256:
        raise SystemExit("exp169's operating point does not rebuild the submitted file")
    for point in args.point:
        weight, exponent, conf, area = (float(v) for v in point.split(":"))
        area = int(area)
        tag = f"w{weight:g}_e{exponent:g}_c{conf:.2f}_a{area}"
        facts = write_csv(entries(ingredients, weight, exponent), conf, area, args.output / f"exp169_{tag}.csv")
        guard = guarded(args.base_csv, args.output / f"exp169_{tag}.csv", args.output / f"guard_{tag}")
        target = args.output / f"exp169_{tag}_guarded.csv"
        (args.output / f"guard_{tag}" / "submission_guarded.csv").rename(target)
        print("POINT", json.dumps(dict(plain=facts, guarded=dict(path=target.name, rows=guard["rows"],
                                                                  added=guard["added"], sha256=guard["sha256"]))),
              flush=True)


if __name__ == "__main__":
    main()
