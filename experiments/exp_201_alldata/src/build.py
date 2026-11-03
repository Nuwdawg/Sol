"""exp201: turn retrained models' test caches into the remnant + rescue submission (CPU, minutes).

exp201 retrains exp139's Swin-S recipe and the Swin-B voter on all 707 labelled photographs, so no
held-out photograph is left to tune on. This builder therefore changes no constant: it runs the
deployed file's rule (final_decision/results/remnant_rescue/README.md) with a new Swin-S in exp139's
place and a new Swin-B in the old Swin-B's place:
  1. exp169's ingredients (exp168's `prepared`): vote masks over the Swin-S full, seven dihedral and
     three shift views plus the two YOLO pools' partners, and the view scores;
  2. the guard's base: the Swin-S full view alone at exp139's deployed rule (score >= 0.85, area 100);
  3. Swin-B support: the best Swin-B full-view partner (class probability >= 0.5, IoU > 0.5);
  4. combo_test.build: keep score >= 0.68, or [0.60, 0.68) with Swin-B support >= 0.95; paint in
     descending score; drop instances keeping < 70% of their vote mask; guard.
Every function is the one that wrote the deployed file. Run on exp139 e40's test cache and the old
Swin-B shards, it must rebuild submission_exp169_remnant_rescue.csv byte for byte (--expect-sha);
that rehearsal is the check that the path is the deployed path (CLAUDE.md section 5).

    python -m experiments.exp_201_alldata.src.build --swins-cache model_e050_test_cache.json
        --swinb model_e032_test_cache.json --yolo yolov8l=test_pool.json --yolo yolo11m=test_pool.json
        --output dir [--expect-sha F9423B74...]
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from experiments.exp_152_rule_and_emit.src.emit import write_csv
from experiments.exp_157_shift_views.src.views import merged_cache, rule_pool
from experiments.exp_159_yolo_votes.src.votes import load_pool
from experiments.exp_168_joint_tuning.src.joint import prepared
from experiments.exp_199_swinb_votes.src.guard import read_csv
from experiments.exp_200_test_truth.src.combo_test import FRACTION, build, write
from experiments.exp_200_test_truth.src.swinb_gate import support


def load_swinb(paths: list[Path]) -> dict:
    """Swin-B full-view candidates per photograph from jsonl shards ({"name", "full"} per line)
    or from exp147-layout caches ({"photographs": {name: {"full": [...]}}})."""
    out = {}
    for path in paths:
        text = Path(path).read_text()
        if path.suffix == ".jsonl":
            items = [json.loads(line) for line in text.splitlines() if line.strip()]
            pairs = [(item["name"], item["full"]) for item in items]
        else:
            pairs = [(name, entry["full"]) for name, entry in json.loads(text)["photographs"].items()]
        for name, full in pairs:
            if name in out:
                raise SystemExit(f"{name} appears twice in the Swin-B inputs")
            out[name] = full
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--swins-cache", type=Path, required=True, help="Swin-S test cache with full, d4 and shift")
    parser.add_argument("--swinb", type=Path, nargs="+", required=True, help="Swin-B full-view test candidates")
    parser.add_argument("--yolo", action="append", default=[], help="tag=path of a YOLO test pool")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expect-sha", default="", help="fail unless the file has this SHA256 (rehearsal)")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    test = merged_cache(args.swins_cache, args.swins_cache)
    if len(test) != 180:
        raise SystemExit(f"expected 180 test photographs, got {len(test)}")
    missing = {mode: sum(1 for e in test.values() if mode not in e) for mode in ("full", "d4", "shift")}
    if any(missing.values()):
        raise SystemExit(f"views missing from the Swin-S cache: {missing}")
    views = {mode: sorted({c["view"] for e in test.values() for c in e[mode]}) for mode in ("d4", "shift")}
    if views != {"d4": list(range(1, 8)), "shift": [1, 2, 3]}:
        raise SystemExit(f"unexpected views {views}")
    pools = {t: load_pool(Path(p)) for t, p in (s.split("=", 1) for s in args.yolo)}
    tags = sorted(pools)
    if tags != ["yolo11m", "yolov8l"]:
        raise SystemExit(f"expected the yolo11m and yolov8l pools, got {tags}")
    swinb = load_swinb(args.swinb)
    if sorted(swinb) != sorted(test):
        raise SystemExit("Swin-B does not cover the same 180 photographs")
    model = json.loads(args.swins_cache.read_text()).get("model", {})

    ingredients = {n: prepared(test[n], [pools[t].get(n, []) for t in tags], ("d4", "shift")) for n in sorted(test)}
    single = write_csv({n: rule_pool(test[n], (), None, False) for n in test}, 0.85, 100,
                       args.output / "single_view.csv")
    base, _ = read_csv(args.output / "single_view.csv")
    s_b = {n: support(test[n]["full"], [c for c in swinb[n] if c["p"] >= 0.5]) for n in sorted(ingredients)}

    rows, guarded = build(ingredients, s_b, base, 0.0, False)
    guarded["sha256"] = write(rows, args.output / "exp169_guarded.csv")
    rows, report = build(ingredients, s_b, base, FRACTION, True)
    report["sha256"] = write(rows, args.output / "submission_remnant_rescue.csv")
    summary = dict(model=model, single_view=single, exp169_guarded=guarded, remnant_rescue=report,
                   inputs=dict(swins_cache=str(args.swins_cache), swinb=[str(p) for p in args.swinb],
                               yolo=sorted(args.yolo)))
    print("BUILT", json.dumps(dict(single_view=single["rows"], exp169_guarded=guarded["rows"],
                                   remnant_rescue=report["rows"], sha256=report["sha256"])), flush=True)
    if args.expect_sha:
        summary["expected"] = args.expect_sha.upper()
        summary["identical"] = report["sha256"] == args.expect_sha.upper()
        print("EXPECT", json.dumps(dict(identical=summary["identical"], sha256=report["sha256"])), flush=True)
    (args.output / "build.json").write_text(json.dumps(summary, indent=1))
    if args.expect_sha and not summary["identical"]:
        raise SystemExit("the build does not rebuild the expected file")


if __name__ == "__main__":
    main()
