"""exp169: write the test file for the configuration exp168 adopted, after re-checking it on validation.

Reads exp168's report. If its cross-validated joint PQ beats the cross-validated fixed rule by at least
+0.002 (the pre-registered bar), the in-sample best configuration is rebuilt on validation (its PQ must
equal exp168's), then applied unchanged to test with exp168's threshold. Checks: exp139's deployed rule
rebuilds the submitted file (parity), and every emitted instance is non-empty and disjoint.

    python -m experiments.exp_169_joint_submission.src.joint_submit --report exp168_report.json
        --val-cache ... --val-shift ... --test-cache ... --test-shift ... --val-yolo ... --test-yolo ...
        --test-views-002 ... --base-csv ... --annotations ... --output ...
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from experiments.exp_033_inference.src.sweep import records_for
from experiments.exp_149_evidence_fusion.src import evidence as E
from experiments.exp_152_rule_and_emit.src.emit import PARITY_ROWS, compare_csv, write_csv
from experiments.exp_157_shift_views.src.views import merged_cache, rule_pool
from experiments.exp_159_yolo_votes.src.votes import load_pool
from experiments.exp_162_final_submission.src.final import fused_views
from experiments.exp_168_joint_tuning.src.joint import prepared, scores

BAR = 0.002


def parse(key: str) -> tuple[tuple[str, ...], float, float]:
    views, weight, exponent = key.split("_")
    return tuple(views.split("+")), float(weight[1:]), float(exponent[1:])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--val-cache", type=Path, required=True)
    parser.add_argument("--val-shift", type=Path, required=True)
    parser.add_argument("--test-cache", type=Path, required=True)
    parser.add_argument("--test-shift", type=Path, required=True)
    parser.add_argument("--val-yolo", action="append", default=[])
    parser.add_argument("--test-yolo", action="append", default=[])
    parser.add_argument("--test-views-002", type=Path, nargs="*", default=[])
    parser.add_argument("--base-csv", type=Path, required=True)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    tuning = json.loads(args.report.read_text())
    gain = tuning["joint"]["cross_validated"] - tuning["fixed"]["cross_validated"]
    decision = dict(cross_validated_gain=gain, bar=BAR, adopted=gain >= BAR, key=tuning["joint"]["in_sample_best"])
    print("DECISION", json.dumps(decision), flush=True)
    if not decision["adopted"]:
        (args.output / "exp169_report.json").write_text(json.dumps(decision, indent=1))
        return
    sources, weight, exponent = parse(decision["key"])
    conf = tuning["joint"]["in_sample"]["conf"]
    records, _ = records_for(args.annotations)
    val = merged_cache(args.val_cache, args.val_shift)
    test = merged_cache(args.test_cache, args.test_shift)
    val_pools = {t: load_pool(Path(p)) for t, p in (s.split("=", 1) for s in args.val_yolo)}
    test_pools = {t: load_pool(Path(p)) for t, p in (s.split("=", 1) for s in args.test_yolo)}
    if args.test_views_002:
        test_pools["yolo11m"] = fused_views(args.test_views_002)
    tags = sorted(val_pools)
    if sorted(test_pools) != tags:
        raise SystemExit(f"pools differ: {tags} vs {sorted(test_pools)}")

    def pool(cache, pools, names):
        out = {}
        for n in names:
            p = prepared(cache[n], [pools[t].get(n, []) for t in tags], sources)
            out[n] = list(zip(map(float, scores(p, weight, exponent)), p["crops"]))
        return out

    rows, _ = E.score_rule(pool(val, val_pools, sorted(records)), records, (conf,), min_area=100)
    replay = rows[0]
    expected = tuning["joint"]["in_sample"]["pq"]
    print("REPLAY", json.dumps(dict(replay=replay, expected=expected)), flush=True)
    if abs(replay["pq"] - expected) > 1e-9:
        raise SystemExit("the validation replay does not reproduce exp168's number")

    parity = write_csv({n: rule_pool(test[n], (), None, False) for n in test}, 0.85, 100, args.output / "parity_deployed.csv")
    parity.update(compare_csv(args.output / "parity_deployed.csv", args.base_csv))
    print("PARITY", json.dumps(parity), flush=True)
    if parity["rows"] != PARITY_ROWS or parity["min_iou"] < 0.99 or parity["count_differences"]:
        raise SystemExit("the test path does not reproduce exp139's submitted file")
    facts = write_csv(pool(test, test_pools, sorted(test)), conf, 100, args.output / "submission_joint.csv")
    print("CSV", json.dumps(facts), flush=True)
    (args.output / "exp169_report.json").write_text(json.dumps(dict(decision, replay=replay, parity=parity,
                                                                    test=facts), indent=1, default=float))


if __name__ == "__main__":
    main()
