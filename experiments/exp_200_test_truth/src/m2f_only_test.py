"""exp200u: the Mask2Former-only two-model file for the test photographs (m2f_only.py's rule).

Rule: score = s^0.3 x mean^0.7, mean = (sum of exp139's 11 view scores + 8 x s_B) / (11 + 8), where s_B
is the score of Swin-B's best full-view partner at IoU > 0.5 (class probability >= 0.5); masks are the
majority vote of exp139's views; keep score >= 0.84, paint by score, area >= 100. No YOLO.

Checks before the test file is written:
1. validation parity: the same functions, painted by write_csv's E.paint, reproduce m2f_only.py's
   per-record PQ for this setting;
2. the test cache rebuilds exp169's submitted file under exp169's rule (emit_test.py's check), so
   the Mask2Former views on test are the ones exp169 used;
3. Swin-B covers the 180 test photographs; write_csv refuses empty or overlapping instances.

    python -m experiments.exp_200_test_truth.src.m2f_only_test --annotations ... --val-cache ... --val-shift ...
        --val-swinb SHARD0 ... SHARD4 --val-report m2f_only.json --test-cache e040_test_cache.json
        --test-swinb shard0.jsonl shard1.jsonl --exp169-ingredients test_ingredients.pkl --output dir
"""
from __future__ import annotations

import argparse
import hashlib
import json
import pickle
from pathlib import Path

from experiments.exp_033_inference.src.sweep import records_for
from experiments.exp_149_evidence_fusion.src import evidence as E
from experiments.exp_152_rule_and_emit.src.emit import write_csv
from experiments.exp_157_shift_views.src.views import merged_cache
from experiments.exp_168_joint_tuning.src.joint import prepared
from experiments.exp_168_joint_tuning.src.joint import scores as exp169_scores
from experiments.exp_199_swinb_votes.src.votes import merge_shards
from experiments.exp_200_test_truth.src.emit_test import EXP169_SHA256
from experiments.exp_200_test_truth.src.hypotheses import as_bytes, match, total
from experiments.exp_200_test_truth.src.m2f_only import FLOOR, scores
from experiments.exp_200_test_truth.src.swinb_gate import support

M, A, CONF, AREA = 8.0, 0.3, 0.84, 100


def entries(cache: dict, swinb: dict) -> dict:
    out = {}
    for n in sorted(cache):
        p = prepared(cache[n], [], ("d4", "shift"))
        s_b = support(cache[n]["full"], [c for c in swinb[n] if c["p"] >= FLOOR])
        out[n] = list(zip(map(float, scores(p, s_b, M, A)), p["crops"]))
    return out


def validation_parity(args) -> float:
    records, _ = records_for(args.annotations)
    names = sorted(records)
    val_swinb, _ = merge_shards(args.val_swinb, names)
    val = entries(merged_cache(args.val_cache, args.val_shift), {n: val_swinb[n]["full"] for n in names})
    rows = []
    for n in names:
        pred = [as_bytes(r) for _, r in E.paint([(s, c) for s, c in val[n] if s >= CONF], AREA)]
        rows.extend(match(pred, [as_bytes(t) for t in truth]) for _, truth in sorted(records[n]))
    rebuilt = total(rows)["pq"]
    report = json.loads(args.val_report.read_text())
    expected = next(r["pq"] for r in report["rows"]
                    if (r["swinb_weight"], r["exponent"], r["threshold"]) == (M, A, CONF))
    print("VAL_PARITY", json.dumps(dict(rebuilt=rebuilt, expected=expected)), flush=True)
    if abs(rebuilt - expected) > 1e-9:
        raise SystemExit("the test path does not reproduce the validation number")
    return rebuilt


def exp169_parity(args) -> dict:
    ingredients = pickle.loads(args.exp169_ingredients.read_bytes())
    exp169 = {n: list(zip(map(float, exp169_scores(p, 4.0, 0.3)), p["crops"])) for n, p in ingredients.items()}
    parity = write_csv(exp169, 0.68, 100, args.output / "exp169_rebuilt.csv")
    print("EXP169_PARITY", json.dumps(parity), flush=True)
    if parity["sha256"] != EXP169_SHA256:
        raise SystemExit("exp169's rule does not rebuild its submitted file from these test ingredients")
    return parity


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--val-cache", type=Path, default=None, help="with --val-shift, --val-swinb and --val-report: "
                                                                     "check 1 (needs the exp198 Swin-B validation caches)")
    parser.add_argument("--val-shift", type=Path, default=None)
    parser.add_argument("--val-swinb", type=Path, nargs="+", default=None)
    parser.add_argument("--val-report", type=Path, default=None, help="m2f_only.py's output")
    parser.add_argument("--test-cache", type=Path, required=True, help="exp139 test cache with full, d4 and shift")
    parser.add_argument("--test-swinb", type=Path, nargs="+", required=True, help="swinb_infer.py's JSONL shards")
    parser.add_argument("--exp169-ingredients", type=Path, default=None, help="emit_test.py's cached test ingredients "
                                                                            "(check 2)")
    parser.add_argument("--expect-sha256", default=None, help="the file this must rebuild (reproduction)")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    rebuilt, parity = None, None
    if args.val_cache:
        rebuilt = validation_parity(args)
    if args.exp169_ingredients:
        parity = exp169_parity(args)
    swinb = {}
    for path in args.test_swinb:
        for line in path.read_text().splitlines():
            if line.strip():
                item = json.loads(line)
                swinb[item["name"]] = item["full"]
    test = merged_cache(args.test_cache, args.test_cache)
    if sorted(swinb) != sorted(test) or len(test) != 180:
        raise SystemExit("Swin-B does not cover the 180 test photographs of the cache")
    facts = write_csv(entries(test, swinb), CONF, AREA, args.output / "submission_m2f_only_w8.csv")
    print("WROTE", json.dumps(facts), flush=True)
    if args.expect_sha256 and facts["sha256"] != args.expect_sha256.upper():
        raise SystemExit("the file differs from the stored one")
    (args.output / "m2f_only_test.json").write_text(json.dumps(dict(val_parity=rebuilt, exp169_parity=parity,
                                                                    file=facts, rule=dict(m=M, a=A, conf=CONF, area=AREA)),
                                                               indent=1))


if __name__ == "__main__":
    main()
