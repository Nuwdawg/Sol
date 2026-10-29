"""exp162: the submission rule - Mask2Former instances, dihedral vote masks, votes from two YOLO models.

For every photograph:
1. exp139 epoch 40 proposes candidates in the full view (class-probability floor 0.02) and in the
   seven other dihedral views (floor 0.05).
2. Each full-view candidate collects partners: the best-IoU (> 0.5) candidate of every other
   dihedral view, and the best-IoU (> 0.5) instance of each YOLO model's dihedral pool (exp097
   YOLOv8l-seg at 2048, exp002 YOLO11m-seg at 1280; eight views fused by exp033's rule).
3. Mask: the pixel majority (k = ceil(n/2)) over the candidate and its dihedral partners. YOLO
   masks do not vote on pixels: they follow another boundary convention.
4. Score: sqrt(s x (s + sum of dihedral partner scores + w x sum of YOLO partner scores) / (8 + w x m)),
   m the number of YOLO models, a missing partner counting 0.
5. Emission: paint by score, drop pieces under 100 px, keep scores >= the threshold chosen on
   validation (exp159).

Validation reads exp147's cache and the two YOLO validation pools; test reads exp151a's cache,
exp161's YOLOv8l test pool and exp002's raw test views (fused here with the identical call).
Checks before any CSV: the deployed exp139 rule on the test cache rebuilds exp139's submitted file,
and the exp002 test pool is fused exactly as exp119 fused it.

    python -m experiments.exp_162_final_submission.src.final --val-cache ... --test-cache ...
        --val-yolo yolov8l=... --val-yolo yolo11m=... --test-yolo yolov8l=... --test-views-002 a.json b.json
        --base-csv exp139.csv --annotations ... --output ...
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from experiments.exp_033_inference.src.cached_tta import fuse
from experiments.exp_033_inference.src.sweep import records_for
from experiments.exp_149_evidence_fusion.src import evidence as E
from experiments.exp_152_rule_and_emit.src.emit import PARITY_ROWS, compare_csv, write_csv
from experiments.exp_157_shift_views.src.views import CONFS, merged_cache, rule_pool
from experiments.exp_159_yolo_votes.src.votes import load_pool, pool_for, yolo_support


def fused_views(paths: list[Path]) -> dict:
    """exp002's raw test views -> fused instances, exp119's `instances_from_views` call for call."""
    views: dict[str, list] = {}
    for path in paths:
        for name, entries in json.loads(Path(path).read_text()).items():
            views.setdefault(Path(name).name, []).extend(entries)
    pool = {}
    for name, entries in views.items():
        items = []
        for entry in entries:
            score, rle = entry if isinstance(entry, (list, tuple)) else (entry["score"], entry["rle"])
            counts = rle["counts"]
            items.append((float(score), dict(size=rle["size"], counts=counts.decode("ascii")
                                             if isinstance(counts, bytes) else counts)))
        pool[name] = [dict(score=float(e["score"]), rle=e["rle"]) for e in fuse(items, 0.5, 0.5)]
    return pool


def pool_with_masks(entry, support, weight, views, yolo_entries, erode):
    """The final rule's pool; with erode >= 0 the YOLO partner masks also vote on pixels (exp166)."""
    if erode < 0 or not yolo_entries:
        return pool_for(entry, support, weight, views)
    from experiments.exp_166_yolo_mask_votes.src.mask_votes import pool_for as voted_pool
    if views != ("d4",):
        raise SystemExit("YOLO mask votes are validated with dihedral views only")
    return voted_pool(entry, yolo_entries, erode)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--val-cache", type=Path, required=True)
    parser.add_argument("--test-cache", type=Path, required=True)
    parser.add_argument("--val-yolo", action="append", default=[], help="tag=pool.json")
    parser.add_argument("--test-yolo", action="append", default=[], help="tag=pool.json")
    parser.add_argument("--test-views-002", type=Path, nargs="*", default=[])
    parser.add_argument("--weight", type=float, default=2.0)
    parser.add_argument("--yolo-mask-erode", type=int, default=-1,
                        help=">= 0: YOLO partner masks, eroded this many pixels, join the pixel vote (exp166)")
    parser.add_argument("--val-shift", type=Path, default=None, help="exp157 validation shift views")
    parser.add_argument("--test-shift", type=Path, default=None, help="exp157 test shift views")
    parser.add_argument("--base-csv", type=Path, required=True)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    records, _ = records_for(args.annotations)
    val, test = merged_cache(args.val_cache, args.val_shift), merged_cache(args.test_cache, args.test_shift)
    shift = args.val_shift is not None and args.test_shift is not None
    if len(test) != 180:
        raise SystemExit(f"expected 180 test photographs, found {len(test)}")
    val_pools = {t: load_pool(Path(p)) for t, p in (s.split("=", 1) for s in args.val_yolo)}
    test_pools = {t: load_pool(Path(p)) for t, p in (s.split("=", 1) for s in args.test_yolo)}
    if args.test_views_002:
        test_pools["yolo11m"] = fused_views(args.test_views_002)
    tags = sorted(val_pools)
    if sorted(test_pools) != tags:
        raise SystemExit(f"validation pools {tags} and test pools {sorted(test_pools)} differ")
    missing = [n for n in test if any(n not in test_pools[t] for t in tags)]
    if missing:
        raise SystemExit(f"{len(missing)} test photographs lack a YOLO pool: {missing[:3]}")
    report = dict(weight=args.weight, yolo=tags)

    # Validation: the final rule and its components, thresholds swept.
    variants = {"deployed": None, "d4_vote_geo": [], "final": tags}
    views = {"d4_vote_geo": ("d4",), "final": ("d4",)}
    if shift:
        variants["final_shift"] = tags
        views["final_shift"] = ("d4", "shift")
    per_record, chosen = {}, {}
    for name, sources in variants.items():
        if sources is None:
            pool = {n: [(c["score"], E.Crop(c["rle"])) for c in val[n]["full"]] for n in records}
            rows, per = E.score_rule(pool, records, (0.85,), min_area=100)
        else:
            support = {n: [yolo_support(val[n], val_pools[t].get(n, [])) for t in sources] for n in records}
            pool = {n: pool_with_masks(val[n], support[n], args.weight if sources else 0.0, views[name],
                                       [val_pools[t].get(n, []) for t in sources], args.yolo_mask_erode)
                    for n in records}
            rows, per = E.score_rule(pool, records, CONFS, min_area=100)
        best = max(rows, key=lambda r: r["pq"])
        per_record[name], chosen[name] = per[best["conf"]], best
    report["validation"] = {name: dict(best=chosen[name],
                                       bootstrap=E.bootstrap_difference(per_record["deployed"], per_record[name], records))
                            for name in variants}
    print("VALIDATION", json.dumps(report["validation"]), flush=True)

    # Parity: the deployed rule on this test cache must rebuild exp139's submitted file.
    parity = write_csv({n: rule_pool(test[n], (), None, False) for n in test}, 0.85, 100, args.output / "parity_deployed.csv")
    parity.update(compare_csv(args.output / "parity_deployed.csv", args.base_csv))
    report["parity"] = parity
    print("PARITY", json.dumps(parity), flush=True)
    if parity["rows"] != PARITY_ROWS or parity["min_iou"] < 0.99 or parity["count_differences"]:
        raise SystemExit("the test path does not reproduce exp139's submitted file")

    report["test"] = {}
    for name in [v for v in variants if v != "deployed"]:
        sources = variants[name]
        conf = chosen[name]["conf"]
        support = {n: [yolo_support(test[n], test_pools[t].get(n, [])) for t in sources] for n in test}
        pool = {n: pool_with_masks(test[n], support[n], args.weight if sources else 0.0, views[name],
                                   [test_pools[t].get(n, []) for t in sources], args.yolo_mask_erode)
                for n in test}
        facts = write_csv(pool, conf, 100, args.output / f"submission_{name}.csv")
        report["test"][name] = facts
        print("CSV", name, json.dumps(facts), flush=True)
    (args.output / "exp162_report.json").write_text(json.dumps(report, indent=1, default=float))


if __name__ == "__main__":
    main()
