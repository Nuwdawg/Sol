"""exp199 guard: the vote may prune small, uncertain detections, but not large Mask2Former filaments.

exp169's YOLO votes carry 8 of 19 shares of its score, so a filament exp139 found confidently can
fall under the 0.68 threshold when YOLO is unsure of it. On test, 9 such instances are larger than
3,000 px and all inspected are clearly visible filaments (final_decision/analysis/review/).

Rule (pre-registered in final_decision/HANDOFF.md, section 2, plus one restriction added after
the test review): after the final rule's instances are painted, re-add every exp139 deployed
instance (single view, score >= 0.85, painted, min area 100) whose painted area is >= 3,000 px,
which no final instance overlaps at IoU >= 0.5, and of which final instances cover less than 10%
(it was deleted, not reshaped). A re-added instance takes only unclaimed pixels and is dropped if
fewer than 100 remain. Re-added instances are processed in exp139's painting order.

The same function serves both modes, so what is validated is what is deployed:

    python -m experiments.exp_199_swinb_votes.src.guard val --payload-check \\
        --cache VAL147 --shift SHIFTVAL --yolo yolov8l=V8LVAL --yolo yolo11m=Y11VAL \\
        --annotations ANN --output OUT               # exp169 + guard on validation
    python -m experiments.exp_199_swinb_votes.src.guard test --base exp139.csv --final exp169.csv \\
        --output OUT                                 # guarded test file
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from pycocotools import mask as mu

MIN_AREA, REMNANT_MIN, MATCH_IOU = 3000, 100, 0.5
# Added after the test review (2026-10-29, no labels involved): two re-added instances were the
# remainder of a filament the final rule had kept in part, which split one filament into two
# adjacent pieces (a one-to-many relation in the rubric). The guard is for filaments the vote
# deleted, so an exp139 instance already covered >= 10% by final instances is left alone.
MAX_COVERED = 0.10


def guard(base: list[dict], final: list[dict]) -> list[dict]:
    """RLEs to add to `final`: large `base` instances (in `base` order) that `final` lost."""
    if not base:
        return []
    claimed = None
    if final:
        claimed = mu.decode(mu.merge(final, intersect=False)) if len(final) > 1 else mu.decode(final[0])
    final_union = None if claimed is None else claimed.copy()
    ious = np.asarray(mu.iou(base, final, [0] * len(final))).reshape(len(base), len(final)) if final else None
    added = []
    for i, rle in enumerate(base):
        if mu.area(rle) < MIN_AREA:
            continue
        if ious is not None and ious[i].max() >= MATCH_IOU:
            continue
        mask = mu.decode(rle)
        if final_union is not None and (mask & final_union).sum() >= MAX_COVERED * mask.sum():
            continue                      # reshaped by the final rule, not deleted: leave it
        if claimed is not None:
            mask = mask & (1 - claimed)
        if int(mask.sum()) < REMNANT_MIN:
            continue
        claimed = mask.copy() if claimed is None else (claimed | mask)
        added.append(mu.encode(np.asfortranarray(mask.astype(np.uint8))))
    return added


def read_csv(path: Path) -> tuple[dict, list[str]]:
    by, order = defaultdict(list), []
    for row in csv.DictReader(open(path, newline="")):
        stem = row["filament_id"].rsplit("_", 1)[0]
        if stem not in by:
            order.append(stem)
        by[stem].append({"size": [2048, 2048], "counts": row["segmentation_rle"].encode()})
    return by, order


def test_mode(args) -> None:
    base, _ = read_csv(args.base)
    final, order = read_csv(args.final)
    stems = order + [s for s in base if s not in final]
    rows, added_log = [], []
    for stem in stems:
        kept = final.get(stem, [])
        added = guard(base.get(stem, []), kept)
        for k, rle in enumerate(kept + added, start=1):
            rows.append((f"{stem}_{k}", rle["counts"].decode("ascii") if isinstance(rle["counts"], bytes) else rle["counts"]))
        for rle in added:
            added_log.append(dict(stem=stem, area=int(mu.area(rle))))
        # overlap check, fatal
        masks = kept + added
        if len(masks) > 1:
            total = sum(int(mu.area(r)) for r in masks)
            union = int(mu.area(mu.merge(masks, intersect=False)))
            if union != total:
                raise SystemExit(f"{stem}: overlapping masks after the guard")
    args.output.mkdir(parents=True, exist_ok=True)
    out = args.output / "submission_guarded.csv"
    with open(out, "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["filament_id", "segmentation_rle"])
        writer.writerows(rows)
    digest = hashlib.sha256(out.read_bytes()).hexdigest().upper()
    n_final = sum(len(v) for v in final.values())
    report = dict(final_rows=n_final, added=len(added_log), rows=len(rows), photographs_with_rows=len({r[0].rsplit('_', 1)[0] for r in rows}),
                  sha256=digest, added_instances=added_log)
    assert len(rows) == n_final + len(added_log)
    (args.output / "guard_test_report.json").write_text(json.dumps(report, indent=1))
    print("GUARD_TEST", json.dumps({k: v for k, v in report.items() if k != "added_instances"}), flush=True)


def val_mode(args) -> None:
    """exp169's validation rule (exp168 code), its exp139 base, the guard, and pooled PQ."""
    from experiments.exp_033_inference.src.sweep import pq_rle, records_for
    from experiments.exp_149_evidence_fusion.src import evidence as E
    from experiments.exp_157_shift_views.src.views import merged_cache
    from experiments.exp_159_yolo_votes.src.votes import load_pool
    from experiments.exp_168_joint_tuning.src.joint import prepared, scores
    from shared.utils import aggregate_pq

    records, _ = records_for(args.annotations)
    cache = merged_cache(args.cache, args.shift)
    pools = [load_pool(Path(p)) for _, p in (s.split("=", 1) for s in sorted(args.yolo))]
    without, with_guard, n_added = [], [], 0
    for name in sorted(records):
        p = prepared(cache[name], [pl.get(name, []) for pl in pools], ("d4", "shift"))
        entries = list(zip(map(float, scores(p, 4.0, 0.3)), p["crops"]))
        final = [rle for s, rle in E.paint([(s, c) for s, c in entries if s >= args.conf], 100)]
        base = [rle for s, rle in E.paint([(c["score"], E.Crop(c["rle"])) for c in cache[name]["full"]
                                           if c["score"] >= 0.85], 100)]
        added = guard(base, final)
        n_added += len(added)
        for _, truth in records[name]:
            without.append(pq_rle(final, truth))
            with_guard.append(pq_rle(final + added, truth))
    a, b = aggregate_pq(without), aggregate_pq(with_guard)
    report = dict(exp169=a, guarded=b, added=n_added, delta=b["pq"] - a["pq"])
    print("GUARD_VAL", json.dumps(report), flush=True)
    if abs(a["pq"] - 0.48894859995182627) > 1e-9:
        raise SystemExit("exp169's validation row was not reproduced")
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "guard_val_report.json").write_text(json.dumps(report, indent=1))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="mode", required=True)
    t = sub.add_parser("test")
    t.add_argument("--base", type=Path, required=True, help="exp139 deployed test CSV")
    t.add_argument("--final", type=Path, required=True, help="the final rule's test CSV")
    t.add_argument("--output", type=Path, required=True)
    v = sub.add_parser("val")
    v.add_argument("--cache", type=Path, required=True)
    v.add_argument("--shift", type=Path, required=True)
    v.add_argument("--yolo", action="append", default=[])
    v.add_argument("--annotations", type=Path, required=True)
    v.add_argument("--conf", type=float, default=0.68)
    v.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    test_mode(args) if args.mode == "test" else val_mode(args)


if __name__ == "__main__":
    main()
