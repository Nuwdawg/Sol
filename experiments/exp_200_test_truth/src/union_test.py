"""exp200o: write union test files: a guarded base file plus Swin-B's uncovered confident detections.

The same `add_uncovered` as validation (union.py) runs on each photograph: Swin-B full-view candidates
(swinb_infer.py) with score >= tau, less than 10% covered by the base file's instances, painted by
descending score into unclaimed pixels, remnants under 100 px dropped. The base rows are copied
unchanged, in order, and the added rows follow them. Fatal checks: no two instances overlap, every
base row is kept, row counts add up, all 180 photographs were run through Swin-B.

    python -m experiments.exp_200_test_truth.src.union_test --base exp169_guarded.csv --tau 0.90
        --swinb shard0.jsonl shard1.jsonl --output union.csv
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path

from pycocotools import mask as mu

from experiments.exp_200_test_truth.src.additions import add_uncovered


def read_csv(path: Path) -> tuple[dict, list]:
    by, order = defaultdict(list), []
    for row in csv.DictReader(open(path, newline="")):
        stem = row["filament_id"].rsplit("_", 1)[0]
        if stem not in by:
            order.append(stem)
        by[stem].append(row["segmentation_rle"])
    return by, order


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--tau", type=float, required=True)
    parser.add_argument("--swinb", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    swinb = {}
    for path in args.swinb:
        for line in path.read_text().splitlines():
            if line.strip():
                item = json.loads(line)
                swinb[Path(item["name"]).stem] = item["full"]
    if len(swinb) != 180:
        raise SystemExit(f"Swin-B ran on {len(swinb)} test photographs, expected 180")
    base, order = read_csv(args.base)
    stems = order + sorted(s for s in swinb if s not in base)
    rows, added = [], []
    for stem in stems:
        kept = [dict(size=[2048, 2048], counts=c.encode()) for c in base.get(stem, [])]
        extra = add_uncovered(kept, swinb.get(stem, []), args.tau)
        masks = kept + extra
        if len(masks) > 1 and int(mu.area(mu.merge(masks, intersect=False))) != sum(int(mu.area(m)) for m in masks):
            raise SystemExit(f"{stem}: overlapping instances")
        for k, rle in enumerate(masks, start=1):
            counts = rle["counts"].decode("ascii") if isinstance(rle["counts"], bytes) else rle["counts"]
            rows.append((f"{stem}_{k}", counts))
        added.extend(dict(stem=stem, area=int(mu.area(m))) for m in extra)
    n_base = sum(len(v) for v in base.values())
    if len(rows) != n_base + len(added):
        raise SystemExit("row counts do not add up")
    with args.output.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["filament_id", "segmentation_rle"])
        writer.writerows(rows)
    report = dict(base=args.base.name, tau=args.tau, base_rows=n_base, added=len(added), rows=len(rows),
                  photographs_with_rows=len({r[0].rsplit("_", 1)[0] for r in rows}),
                  sha256=hashlib.sha256(args.output.read_bytes()).hexdigest().upper())
    print("UNION", json.dumps(report), flush=True)
    args.output.with_suffix(".json").write_text(json.dumps(dict(report, added_instances=added), indent=1))


if __name__ == "__main__":
    main()
