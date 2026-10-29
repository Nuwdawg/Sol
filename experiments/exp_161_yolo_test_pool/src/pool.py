"""exp161: the YOLOv8l dihedral instance pool on the test photographs, built exactly like exp098's.

exp159 found that exp097's YOLOv8l-seg (stage2_best.pt) adds an independent vote to rule A on
validation. Its validation pool came from exp098: eight dihedral views at confidence floor 0.25,
max 100 detections, clustered by exp033's `fuse(entries, 0.5, 0.5)`. No test pool was ever kept.

This script rebuilds that pool with the same calls (`predict_views`, `fuse`, Ultralytics 8.4.153)
for the validation photographs first and compares it with exp098's file - the deployment path must
reproduce the validation path (CLAUDE.md section 5) - and then for the 180 test photographs.
Photographs are split over the visible GPUs by `--shard i --shards n`.

    python -m experiments.exp_161_yolo_test_pool.src.pool --weights stage2_best.pt --annotations ...
        --train-images ... --test-images ... --output ... [--shard 0 --shards 2]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from pycocotools import mask as mu

from experiments.exp_007_tta.src.dihedral import predict_views
from experiments.exp_033_inference.src.cached_tta import fuse
from experiments.exp_033_inference.src.sweep import records_for


def as_text(rle: dict) -> dict:
    counts = rle["counts"]
    return dict(size=[int(v) for v in rle["size"]], counts=counts.decode("ascii") if isinstance(counts, bytes) else counts)


def pool_for(model, path: Path) -> list[dict]:
    views = predict_views(model, path, 2048, 0.25, 100)
    entries = [(float(score), as_text(rle)) for score, rle in views]
    return [dict(score=float(e["score"]), rle=as_text(e["rle"])) for e in fuse(entries, 0.5, 0.5)]


def compare(ours: dict, theirs: dict) -> dict:
    """Instance agreement of two pools: counts, worst matched IoU, largest score difference."""
    count_diff, worst_iou, worst_score = 0, 1.0, 0.0
    for name in ours:
        a = ours[name]
        b = theirs.get(name) or theirs.get(Path(name).stem) or []
        count_diff += abs(len(a) - len(b))
        if a and b:
            ra = [dict(size=e["rle"]["size"], counts=e["rle"]["counts"].encode("ascii")) for e in a]
            rb = [dict(size=e["rle"]["size"], counts=e["rle"]["counts"].encode("ascii")
                       if isinstance(e["rle"]["counts"], str) else e["rle"]["counts"]) for e in b]
            ious = np.asarray(mu.iou(ra, rb, [0] * len(rb))).reshape(len(ra), len(rb))
            best = ious.argmax(axis=1)
            worst_iou = min(worst_iou, float(ious.max(axis=1).min()))
            worst_score = max(worst_score, max(abs(a[i]["score"] - b[j]["score"]) for i, j in enumerate(best)))
    return dict(count_differences=count_diff, worst_iou=worst_iou, worst_score_difference=worst_score)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--train-images", type=Path, required=True)
    parser.add_argument("--test-images", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--shards", type=int, default=1)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    from ultralytics import YOLO
    model = YOLO(str(args.weights))
    records, _ = records_for(args.annotations)
    jobs = [("val", args.train_images / name) for name in sorted(records)]
    jobs += [("test", path) for path in sorted(args.test_images.glob("*.jpeg"))]
    if args.limit:
        jobs = jobs[:args.limit]
    mine = jobs[args.shard::args.shards]
    pools = {"val": {}, "test": {}}
    for position, (split, path) in enumerate(mine, start=1):
        pools[split][path.name] = pool_for(model, path)
        if position % 20 == 0 or position == len(mine):
            print("PROGRESS", json.dumps(dict(shard=args.shard, done=position, of=len(mine))), flush=True)
    for split, pool in pools.items():
        (args.output / f"yolov8l_{split}_pool_shard{args.shard}.json").write_text(json.dumps(pool))
    print("SHARD_DONE", args.shard, flush=True)


if __name__ == "__main__":
    main()
