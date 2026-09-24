"""Harvest D4 proposals on the training split with the frozen exp002 detector.

The refinement experiment needs proposals whose error distribution matches the
validation proposals it must repair. So this reuses the exact exp007 dihedral
path, the exact fusion constants of the retained validation pool, and the same
frozen weights. Nothing is retrained and no annotation is read for inference.

A shipped slice of the validation pool is regenerated first. If the reproduced
masks do not agree with the stored pool, the harvest is not comparable and the
run stops before it spends an hour on incomparable data.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import numpy as np
from pycocotools import mask as mask_util

from experiments.exp_007_tta.src.dihedral import TRANSFORMS, fuse, predict_views
from shared.data_split import assert_disjoint, make_split

LINK_IOU = 0.5
VOTE_FRACTION = 0.5


def encode_view(entries) -> list:
    return [(score, dict(size=rle["size"], counts=rle["counts"].decode("ascii")))
            for score, rle in entries]


def fused_rles(entries) -> list[dict]:
    output = []
    for score, binary in fuse(entries, LINK_IOU, VOTE_FRACTION, len(TRANSFORMS)):
        rle = mask_util.encode(np.asfortranarray(binary.astype(np.uint8)))
        output.append(dict(score=float(score),
                           rle=dict(size=rle["size"], counts=rle["counts"].decode("ascii"))))
    return output


def agreement(produced: list[dict], stored: list[dict]) -> dict:
    if not produced or not stored:
        return dict(produced=len(produced), stored=len(stored), mean_best_iou=0.)
    table = np.asarray(mask_util.iou([row["rle"] for row in produced],
                                     [row["rle"] for row in stored],
                                     [0] * len(stored)))
    return dict(produced=len(produced), stored=len(stored),
                mean_best_iou=float(table.max(axis=0).mean()))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", required=True)
    parser.add_argument("--annotations", required=True)
    parser.add_argument("--images", required=True)
    parser.add_argument("--reference", required=True,
                        help="stored validation pool slice used as the parity check")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--imgsz", type=int, default=2048)
    parser.add_argument("--floor-conf", type=float, default=0.25)
    parser.add_argument("--max-det", type=int, default=100)
    parser.add_argument("--min-parity", type=float, default=0.97)
    parser.add_argument("--time-hours", type=float, default=2.0)
    args = parser.parse_args()

    split = make_split(args.annotations)
    assert_disjoint(split)
    coco = json.loads(Path(args.annotations).read_text())
    stem_of = {record["id"]: record["file_name"] for record in coco["images"]}
    train_photos = sorted({stem_of[i] for i in split.train_image_ids})
    val_photos = sorted({stem_of[i] for i in split.val_image_ids})
    reference = json.loads(Path(args.reference).read_text())
    assert set(reference) <= set(val_photos), "parity slice must come from validation"

    from ultralytics import YOLO
    model = YOLO(args.weights)
    images = Path(args.images)
    args.output.mkdir(parents=True, exist_ok=True)

    parity = {}
    for stem in sorted(reference):
        produced = fused_rles(predict_views(model, images / stem, args.imgsz,
                                            args.floor_conf, args.max_det))
        parity[stem] = agreement(produced, reference[stem])
        print("parity", stem, json.dumps(parity[stem]), flush=True)
    mean_parity = float(np.mean([row["mean_best_iou"] for row in parity.values()]))
    (args.output / "parity.json").write_text(json.dumps(
        dict(photographs=parity, mean_best_iou=mean_parity,
             link=LINK_IOU, vote=VOTE_FRACTION, imgsz=args.imgsz,
             floor_conf=args.floor_conf, max_det=args.max_det), indent=2))
    assert mean_parity >= args.min_parity, f"validation pool not reproduced: {mean_parity}"
    print("PARITY_OK", mean_parity, flush=True)

    views_path = args.output / "train_views.json"
    candidates_path = args.output / "train_candidates.json"
    views = json.loads(views_path.read_text()) if views_path.exists() else {}
    candidates = json.loads(candidates_path.read_text()) if candidates_path.exists() else {}
    started, deadline = time.time(), time.time() + args.time_hours * 3600
    for position, stem in enumerate(train_photos, start=1):
        if stem in candidates:
            continue
        if time.time() > deadline:
            print("TIME_LIMIT", position, flush=True)
            break
        entries = predict_views(model, images / stem, args.imgsz,
                                args.floor_conf, args.max_det)
        views[stem] = encode_view(entries)
        candidates[stem] = fused_rles(entries)
        if position % 20 == 0 or position == len(train_photos):
            views_path.write_text(json.dumps(views))
            candidates_path.write_text(json.dumps(candidates))
            print(f"harvest {position}/{len(train_photos)} "
                  f"proposals={sum(len(v) for v in candidates.values())} "
                  f"elapsed={time.time() - started:.0f}s", flush=True)
    views_path.write_text(json.dumps(views))
    candidates_path.write_text(json.dumps(candidates))
    summary = dict(status="complete" if len(candidates) == len(train_photos) else "partial",
                   split=split.summary(), photographs=len(candidates),
                   expected=len(train_photos),
                   proposals=sum(len(v) for v in candidates.values()),
                   parity=mean_parity, elapsed_seconds=time.time() - started)
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
