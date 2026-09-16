"""Separate inference configuration from model quality for the exp002 detector.

A public inference notebook runs a YOLOv8l-seg at imgsz 2048 with conf 0.3, NMS
IoU 0.0, default (non-retina) masks, no erosion and a 5-pixel area floor. Its
team sits at 0.55 public while our tuned D4 emission of a same-family model sits
at 0.39. Before training anything, this measures how much of that gap is the
inference recipe alone, on the canonical validation split with exact pooled
per-record PQ.

Predictions are cached once per (NMS IoU, retina) pair at a low confidence floor
as score + RLE. Every confidence, erosion, area and paint-order rule is then
scored offline on RLEs, decoding only masks that actually overlap. A parity rule
reproduces a known number first; the resolved Ultralytics arguments are read
back from the predictor and recorded, not assumed.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import cv2
import numpy as np
from pycocotools import mask as mu

from experiments.exp_033_inference.src.sweep import encode, pq_rle, records_for
from shared.utils import aggregate_pq

NATIVE = 2048
KERNEL = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))


def read_image(path: Path) -> np.ndarray:
    grey = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if grey is None or grey.shape != (NATIVE, NATIVE):
        raise SystemExit(f"cannot read {path}")
    return np.repeat(grey[..., None], 3, axis=2)


def predict(model, image, iou: float, retina: bool, floor: float, max_det: int) -> list[dict]:
    result = model.predict(image, imgsz=NATIVE, conf=floor, iou=iou, max_det=max_det,
                           retina_masks=retina, verbose=False)[0]
    if result.masks is None or len(result.masks.data) == 0:
        return []
    data = result.masks.data.cpu().numpy()
    scores = result.boxes.conf.cpu().numpy()
    entries = []
    for index in range(len(data)):
        mask = (data[index] > .5).astype(np.uint8)
        if mask.shape != (NATIVE, NATIVE):
            mask = cv2.resize(mask, (NATIVE, NATIVE), interpolation=cv2.INTER_NEAREST)
        if mask.sum() == 0:
            continue
        entries.append(dict(score=float(scores[index]), rle=encode(mask),
                            eroded=encode(cv2.erode(mask, KERNEL))))
    return entries


def paint(entries: list[dict], conf: float, erode: bool, min_area: int, order: str) -> list[dict]:
    """Pixel-disjoint emission on RLEs; decode only masks that overlap a claim.

    order "score": highest confidence claims first and dropped masks claim
    nothing, which is `shared.utils.paint_panoptic`.
    order "public": the public notebook's order. It sorts by class id, which is
    constant, so the reversed argsort claims the LOWEST-confidence mask first,
    and every mask claims its pixels even when its remainder is then dropped.
    """
    kept = [e for e in entries if e["score"] >= conf]
    key = "eroded" if erode else "rle"
    if order == "score":
        ordered = sorted(kept, key=lambda e: -e["score"])
    else:
        ordered = sorted(kept, key=lambda e: e["score"])
    claimed, output = [], []
    for entry in ordered:
        rle = entry[key]
        remaining_rle, area = rle, int(mu.area(rle))
        if claimed and area:
            overlap = np.asarray(mu.iou([rle], claimed, [0] * len(claimed)))[0]
            hits = np.nonzero(overlap > 0)[0]
            if len(hits):
                mask = mu.decode(rle)
                for j in hits:
                    mask &= (1 - mu.decode(claimed[j]))
                area = int(mask.sum())
                remaining_rle = encode(mask)
        if area >= min_area and area > 0:
            output.append(remaining_rle)
            claimed.append(remaining_rle if order == "score" else rle)
        elif order == "public":
            claimed.append(rle)
    return output


def score(cache: dict, records: dict, conf: float, erode: bool, min_area: int, order: str) -> dict:
    metrics, instances = [], 0
    for name, passes in records.items():
        predicted = paint(cache[name], conf, erode, min_area, order)
        instances += len(predicted)
        metrics.extend(pq_rle(predicted, truth) for _, truth in passes)
    return dict(instances=instances, **aggregate_pq(metrics))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", required=True)
    parser.add_argument("--annotations", required=True)
    parser.add_argument("--images", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    # 0.05 matches the exp033 cache that produced the 0.4403668 single-view baseline.
    parser.add_argument("--floor", type=float, default=0.05)
    parser.add_argument("--max-det", type=int, default=100)
    parser.add_argument("--parity-iou", type=float, default=0.60)
    parser.add_argument("--parity-retina", type=int, default=1)
    parser.add_argument("--parity-conf", type=float, default=0.35)
    parser.add_argument("--parity-erode", type=int, default=1)
    parser.add_argument("--parity-min-area", type=int, default=300)
    parser.add_argument("--parity-pq", type=float, default=None)
    parser.add_argument("--nms", type=float, nargs="+", default=[0.0, 0.3, 0.6])
    parser.add_argument("--retina", type=int, nargs="+", default=[0, 1])
    args = parser.parse_args()
    cv2.setNumThreads(2)

    from ultralytics import YOLO
    model = YOLO(args.weights)
    records, split = records_for(args.annotations)
    names = sorted(records)
    args.output.mkdir(parents=True, exist_ok=True)

    configurations = [(iou, bool(retina)) for iou in args.nms for retina in args.retina]
    if (args.parity_iou, bool(args.parity_retina)) not in configurations:
        configurations.append((args.parity_iou, bool(args.parity_retina)))
    caches, resolved = {}, {}
    for iou, retina in configurations:
        started, cache = time.time(), {}
        for position, name in enumerate(names, start=1):
            cache[name] = predict(model, read_image(args.images / name), iou, retina,
                                  args.floor, args.max_det)
            if position == 1:
                predictor = model.predictor.args
                resolved[f"iou{iou}_retina{int(retina)}"] = dict(
                    iou=float(predictor.iou), conf=float(predictor.conf),
                    imgsz=predictor.imgsz, retina_masks=bool(predictor.retina_masks),
                    max_det=int(predictor.max_det))
                print("RESOLVED", json.dumps(resolved[f"iou{iou}_retina{int(retina)}"]), flush=True)
        key = f"iou{iou}_retina{int(retina)}"
        caches[key] = cache
        (args.output / f"cache_{key}.json").write_text(json.dumps(
            {n: [dict(score=e["score"], rle=e["rle"]) for e in v] for n, v in cache.items()}))
        print(f"cached {key}: {sum(len(v) for v in cache.values())} masks "
              f"in {time.time() - started:.0f}s", flush=True)

    parity_key = f"iou{args.parity_iou}_retina{int(bool(args.parity_retina))}"
    parity = score(caches[parity_key], records, args.parity_conf, bool(args.parity_erode),
                   args.parity_min_area, "score")
    parity.update(key=parity_key, expected=args.parity_pq,
                  reproduced=None if args.parity_pq is None
                  else abs(parity["pq"] - args.parity_pq) < 1e-3)  # exp033's own tolerance
    print("PARITY", json.dumps(parity), flush=True)

    rows = []
    for key, cache in caches.items():
        for conf in (0.20, 0.25, 0.30, 0.35, 0.40):
            for erode in (False, True):
                for min_area in (5, 100, 300):
                    for order in ("score", "public"):
                        row = dict(config=key, conf=conf, erode=erode, min_area=min_area,
                                   order=order, **score(cache, records, conf, erode,
                                                        min_area, order))
                        rows.append(row)
        best = max((r for r in rows if r["config"] == key), key=lambda r: r["pq"])
        print("BEST", json.dumps(best), flush=True)
        (args.output / "rules.json").write_text(json.dumps(rows, indent=1))
    rows.sort(key=lambda r: -r["pq"])
    public_recipe = next((r for r in rows if r["config"] == "iou0.0_retina0" and r["conf"] == 0.30
                          and not r["erode"] and r["min_area"] == 5 and r["order"] == "public"), None)
    report = dict(status="complete", split=split.summary(), resolved=resolved, parity=parity,
                  public_recipe_exact=public_recipe, best=rows[0], top=rows[:25])
    (args.output / "report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(dict(parity=parity, public_recipe_exact=public_recipe, best=rows[0]),
                     indent=2), flush=True)


if __name__ == "__main__":
    main()
