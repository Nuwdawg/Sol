"""Score exp097 checkpoints on Kaggle with the pipeline that produced the 0.39 submission.

Runs in a Kaggle T4 kernel (`scripts/package_exp098_evaluation.py`), reading the weights
from the private dataset the pod uploaded. Nothing is downloaded to the user's PC.

For every checkpoint the pod produced (best.pt, last.pt, stage<k>_best.pt):

1. Integrity: the file's SHA256 must match the pod's manifest.json.
2. Single view, the exp033 settings: imgsz 2048, confidence floor 0.05, NMS IoU 0.60,
   retina masks. An emission sweep over confidence x erosion {none, one pixel} x
   minimum area {100, 300}, with exact pooled per-record PQ.
3. D4, the 0.39 pipeline itself, for the best `--d4-top` checkpoints by single-view
   PQ: eight dihedral views at floor 0.25, clustered at IoU 0.5 and pixel-voted at
   0.5 (`cached_tta.fuse`), then the same emission sweep. The operating point is
   re-chosen, not copied: exp_010 showed a 2048-trained model prefers no erosion
   where exp002 needed one pixel.

Before any checkpoint is scored, the evaluator must reproduce the exp062 baseline,
0.4558638687, from the stored exp002 D4 pool with the same code path. A new number
is therefore compared with the old one by construction, not by assumption.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time

import cv2
import numpy as np

from experiments.exp_007_tta.src.dihedral import predict_views
from experiments.exp_033_inference.src.cached_tta import fuse
from experiments.exp_033_inference.src.sweep import encode, evaluate_thresholds, records_for
from experiments.exp_097_runpod_yolo.src.events import emit

BASELINE_PQ = 0.4558638687292126          # exp062: exp002 D4, conf 0.35, erosion 1, area 300
SINGLE_CONFS = (0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50)
# The first run put the D4 optimum on the grid's lower edge (0.25). `fuse` scales a
# cluster's score by 0.5 + 0.5 * support, so a mask seen by one of eight views keeps
# barely more than half its confidence: fused scores below the 0.25 view floor are
# ordinary, and the grid has to reach them or it reports its own boundary.
D4_CONFS = (0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50)
EMISSION = [(grow, area) for grow in (0, -1) for area in (100, 300)]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read(path: Path) -> np.ndarray:
    grey = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if grey is None or grey.shape != (2048, 2048):
        raise SystemExit(f"cannot read {path}")
    return np.repeat(grey[..., None], 3, axis=2)


def single_view(model, image_path: Path) -> list[dict]:
    result = model.predict(read(image_path), imgsz=2048, conf=0.05, iou=0.60, max_det=100,
                           retina_masks=True, verbose=False)[0]
    if result.masks is None or len(result.masks.data) == 0:
        return []
    entries = []
    for mask, score in zip(result.masks.data.cpu().numpy(), result.boxes.conf.cpu().numpy()):
        binary = (mask > 0.5).astype(np.uint8)
        if binary.shape != (2048, 2048):
            binary = cv2.resize(binary, (2048, 2048), interpolation=cv2.INTER_NEAREST)
        if binary.sum() >= 40:
            entries.append(dict(score=float(score), rle=encode(binary)))
    return entries


def d4(model, image_path: Path) -> list[dict]:
    views = predict_views(model, image_path, 2048, 0.25, 100)
    entries = [(score, dict(size=rle["size"], counts=rle["counts"].decode("ascii")
                            if isinstance(rle["counts"], bytes) else rle["counts"]))
               for score, rle in views]
    return fuse(entries, 0.5, 0.5)


def sweep(cache: dict, records: dict, confs: tuple) -> list[dict]:
    rows = []
    for grow, area in EMISSION:
        rows.extend(evaluate_thresholds(cache, records, confs, grow=grow, min_area=area))
    return sorted(rows, key=lambda row: -row["pq"])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights-dir", type=Path, required=True)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--images", type=Path, required=True)
    parser.add_argument("--baseline-pool", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--d4-top", type=int, default=2)
    args = parser.parse_args()
    cv2.setNumThreads(2)
    args.output.mkdir(parents=True, exist_ok=True)
    records, split = records_for(args.annotations)

    pool = json.loads(args.baseline_pool.read_text())
    parity = evaluate_thresholds(pool, records, (0.35,), grow=-1, min_area=300)[0]
    if abs(parity["pq"] - BASELINE_PQ) > 1e-9:
        raise SystemExit(f"evaluator does not reproduce the exp062 baseline: {parity}")
    emit("PARITY", dict(pq=parity["pq"], tp=parity["tp"], fp=parity["fp"], fn=parity["fn"]))

    manifest = json.loads((args.weights_dir / "manifest.json").read_text())
    checkpoints = [args.weights_dir / name for name in sorted(manifest) if name.endswith(".pt")]
    for path in checkpoints:
        if sha256(path) != manifest[path.name]["sha256"]:
            raise SystemExit(f"checkpoint corrupted in transfer: {path.name}")

    from ultralytics import YOLO
    report = dict(split=split.summary(), baseline=dict(pq=BASELINE_PQ, parity=parity),
                  single_view={}, d4={})
    for path in checkpoints:
        started = time.time()
        model = YOLO(str(path))
        cache = {name: single_view(model, args.images / name) for name in records}
        rows = sweep(cache, records, SINGLE_CONFS)
        report["single_view"][path.name] = dict(best=rows[0], top=rows[:8],
                                                seconds=round(time.time() - started, 1))
        emit("SINGLE_VIEW", dict(checkpoint=path.name, **rows[0]))
        (args.output / "report.json").write_text(json.dumps(report, indent=2))

    ranked = sorted(report["single_view"], key=lambda name: -report["single_view"][name]["best"]["pq"])
    for name in ranked[:args.d4_top]:
        started = time.time()
        model = YOLO(str(args.weights_dir / name))
        cache = {photo: d4(model, args.images / photo) for photo in records}
        rows = sweep(cache, records, D4_CONFS)
        report["d4"][name] = dict(best=rows[0], top=rows[:8], seconds=round(time.time() - started, 1))
        (args.output / f"d4_pool_{Path(name).stem}.json").write_text(json.dumps(cache))
        emit("D4", dict(checkpoint=name, delta_vs_baseline=rows[0]["pq"] - BASELINE_PQ, **rows[0]))
        (args.output / "report.json").write_text(json.dumps(report, indent=2))

    candidates = ([("single_view", n, v["best"]) for n, v in report["single_view"].items()]
                  + [("d4", n, v["best"]) for n, v in report["d4"].items()])
    mode, name, best = max(candidates, key=lambda item: item[2]["pq"])
    report["best"] = dict(mode=mode, checkpoint=name, rule=best,
                          delta_vs_baseline=best["pq"] - BASELINE_PQ)
    (args.output / "report.json").write_text(json.dumps(report, indent=2))
    emit("SUMMARY", report["best"])


if __name__ == "__main__":
    main()
