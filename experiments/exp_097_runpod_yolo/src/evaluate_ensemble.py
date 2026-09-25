"""Score an ensemble of exp097 checkpoints against each of them alone, in one run.

Why an ensemble at all
----------------------
exp098 measured that the deficit against the exp002 baseline is recognition, not
mask quality: stage1 found 788 filaments where the baseline found 860, while its
masks were better (SQ 0.6899 against 0.6843). Twenty-five more epochs moved
stage2 to 819. Checkpoints from different stages disagree about marginal
filaments, and a filament that one checkpoint misses is often one another keeps,
so pooling their proposals is the cheapest remaining route to recall.

How the pooling works, and why it is not `cached_tta.fuse`
----------------------------------------------------------
The D4 pipeline clusters eight dihedral views of one model and keeps a pixel when
enough of the views that saw an instance agree. `cached_tta.fuse` divides cluster
support by a hardcoded eight, so feeding it a union of two models' instances
would score every cluster as if six views had missed it.

This module pools the **raw views of every checkpoint** - eight per model - and
fuses once with the denominator set to `8 x models`. A second checkpoint is
treated as eight more views of the same sun, which is what it is. Each
checkpoint's own eight views are then fused separately inside the same run, from
the same predictions, so "the ensemble beats the single model" cannot be an
artefact of comparing two different runs.

Everything else is unchanged from exp098: the same parity check against the
stored exp002 pool, the same SHA256 verification of every checkpoint against the
pod's manifest, and the same emission sweep over confidence, erosion and minimum
area with exact pooled per-record PQ.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time

import cv2
import numpy as np
from pycocotools import mask as mu

from experiments.exp_007_tta.src.dihedral import TRANSFORMS, predict_views
from experiments.exp_033_inference.src.cached_tta import vote_rle
from experiments.exp_033_inference.src.sweep import evaluate_thresholds, records_for
from experiments.exp_097_runpod_yolo.src.events import emit

BASELINE_PQ = 0.4558638687292126      # exp062: exp002 D4, conf 0.35, erosion 1, area 300
CONFS = (0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50)
EMISSION = [(grow, area) for grow in (0, -1) for area in (100, 300)]
VIEWS = len(TRANSFORMS)               # eight dihedral views per checkpoint


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def fuse_sources(entries: list[tuple[float, dict]], link: float, vote: float,
                 sources: int) -> list[dict]:
    """`cached_tta.fuse` with the support denominator passed in rather than fixed at eight.

    `entries` are (score, RLE) pairs from every view of every checkpoint. The
    highest-scoring unclustered mask seeds a cluster, every mask overlapping it
    above `link` joins, and a pixel survives when `vote` of the cluster's members
    contain it. The cluster's score is scaled by how much of the available
    evidence supports it, so an instance seen by one view of sixteen is ranked
    below one seen by twelve.
    """
    if not entries:
        return []
    order = sorted(range(len(entries)), key=lambda index: -entries[index][0])
    rles = [entry[1] for entry in entries]
    ious = mu.iou(rles, rles, [0] * len(rles))
    used, output = set(), []
    for seed in order:
        if seed in used:
            continue
        members = [seed]
        used.add(seed)
        for other in order:
            if other not in used and ious[seed, other] >= link:
                members.append(other)
                used.add(other)
        binary = vote_rle([rles[index] for index in members],
                          max(1, round(vote * len(members))))
        if mu.area(binary) < 40:
            continue
        score = float(np.mean([entries[index][0] for index in members]))
        output.append(dict(score=score * (0.5 + 0.5 * len(members) / sources),
                           rle=binary))
    return output


def views_for(model, image_path: Path, floor: float, max_det: int) -> list[tuple[float, dict]]:
    """The eight dihedral views, with counts as text so pools stay JSON-serialisable."""
    return [(score, dict(size=rle["size"],
                         counts=rle["counts"].decode("ascii")
                         if isinstance(rle["counts"], bytes) else rle["counts"]))
            for score, rle in predict_views(model, image_path, 2048, floor, max_det)]


def sweep(pool: dict, records: dict) -> list[dict]:
    rows = []
    for grow, area in EMISSION:
        rows.extend(evaluate_thresholds(pool, records, CONFS, grow=grow, min_area=area))
    return sorted(rows, key=lambda row: -row["pq"])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights-dir", type=Path, required=True)
    parser.add_argument("--checkpoints", nargs="+", required=True)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--images", type=Path, required=True)
    parser.add_argument("--baseline-pool", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--link", type=float, default=0.5)
    parser.add_argument("--vote", type=float, default=0.5)
    parser.add_argument("--floor", type=float, default=0.25)
    parser.add_argument("--max-det", type=int, default=100)
    args = parser.parse_args()
    cv2.setNumThreads(2)
    args.output.mkdir(parents=True, exist_ok=True)
    records, split = records_for(args.annotations)

    # Parity first: a new number is comparable with the old one by construction.
    baseline = json.loads(args.baseline_pool.read_text())
    parity = evaluate_thresholds(baseline, records, (0.35,), grow=-1, min_area=300)[0]
    if abs(parity["pq"] - BASELINE_PQ) > 1e-9:
        raise SystemExit(f"evaluator does not reproduce the exp062 baseline: {parity}")
    emit("PARITY", dict(pq=parity["pq"], tp=parity["tp"], fp=parity["fp"], fn=parity["fn"]))

    manifest = json.loads((args.weights_dir / "manifest.json").read_text())
    paths = []
    for name in args.checkpoints:
        path = args.weights_dir / name
        digest = sha256(path)
        if digest != manifest[name]["sha256"]:
            raise SystemExit(f"checkpoint corrupted in transfer: {name}")
        paths.append(path)
    emit("CHECKPOINTS", dict(names=args.checkpoints, sources=VIEWS * len(paths)))

    from ultralytics import YOLO
    models = [(path.name, YOLO(str(path))) for path in paths]
    started = time.time()
    single_pools = {name: {} for name, _ in models}
    ensemble_pool = {}
    for position, photograph in enumerate(sorted(records), start=1):
        image_path = args.images / photograph
        pooled = []
        for name, model in models:
            views = views_for(model, image_path, args.floor, args.max_det)
            single_pools[name][photograph] = fuse_sources(views, args.link, args.vote, VIEWS)
            pooled.extend(views)
        ensemble_pool[photograph] = fuse_sources(pooled, args.link, args.vote,
                                                 VIEWS * len(models))
        if position % 20 == 0 or position == len(records):
            emit("PROGRESS", dict(photographs=position, total=len(records),
                                  seconds=round(time.time() - started, 1)))

    report = dict(split=split.summary(), checkpoints=args.checkpoints,
                  link=args.link, vote=args.vote, sources=VIEWS * len(models),
                  baseline=dict(pq=BASELINE_PQ, parity=parity), single={}, ensemble={})
    for name in single_pools:
        rows = sweep(single_pools[name], records)
        report["single"][name] = dict(best=rows[0], top=rows[:6])
        emit("SINGLE", dict(checkpoint=name, **rows[0]))
        (args.output / "report.json").write_text(json.dumps(report, indent=2))

    rows = sweep(ensemble_pool, records)
    best_single = max((block["best"]["pq"] for block in report["single"].values()), default=0.0)
    report["ensemble"] = dict(best=rows[0], top=rows[:8],
                              delta_vs_best_single=rows[0]["pq"] - best_single,
                              delta_vs_baseline=rows[0]["pq"] - BASELINE_PQ)
    (args.output / "ensemble_pool.json").write_text(json.dumps(ensemble_pool))
    (args.output / "report.json").write_text(json.dumps(report, indent=2))
    emit("ENSEMBLE", dict(checkpoints=args.checkpoints, **report["ensemble"]["best"],
                          delta_vs_best_single=report["ensemble"]["delta_vs_best_single"],
                          delta_vs_baseline=report["ensemble"]["delta_vs_baseline"]))
    emit("SUMMARY", dict(best_single=best_single, ensemble=rows[0]["pq"],
                         wins=rows[0]["pq"] > best_single, seconds=round(time.time() - started, 1)))


if __name__ == "__main__":
    main()
