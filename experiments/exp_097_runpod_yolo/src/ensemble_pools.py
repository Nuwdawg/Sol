"""Fuse precomputed D4 pools from different models, on CPU, and score the result.

Why this exists
---------------
The two models we have disagree in a useful way. exp002 (yolo11m-seg trained at
1280) finds 860 filaments on validation with 397 false positives. exp097's
best.pt (YOLOv8l-seg trained at 2048) finds 819 with 269. Every measurement since
exp098 says the binding constraint is recognition, not mask quality, so the
cheapest remaining gain is a filament that one architecture found and the other
missed.

Both models' D4 pools already exist as JSON - exp002's in the repository,
exp097's in exp098's kernel output - so this needs no GPU and no inference. It is
RLE arithmetic over 180 photographs and runs in minutes.

Instance-level fusion, and why the denominator is the model count
-----------------------------------------------------------------
These pools hold *fused instances*, one per filament per model, not raw views.
Each score already carries its own eight-view support factor. Clustering them and
dividing support by eight again would punish every instance for the seven views
it never had. Here a source is a model: an instance both architectures propose
keeps its full score, one proposed by a single architecture keeps 0.75 of it, and
the pixel vote then decides the boundary between them.

Each input pool is also swept alone, in the same run, so "the ensemble wins"
cannot be an artefact of comparing numbers from different runs (CLAUDE.md 5).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import cv2

from experiments.exp_033_inference.src.sweep import evaluate_thresholds, records_for
from experiments.exp_097_runpod_yolo.src.evaluate_ensemble import fuse_sources
from experiments.exp_097_runpod_yolo.src.events import emit

BASELINE_PQ = 0.4558638687292126      # exp062: exp002 D4, conf 0.35, erosion 1, area 300
CONFS = (0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60)
EMISSION = [(grow, area) for grow in (0, -1) for area in (100, 300)]


def load_pool(path: Path) -> dict[str, list[dict]]:
    """A pool is {photograph: [{score, rle}]}; counts stay text for JSON safety."""
    pool = json.loads(Path(path).read_text())
    cleaned = {}
    for photograph, entries in pool.items():
        rows = []
        for entry in entries:
            rle = entry["rle"]
            counts = rle["counts"]
            rows.append(dict(score=float(entry["score"]),
                             rle=dict(size=rle["size"],
                                      counts=counts.decode("ascii")
                                      if isinstance(counts, bytes) else counts)))
        cleaned[Path(photograph).name] = rows
    return cleaned


def sweep(pool: dict, records: dict) -> list[dict]:
    rows = []
    for grow, area in EMISSION:
        rows.extend(evaluate_thresholds(pool, records, CONFS, grow=grow, min_area=area))
    return sorted(rows, key=lambda row: -row["pq"])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pool", action="append", required=True,
                        help="label=path, repeatable; each is one model's D4 pool")
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--link", type=float, default=0.5)
    parser.add_argument("--vote", type=float, default=0.5)
    args = parser.parse_args()
    cv2.setNumThreads(2)
    args.output.mkdir(parents=True, exist_ok=True)
    records, split = records_for(args.annotations)

    pools = {}
    for item in args.pool:
        label, _, path = item.partition("=")
        pools[label] = load_pool(Path(path))
        emit("POOL", dict(label=label, path=path, photographs=len(pools[label]),
                          instances=sum(len(v) for v in pools[label].values())))
    missing = {label: [name for name in records if name not in pool]
               for label, pool in pools.items()}
    if any(missing.values()):
        raise SystemExit(f"pools do not cover the validation split: "
                         f"{ {k: len(v) for k, v in missing.items()} }")

    report = dict(split=split.summary(), baseline=BASELINE_PQ, link=args.link,
                  vote=args.vote, sources=len(pools), single={}, ensemble={})
    started = time.time()
    for label, pool in pools.items():
        rows = sweep(pool, records)
        report["single"][label] = dict(best=rows[0], top=rows[:6],
                                       delta_vs_baseline=rows[0]["pq"] - BASELINE_PQ)
        emit("SINGLE", dict(label=label, **rows[0]))
        (args.output / "report.json").write_text(json.dumps(report, indent=2))

    fused = {}
    for photograph in sorted(records):
        entries = [(entry["score"], entry["rle"])
                   for pool in pools.values() for entry in pool[photograph]]
        fused[photograph] = fuse_sources(entries, args.link, args.vote, len(pools))
    rows = sweep(fused, records)
    best_single = max(block["best"]["pq"] for block in report["single"].values())
    report["ensemble"] = dict(best=rows[0], top=rows[:8],
                              delta_vs_best_single=rows[0]["pq"] - best_single,
                              delta_vs_baseline=rows[0]["pq"] - BASELINE_PQ,
                              seconds=round(time.time() - started, 1))
    (args.output / "ensemble_pool.json").write_text(json.dumps(fused))
    (args.output / "report.json").write_text(json.dumps(report, indent=2))
    emit("ENSEMBLE", dict(labels=sorted(pools), **rows[0],
                          delta_vs_best_single=report["ensemble"]["delta_vs_best_single"],
                          delta_vs_baseline=report["ensemble"]["delta_vs_baseline"]))
    emit("SUMMARY", dict(best_single=best_single, ensemble=rows[0]["pq"],
                         wins=rows[0]["pq"] > best_single,
                         clears_040_bar=rows[0]["pq"] >= 0.472))


if __name__ == "__main__":
    main()
