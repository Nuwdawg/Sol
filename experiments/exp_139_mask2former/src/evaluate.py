"""exp139: validation PQ for Mask2Former checkpoints, on the exp033 scoring path.

Two stages, separable so the pod never idles its GPU on CPU work:

1. `cache`: GPU inference over the 106 validation photographs at native 2048,
   single view or D4, writing candidates {score, rle} per photograph.
2. `sweep`: exp033's `evaluate_thresholds` over the emission grid exp097 used
   (erosion {none, 1 px} x minimum area {100, 300}) and a confidence grid, pooled
   per annotator record. One process per emission setting.

Before a sweep is trusted, `--parity-pool` re-scores the stored exp002 D4 pool and
must reproduce exp062's 0.4558638687 exactly, as exp097's evaluator required. A
Mask2Former number is then comparable with exp119's 0.4681455 by construction.
"""
from __future__ import annotations

import argparse
import json
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import torch

from experiments.exp_033_inference.src.sweep import evaluate_thresholds, records_for
from experiments.exp_139_mask2former.src.data import read_gray
from experiments.exp_139_mask2former.src.model import build_model, predict_photograph

BASELINE_PQ = 0.4558638687292126          # exp062: exp002 D4, conf 0.35, erosion 1, area 300
REFERENCE_PQ = 0.46814553                 # exp119: the current best, public 0.39
CONFS = tuple(round(0.05 * i, 2) for i in range(1, 20))
EMISSION = [(grow, area) for grow in (0, -1) for area in (100, 300)]


def event(tag: str, log: Path | None = None, **payload) -> None:
    line = f"{tag} {json.dumps(payload, sort_keys=True, default=str)}"
    print(line, flush=True)
    if log is not None:
        with Path(log).open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")


def load_checkpoint(path: Path, device: torch.device):
    state = torch.load(path, map_location="cpu", weights_only=False)
    model = build_model(config=state["config"])
    model.load_state_dict(state["model"], strict=True)
    return model.to(device).eval(), state


def cache_validation(model, records: dict, images: Path, device: torch.device, bf16: bool,
                     d4: bool = False, limit: int = 0, floor: float = 0.02) -> tuple[dict, float]:
    names = sorted(records)[:limit] if limit else sorted(records)
    started = time.time()
    cache = {}
    for name in names:
        cache[name] = predict_photograph(model, read_gray(images / name), device, bf16, d4=d4, floor=floor)
    return cache, time.time() - started


def _sweep_one(job: tuple) -> list[dict]:
    cache_path, annotations, confs, grow, area, names = job
    records, _ = records_for(annotations)
    if names:
        records = {name: records[name] for name in names}
    cache = json.loads(Path(cache_path).read_text())
    return evaluate_thresholds(cache, records, confs, grow=grow, min_area=area)


def sweep(cache_path: Path, annotations: Path, processes: int = 4, confs=CONFS,
          names: list[str] | None = None) -> list[dict]:
    jobs = [(str(cache_path), str(annotations), tuple(confs), grow, area, names)
            for grow, area in EMISSION]
    with ProcessPoolExecutor(max_workers=max(1, min(processes, len(jobs)))) as pool:
        rows = [row for part in pool.map(_sweep_one, jobs) for row in part]
    return sorted(rows, key=lambda row: -row["pq"])


def parity(pool_path: Path, annotations: Path) -> dict:
    records, _ = records_for(annotations)
    pool = json.loads(Path(pool_path).read_text())
    row = evaluate_thresholds(pool, records, (0.35,), grow=-1, min_area=300)[0]
    if abs(row["pq"] - BASELINE_PQ) > 1e-9:
        raise SystemExit(f"evaluator does not reproduce the exp062 baseline: {row}")
    return row


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--images", type=Path, default=None)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--cache", type=Path, default=None, help="sweep an existing cache")
    parser.add_argument("--tag", required=True)
    parser.add_argument("--d4", action="store_true")
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--msda", choices=("pytorch", "hub"), default="pytorch")
    parser.add_argument("--msda-revision", default=None)
    parser.add_argument("--limit", type=int, default=0, help="first N photographs, smoke tests")
    parser.add_argument("--parity-pool", type=Path, default=None)
    parser.add_argument("--processes", type=int, default=4)
    parser.add_argument("--log", type=Path, default=None)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    if args.parity_pool is not None:
        row = parity(args.parity_pool, args.annotations)
        event("PARITY", args.log, pq=row["pq"], tp=row["tp"], fp=row["fp"], fn=row["fn"])
        if args.cache is None and args.checkpoint is None:
            return

    records, split = records_for(args.annotations)
    names = sorted(records)[:args.limit] if args.limit else None
    cache_path = args.cache
    report = dict(tag=args.tag, split=split.summary(), d4=args.d4)
    if cache_path is None:
        if args.checkpoint is None or args.images is None:
            raise SystemExit("give --cache, or --checkpoint with --images")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if args.msda != "pytorch":
            from experiments.exp_139_mask2former.src.fast_ops import (install_deformable_kernel,
                                                                      verify_deformable_kernel)
            kernel = install_deformable_kernel(args.msda, args.msda_revision)
            rows = verify_deformable_kernel(device)
            event("KERNEL", args.log, **kernel, verification=rows)
            if not all(row["passed"] for row in rows):
                raise SystemExit("deformable attention kernel differs from the reference")
        model, state = load_checkpoint(args.checkpoint, device)
        cache, seconds = cache_validation(model, records, args.images, device, args.bf16,
                                          args.d4, args.limit)
        cache_path = args.output / f"{args.tag}_cache.json"
        cache_path.write_text(json.dumps(cache))
        report.update(checkpoint=str(args.checkpoint), epoch=state.get("epoch"),
                      step=state.get("step"), inference_seconds=round(seconds, 1),
                      candidates=sum(len(entries) for entries in cache.values()))
    started = time.time()
    rows = sweep(cache_path, args.annotations, args.processes, names=names)
    report.update(sweep_seconds=round(time.time() - started, 1), best=rows[0], rows=rows,
                  photographs=len(names) if names else len(records),
                  reference=dict(exp119=REFERENCE_PQ))
    (args.output / f"{args.tag}_sweep.json").write_text(json.dumps(report, indent=2))
    event("EVAL", args.log, run=args.tag, d4=args.d4, best=rows[0],
          photographs=report["photographs"], sweep_seconds=report["sweep_seconds"])


if __name__ == "__main__":
    main()
