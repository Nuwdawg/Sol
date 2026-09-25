"""exp097 preflight: find the fastest training configuration before the real run.

Why a preflight
---------------
The goal is to train as fast as the hardware allows. At 2048 px two things decide
speed, and neither can be known off-pod: how many images fit in memory at once,
and whether the CPUs can decode and augment 4-megapixel images fast enough to
keep the GPU fed. Guessing either wastes paid time. Too small a batch leaves
compute idle; too large a batch crashes hours later on a crowded image.

What it measures
----------------
For each candidate batch size, in a fresh subprocess, so a failed allocation
cannot pollute the next attempt (CLAUDE.md section 3), it runs a short training
burst on the most instance-dense training samples. Those are the worst case for
mask-loss memory. The burst uses the real model, real augmentation, RAM caching
and worker count. After a warmup (cuDNN autotuning, allocator growth) it measures
at least `measure_images` images and records:

- steady images per second;
- the fraction of loop time spent waiting on the dataloader;
- peak reserved CUDA memory, which covers the validation pass that closes the
  burst, because that pass runs one full validation batch of the densest images;
- nvidia-smi utilization and power over exactly the steady window.

Candidates run in ascending order and stop at the first out-of-memory or
headroom violation.

Headroom 0.92, from evidence. On the first pod (NVIDIA L40, 45,468 MiB),
YOLOv8l-seg at 2048 peaked at 39,418 MiB (86.7%) at batch 2 on the densest images.
Those images are the dataset's worst case. A 0.85 rule rejected the only batch
that fits a 48 GB card. Batch 3 would need about 59 GB, so it OOMs and stops the
ladder.

Decision
--------
It takes the fastest batch that fits within `headroom` of total memory. A larger
batch wins only on a near-tie (within `tie_tolerance`, the measurement noise),
since it gives BatchNorm steadier statistics at no speed cost. Utilization is
reported, never enforced. The report says what limits speed: the input pipeline
(more vCPUs would help) or the GPU (it is already the bottleneck). The preflight
fails only when no batch size fits at all.
"""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

from experiments.exp_097_runpod_yolo.src.events import emit
from experiments.exp_097_runpod_yolo.src.gpu_monitor import GPUMonitor

DEFAULT_BATCHES = "1,2,3,4,6,8,12,16,24"
DEFAULT_HEADROOM = 0.92


def build_repeated_subset(base: Path, target: Path, samples: int, val_samples: int) -> Path:
    """An isolated dataset that cycles the densest images for training and validation.

    Repeats are distinct symlinks, so one epoch is exactly warmup + measurement
    batches. Ultralytics validates on the final epoch even when `val=False`, with
    twice the training batch (8.4.153 `_build_train_pipeline`). The caller passes
    that validation batch as `val_samples`, so the measured peak memory includes
    one full validation batch of the densest images: the pass that ends every
    real training epoch.
    """
    order = json.loads((base / "order.json").read_text())
    if not order:
        raise RuntimeError(f"empty preflight base at {base}")
    if target.exists():
        shutil.rmtree(target)
    for split in ("train", "val"):
        (target / "images" / split).mkdir(parents=True)
        (target / "labels" / split).mkdir(parents=True)
    for split, count, tag in (("train", samples, "r"), ("val", val_samples, "v")):
        for index in range(count):
            stem = order[index % len(order)]["stem"]
            name = f"{stem}_{tag}{index}"
            (target / "images" / split / f"{name}.jpeg").symlink_to(
                (base / "images" / f"{stem}.jpeg").resolve())
            shutil.copy2(base / "labels" / f"{stem}.txt",
                         target / "labels" / split / f"{name}.txt")
    (target / "data.yaml").write_text(
        f"path: {target.resolve()}\ntrain: images/train\nval: images/val\nnames:\n  0: filament\n")
    return target / "data.yaml"


def measure_batches_for(batch: int, minimum_batches: int, measure_images: int) -> int:
    """Enough batches that every candidate measures at least `measure_images` images."""
    return max(minimum_batches, math.ceil(measure_images / batch))


def select(results: list[dict], headroom: float = DEFAULT_HEADROOM, tie_tolerance: float = 0.01,
           target_utilization: float = 70.0) -> dict:
    """Pure decision: the fastest configuration that fits. Utilization is informational."""
    fitting = [r for r in results
               if not r.get("oom") and r.get("images_per_second")
               and r.get("peak_reserved_mib") is not None and r.get("total_mib")
               and r["peak_reserved_mib"] <= headroom * r["total_mib"]]
    if not fitting:
        return dict(ok=False, choice=None,
                    reason="no candidate batch size completed within the memory headroom")
    fastest = max(r["images_per_second"] for r in fitting)
    choice = max((r for r in fitting if r["images_per_second"] >= (1 - tie_tolerance) * fastest),
                 key=lambda r: r["batch"])
    utilization = choice.get("gpu_utilization_mean")
    waiting = choice.get("data_wait_fraction") or 0.0
    if utilization is None:
        note = "no GPU utilization samples were recorded"
    elif utilization >= target_utilization:
        note = f"GPU utilization {utilization:.1f}% at the fastest setting"
    elif waiting > 0.20:
        note = (f"GPU utilization {utilization:.1f}% at the fastest setting; the input pipeline "
                f"limits speed (data wait {waiting:.0%}), so more vCPUs would train faster")
    else:
        note = (f"GPU utilization {utilization:.1f}% at the fastest setting; the GPU itself "
                f"limits speed at the largest batch that fits")
    return dict(ok=True, choice=choice, reason=note)


def project(choice: dict, train_images: int, price_per_hour: float,
            validation_overhead: float) -> dict:
    train_seconds = train_images / choice["images_per_second"]
    epoch_seconds = train_seconds * (1.0 + validation_overhead)
    cost = epoch_seconds / 3600.0 * price_per_hour
    return dict(train_images=train_images, epoch_train_seconds=round(train_seconds, 1),
                epoch_seconds_estimate=round(epoch_seconds, 1),
                usd_per_epoch_estimate=round(cost, 4),
                epochs_per_usd_estimate=round(1.0 / cost, 2) if cost else None,
                validation_overhead_assumed=validation_overhead)


def run_worker(args: argparse.Namespace) -> int:
    import torch
    from ultralytics import YOLO

    torch.backends.cudnn.benchmark = True
    monitor = GPUMonitor(csv_path=None, interval=1.0)
    monitor.start()
    target = args.warmup_batches + args.measure_batches
    stats = dict(batches=0, wait=0.0, last_end=None, t0=None, wall0=None, wait0=0.0,
                 images=0, t1=None, wall1=None)

    def batch_start(trainer):
        now = time.perf_counter()
        if stats["last_end"] is not None:
            stats["wait"] += now - stats["last_end"]

    def batch_end(trainer):
        stats["batches"] += 1
        if stats["batches"] % 4 == 0 or stats["batches"] in (args.warmup_batches, target):
            torch.cuda.synchronize()
        now = time.perf_counter()
        stats["last_end"] = now
        if stats["batches"] == args.warmup_batches:
            stats.update(t0=now, wall0=time.time(), wait0=stats["wait"])
        elif args.warmup_batches < stats["batches"] <= target:
            stats["images"] += trainer.batch_size
            if stats["batches"] == target:
                stats.update(t1=now, wall1=time.time())

    model = YOLO(args.model)
    model.add_callback("on_train_batch_start", batch_start)
    model.add_callback("on_train_batch_end", batch_end)
    result = dict(batch=args.batch, workers=args.workers, oom=False)
    try:
        model.train(data=args.data, imgsz=args.imgsz, batch=args.batch, workers=args.workers,
                    epochs=1, cache="ram", val=False, plots=False, save=False, amp=True,
                    deterministic=False, optimizer="AdamW", lr0=1e-4, lrf=1.0,
                    warmup_epochs=0, mosaic=0.0, close_mosaic=0, degrees=15.0,
                    translate=0.1, scale=0.5, flipud=0.5, fliplr=0.5, hsv_h=0.0,
                    hsv_s=0.0, hsv_v=0.4, patience=100000, project=args.project,
                    name=f"batch{args.batch}", exist_ok=True, verbose=False)
    except torch.cuda.OutOfMemoryError as error:
        result.update(oom=True, error=str(error)[:300])
    except RuntimeError as error:
        if "out of memory" not in str(error).lower():
            raise
        result.update(oom=True, error=str(error)[:300])
    finally:
        monitor.stop()
    result.update(
        total_mib=round(torch.cuda.get_device_properties(0).total_memory / 2**20, 1),
        peak_reserved_mib=round(torch.cuda.max_memory_reserved() / 2**20, 1),
        peak_allocated_mib=round(torch.cuda.max_memory_allocated() / 2**20, 1),
        batches=stats["batches"])
    if stats["t1"] is not None:
        seconds = stats["t1"] - stats["t0"]
        result.update(steady_seconds=round(seconds, 2),
                      images_per_second=stats["images"] / seconds,
                      data_wait_fraction=(stats["wait"] - stats["wait0"]) / seconds)
        gpu = monitor.summarize(stats["wall0"], stats["wall1"])
        result.update({f"gpu_{key}": value for key, value in gpu.items()})
    elif not result["oom"]:
        result["error"] = f"only {stats['batches']} of {target} batches completed"
    Path(args.result).write_text(json.dumps(result, indent=2))
    return 3 if result["oom"] else 0


def run_orchestrator(args: argparse.Namespace) -> int:
    workdir = Path(args.workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    results = []
    for batch in [int(b) for b in args.batches.split(",")]:
        measure = measure_batches_for(batch, args.measure_batches, args.measure_images)
        data = build_repeated_subset(Path(args.base), workdir / f"subset_b{batch}",
                                     batch * (args.warmup_batches + measure), 2 * batch)
        result_path = workdir / f"batch{batch}.json"
        result_path.unlink(missing_ok=True)
        command = [sys.executable, "-u", "-m", "experiments.exp_097_runpod_yolo.src.preflight",
                   "--worker", "--model", args.model, "--data", str(data), "--imgsz",
                   str(args.imgsz), "--batch", str(batch), "--workers", str(args.workers),
                   "--warmup-batches", str(args.warmup_batches), "--measure-batches",
                   str(measure), "--result", str(result_path),
                   "--project", str(workdir / "runs")]
        try:
            completed = subprocess.run(command, capture_output=True, text=True,
                                       timeout=args.timeout)
            output = completed.stdout + completed.stderr
            code = completed.returncode
        except subprocess.TimeoutExpired as error:
            output, code = f"timeout after {args.timeout}s: {error}", -1
        (workdir / f"batch{batch}.log").write_text(output)
        if result_path.exists():
            result = json.loads(result_path.read_text())
        else:
            result = dict(batch=batch, oom="out of memory" in output.lower(),
                          error=f"exit {code}: {output.strip()[-300:]}")
        results.append(result)
        emit("PREFLIGHT_CANDIDATE", {k: result.get(k) for k in (
            "batch", "oom", "images_per_second", "data_wait_fraction",
            "gpu_utilization_mean", "gpu_power_fraction", "peak_reserved_mib",
            "total_mib", "steady_seconds", "error")}, args.log)
        if result.get("oom") or result.get("error"):
            break
        if result["peak_reserved_mib"] > args.headroom * result["total_mib"]:
            break
    decision = select(results, args.headroom, target_utilization=args.target_utilization)
    report = dict(decision=decision, candidates=results,
                  settings=dict(model=args.model, imgsz=args.imgsz, workers=args.workers,
                                warmup_batches=args.warmup_batches,
                                measure_batches=args.measure_batches,
                                measure_images=args.measure_images,
                                headroom=args.headroom,
                                target_utilization=args.target_utilization,
                                vcpus=os.cpu_count()))
    if decision["choice"] is not None and decision["choice"].get("images_per_second"):
        report["projection"] = project(decision["choice"], args.train_images,
                                       args.price_per_hour, args.validation_overhead)
    Path(args.output).write_text(json.dumps(report, indent=2))
    emit("PREFLIGHT", dict(ok=decision["ok"], reason=decision["reason"],
                           batch=(decision["choice"] or {}).get("batch"),
                           projection=report.get("projection")), args.log)
    # Non-zero only when nothing fits in memory; low utilization never blocks training.
    return 0 if decision["ok"] else 4


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--model", required=True)
    parser.add_argument("--imgsz", type=int, default=2048)
    parser.add_argument("--workers", type=int, required=True)
    parser.add_argument("--warmup-batches", type=int, default=6)
    parser.add_argument("--measure-batches", type=int, default=20)
    # worker-only
    parser.add_argument("--data")
    parser.add_argument("--batch", type=int)
    parser.add_argument("--result")
    parser.add_argument("--project")
    # orchestrator-only
    parser.add_argument("--base", help="densest-sample directory from fetch_data")
    parser.add_argument("--workdir")
    parser.add_argument("--batches", default=DEFAULT_BATCHES)
    parser.add_argument("--headroom", type=float, default=DEFAULT_HEADROOM)
    parser.add_argument("--measure-images", type=int, default=240,
                        help="minimum images per steady measurement window")
    parser.add_argument("--target-utilization", type=float, default=70.0,
                        help="reported against, never enforced")
    parser.add_argument("--train-images", type=int, default=974)
    parser.add_argument("--price-per-hour", type=float, default=0.0)
    parser.add_argument("--validation-overhead", type=float, default=0.25)
    parser.add_argument("--timeout", type=int, default=1200)
    parser.add_argument("--output")
    parser.add_argument("--log", type=Path, default=None)
    args = parser.parse_args()
    if args.worker:
        os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
        return run_worker(args)
    return run_orchestrator(args)


if __name__ == "__main__":
    # Allocator settings only take effect when set before torch is imported.
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    sys.exit(main())
