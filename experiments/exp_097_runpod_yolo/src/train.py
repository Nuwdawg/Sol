"""exp097 training driver: YOLOv8l-seg at native 2048, trained until converged.

What this run is for
--------------------
The 0.39 public submission is a D4-TTA emission of a yolo11m-seg that was trained
at 1280 and inferred at 2048. exp095 showed the 0.55 team's advantage is in its
weights, not its inference recipe; their model file is a YOLOv8l-seg trained at
2048. This run trains that architecture at native resolution, from COCO-pretrained
weights, on the canonical split, until validation stops improving. Every
inference-time technique from the 0.39 pipeline then applies to the result
unchanged.

How long it trains
------------------
Until converged, as defined in `convergence.py`: a constant learning rate after
warmup, a 5x decay each time smoothed validation fitness plateaus for 12 epochs,
and a stop only when a plateau survives three decays. `epochs` is set to a
ceiling Ultralytics requires and that the run cannot reach. There is no `time=`
and no schedule tied to an epoch count. The one non-convergence stop is the
user's money cap; the log says so if it fires.

How the decay is applied
------------------------
Ultralytics runs a LambdaLR whose factor is 1.0 throughout, because `lrf=1.0`
with `cos_lr=False`. The callback scales that scheduler's `base_lrs` and every
parameter group's `lr` and `initial_lr`, so the next `scheduler.step()` keeps
the new rate. At the first batch of every epoch after warmup, the driver reads
the rate the optimizer actually uses and asserts it against the controller's
intent. A silent override would stop the run instead of quietly training on the
wrong schedule (CLAUDE.md section 2).

Ultralytics 8.4.153 behaviour this driver relies on, read from its source
-------------------------------------------------------------------------
Pod qsatuc7fdharyi crashed in `on_train_start` because an earlier version of
this file assumed these instead of reading them. `smoke/smoke.py` runs this file
against the pinned library on a Kaggle T4 before any pod starts.

- A complete RAM cache replaces `dataset.ims` with an `_ImageCache` that packs
  every image into one buffer. It has no `len()`; `shapes` has one row per
  image. When RAM looks short, `check_cache_ram` sets `dataset.cache` to None
  and `ims` stays a plain list.
- `scheduler.step()` runs at the top of each epoch, after
  `on_train_epoch_start`. Warmup then overrides the rate for every iteration
  below `_get_warmup_iterations(batches)`.
- `final_eval` validates best.pt after the loop and calls `on_fit_epoch_end`
  again with `trainer.epoch` advanced by one. The driver records that call as
  the final validation, not as a training epoch.
- `tloss` is a dict of running mean losses keyed by loss name.
- An out-of-memory error in the first epoch halves the batch and rebuilds the
  loaders, optimizer and scheduler, with only a warning. The driver reports it
  as a STAGE event, checks the rebuilt pipeline again and records it in the
  summary.

Hyperparameters, and why
------------------------
- imgsz 2048, rect off, multi-scale off: native frame, the resolution every
  submission infers at.
- AdamW, lr0 1e-3, weight decay 5e-4: named explicitly, because
  `optimizer='auto'` replaces lr0 with its own choice.
- warmup 3 epochs with warmup_bias_lr 0: the default bias warmup of 0.1 is an SGD
  value and would spike biases under AdamW.
- nbs 64 (default): gradient accumulation keeps the effective batch at 64
  whatever batch size the preflight chooses.
- Augmentation as in exp002 (degrees 15, translate 0.1, scale 0.5, flips 0.5,
  mosaic off): the dihedral TTA and D4 fusion assume flip and rotation
  invariance.
- hsv_h 0 and hsv_s 0: the images are grayscale repeated to three channels, so
  hue and saturation jitter cost CPU and change nothing; hsv_v 0.4 keeps exposure
  jitter, which is real across GONG sites.
- cache 'ram': removes JPEG decoding from the input path. It is asserted to be
  complete, because Ultralytics falls back to disk with only a warning when it
  judges RAM short.
- mask_ratio 4, overlap_mask on: stock mask supervision, identical to exp002.
- val every epoch: the convergence decision needs it. Checkpoint choice among
  stages is made later on Kaggle with exact PQ, not here.
- deterministic off, cudnn.benchmark on: fixed 2048 x 2048 shapes make autotuned
  kernels a pure speed gain.

Artifacts, all uploaded to the private Kaggle dataset
----------------------------------------------------
best.pt and last.pt (optimizer stripped), stage<k>_best.pt at every decay,
epochs.jsonl (one record per epoch: losses, metrics, learning rate, decision,
throughput, GPU utilization and power, spend), resolved_args.json, args.yaml,
results.csv, gpu_samples.csv, summary.json, preflight.json, data_manifest.json and
the full training log.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import queue
import shutil
import threading
import time
import traceback

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

from experiments.exp_097_runpod_yolo.src.convergence import PlateauConfig, PlateauController
from experiments.exp_097_runpod_yolo.src.events import emit
from experiments.exp_097_runpod_yolo.src.gpu_monitor import GPUMonitor
from experiments.exp_097_runpod_yolo.src.kaggle_sync import publish, verify
from experiments.exp_097_runpod_yolo.src.telemetry import EpochTelemetry


def scalar(value):
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        try:
            return float(value.item())
        except Exception:
            return None


def plain(value):
    if isinstance(value, (int, float, str, bool)) or value is None:
        return value
    return str(value)


def cache_report(dataset) -> dict:
    """How much of the training set Ultralytics holds in RAM, read from the dataset itself.

    A complete RAM cache is an `_ImageCache`: no `len()`, one `shapes` row per
    image. Anything else is a list whose non-None entries happen to be loaded.
    """
    images = getattr(dataset, "ims", None)
    if hasattr(images, "shapes"):
        cached = len(images.shapes)
    else:
        cached = sum(image is not None for image in images or [])
    return dict(mode=getattr(dataset, "cache", None), container=type(images).__name__,
                cached=cached, samples=len(dataset.im_files))


class Uploader(threading.Thread):
    """Serialises mid-run uploads so training never waits on the network."""

    def __init__(self, dataset: str, title: str, staging: Path, log: Path):
        super().__init__(daemon=True, name="kaggle-uploader")
        self.dataset, self.title, self.staging, self.log = dataset, title, staging, log
        self.jobs: queue.Queue = queue.Queue()
        self.results: list[dict] = []

    def submit(self, files: dict, message: str) -> None:
        self.jobs.put((dict(files), message))

    def run(self) -> None:
        while True:
            files, message = self.jobs.get()
            try:
                upload = publish(files, self.dataset, self.title, message, self.staging)
                self.results.append(upload)
                emit("UPLOAD", dict(dataset=self.dataset, message=message,
                                    action=upload["action"], bytes=upload["bytes"],
                                    files=len(upload["files"]), seconds=upload["seconds"]),
                     self.log)
            except Exception as error:  # a failed mid-run upload must not kill training
                emit("UPLOAD_ERROR", dict(message=message, error=str(error)[:400]), self.log)
            finally:
                self.jobs.task_done()

    def drain(self) -> None:
        self.jobs.join()


def snapshot(source: Path, target: Path) -> Path:
    """Copy a checkpoint and strip its optimizer state, about 90 MB for YOLOv8l-seg."""
    from ultralytics.utils.torch_utils import strip_optimizer
    shutil.copy2(source, target)
    strip_optimizer(str(target))
    return target


def parse() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--imgsz", type=int, default=2048)
    parser.add_argument("--batch", type=int, required=True)
    parser.add_argument("--workers", type=int, required=True)
    parser.add_argument("--cache", default="ram")
    parser.add_argument("--lr0", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    parser.add_argument("--warmup-epochs", type=float, default=3.0)
    parser.add_argument("--patience", type=int, default=12)
    parser.add_argument("--min-delta", type=float, default=0.002)
    parser.add_argument("--decay-factor", type=float, default=0.2)
    parser.add_argument("--max-decays", type=int, default=3)
    parser.add_argument("--smoothing", type=int, default=3)
    parser.add_argument("--epoch-ceiling", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--project", required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--artifacts", type=Path, required=True)
    parser.add_argument("--budget-usd", type=float, required=True)
    parser.add_argument("--price-per-hour", type=float, required=True)
    parser.add_argument("--pod-start-epoch", type=float, required=True)
    parser.add_argument("--reserve-usd", type=float, default=0.50)
    parser.add_argument("--upload-dataset", default=None)
    parser.add_argument("--upload-title", default="sol exp097 yolov8l 2048 weights")
    parser.add_argument("--staging", type=Path, default=None)
    return parser.parse_args()


def main() -> int:
    args = parse()
    import torch
    from ultralytics import YOLO

    torch.backends.cudnn.benchmark = True
    artifacts = args.artifacts
    artifacts.mkdir(parents=True, exist_ok=True)
    events_log = artifacts / "events.log"
    epochs_log = artifacts / "epochs.jsonl"
    run_dir = Path(args.project) / args.name
    staging = args.staging or artifacts.parent / "staging"

    monitor = GPUMonitor(artifacts / "gpu_samples.csv", interval=2.0)
    monitor.start()
    controller = PlateauController(PlateauConfig(
        patience=args.patience, min_delta=args.min_delta, decay_factor=args.decay_factor,
        max_decays=args.max_decays, smoothing=args.smoothing,
        warmup_epochs=int(round(args.warmup_epochs))))
    telemetry = EpochTelemetry(synchronize=torch.cuda.synchronize, sync_every=25)
    uploader = None
    if args.upload_dataset:
        uploader = Uploader(args.upload_dataset, args.upload_title, staging, events_log)
        uploader.start()
    state = dict(stop_reason=None, epoch_wall=None, train_end_wall=None, epoch_index=None,
                 epoch_open=False, rate_checked=True, batch=None, batches=None,
                 warmup_iterations=None, stage=0, batch_reductions=[], final_validation=None)

    def artifact_files(final: bool = False) -> dict:
        files = {name: artifacts / name for name in (
            "preflight.json", "data_manifest.json", "resolved_args.json", "epochs.jsonl",
            "gpu_samples.csv", "events.log", "summary.json", "error.json", "train.log")}
        files["args.yaml"] = run_dir / "args.yaml"
        files["results.csv"] = run_dir / "results.csv"
        for path in sorted(artifacts.glob("stage*_best.pt")):
            files[path.name] = path
        if final:
            files["best.pt"] = run_dir / "weights" / "best.pt"
            files["last.pt"] = run_dir / "weights" / "last.pt"
        return files

    def check_pipeline(trainer) -> dict:
        """Read back what Ultralytics built and raise on anything that was not asked for."""
        cache = cache_report(trainer.train_loader.dataset)
        if args.cache == "ram" and (cache["mode"] != "ram" or cache["cached"] != cache["samples"]):
            raise RuntimeError(f"RAM cache incomplete {cache}: Ultralytics fell back to disk reads")
        optimizer = type(trainer.optimizer).__name__
        if optimizer != "AdamW":
            raise RuntimeError(f"optimizer resolved to {optimizer}, not AdamW")
        initial = sorted({g.get("initial_lr", g["lr"]) for g in trainer.optimizer.param_groups})
        if any(abs(value - args.lr0) > 1e-12 for value in initial):
            raise RuntimeError(f"initial learning rates {initial} differ from lr0 {args.lr0}")
        batches = len(trainer.train_loader)
        state.update(batch=trainer.batch_size, batches=batches,
                     warmup_iterations=trainer._get_warmup_iterations(batches))
        return dict(cache=cache, optimizer=optimizer, initial_lr=initial,
                    batch=trainer.batch_size, batches_per_epoch=batches,
                    warmup_iterations=state["warmup_iterations"])

    def on_train_start(trainer):
        if trainer.batch_size != args.batch:
            raise RuntimeError(f"batch resolved to {trainer.batch_size}, not {args.batch}")
        pipeline = check_pipeline(trainer)
        (artifacts / "resolved_args.json").write_text(json.dumps(
            {key: plain(value) for key, value in vars(trainer.args).items()}, indent=2))
        emit("STAGE", dict(name="train_start", **pipeline, imgsz=plain(trainer.args.imgsz),
                           epoch_ceiling=trainer.epochs, save_dir=str(trainer.save_dir)),
             events_log)

    def on_train_epoch_start(trainer):
        telemetry.epoch_start()
        state.update(epoch_wall=time.time(), epoch_index=trainer.epoch, rate_checked=False)

    def on_train_batch_start(trainer):
        if trainer.batch_size != state["batch"]:
            # Ultralytics' first-epoch out-of-memory retry halved the batch and rebuilt the
            # loaders, optimizer and scheduler. Check the rebuilt pipeline, then say so loudly.
            previous = state["batch"]
            reduction = dict(epoch=trainer.epoch + 1, previous_batch=previous,
                             **check_pipeline(trainer))
            state["batch_reductions"].append(reduction)
            emit("STAGE", dict(name="batch_reduced", reason="out of memory in the first epoch",
                               **reduction), events_log)
        if not state["rate_checked"]:
            state["rate_checked"] = True
            # This epoch's scheduler.step() has run; past warmup nothing else sets the rate.
            if trainer.epoch * state["batches"] >= state["warmup_iterations"]:
                expected = args.lr0 * controller.lr_scale
                actual = [g["lr"] for g in trainer.optimizer.param_groups]
                if any(abs(value - expected) > 1e-9 + 1e-6 * expected for value in actual):
                    raise RuntimeError(f"learning rate drift in epoch {trainer.epoch + 1}: "
                                       f"expected {expected}, found {actual}")
        telemetry.batch_start(trainer.batch_size)

    def on_train_batch_end(trainer):
        telemetry.batch_end()

    def on_train_epoch_end(trainer):
        state.update(train_end_wall=time.time(), epoch_open=True)

    def on_fit_epoch_end(trainer):
        if not state["epoch_open"] or trainer.epoch != state["epoch_index"]:
            # final_eval's call: best.pt validated after the loop, trainer.epoch advanced by one.
            metrics = {key: scalar(value) for key, value in (trainer.metrics or {}).items()}
            state["final_validation"] = metrics
            emit("FINAL_VALIDATION", dict(
                weights=str(trainer.best), mask_map50_95=metrics.get("metrics/mAP50-95(M)"),
                mask_map50=metrics.get("metrics/mAP50(M)"),
                box_map50_95=metrics.get("metrics/mAP50-95(B)")), events_log)
            return
        state["epoch_open"] = False
        now = time.time()
        epoch = trainer.epoch + 1
        dataset_images = len(trainer.train_loader.dataset)
        train_seconds = state["train_end_wall"] - state["epoch_wall"]
        throughput = telemetry.summary().as_dict()
        throughput.update(dataset_images=dataset_images, train_seconds=round(train_seconds, 2),
                          validation_seconds=round(now - state["train_end_wall"], 2),
                          images_per_second=dataset_images / max(train_seconds, 1e-9))
        gpu_train = monitor.summarize(state["epoch_wall"], state["train_end_wall"])
        gpu_validation = monitor.summarize(state["train_end_wall"], now)
        metrics = {key: scalar(value) for key, value in (trainer.metrics or {}).items()}
        fitness = scalar(trainer.fitness)
        if fitness is None:
            raise RuntimeError("validation produced no fitness; convergence cannot be judged")
        tloss = getattr(trainer, "tloss", None)   # running mean losses keyed by name
        losses = ({name: scalar(value) for name, value in tloss.items()}
                  if isinstance(tloss, dict) else {})
        rates = [g["lr"] for g in trainer.optimizer.param_groups]

        decision = controller.update(epoch, fitness)
        if decision.action == "decay":
            new_rate = args.lr0 * controller.lr_scale
            for group in trainer.optimizer.param_groups:
                group["lr"] = new_rate
                group["initial_lr"] = new_rate
            trainer.scheduler.base_lrs = [new_rate for _ in trainer.scheduler.base_lrs]
            state["stage"] += 1
            best = Path(trainer.best)
            if best.exists():
                snapshot(best, artifacts / f"stage{state['stage']}_best.pt")
            if uploader is not None:
                uploader.submit(artifact_files(), f"stage {state['stage']} ended at epoch {epoch}")
        elif decision.action == "stop":
            trainer.stop = True
            state["stop_reason"] = "converged"

        # Utilization is recorded below for diagnosis only; it never stops training.
        spent = (now - args.pod_start_epoch) / 3600.0 * args.price_per_hour
        epoch_cost = (now - state["epoch_wall"]) / 3600.0 * args.price_per_hour
        if spent + epoch_cost + args.reserve_usd >= args.budget_usd and not trainer.stop:
            trainer.stop = True
            state["stop_reason"] = "budget"

        record = dict(epoch=epoch, fitness=fitness, metrics=metrics, losses=losses,
                      learning_rates=rates, decision=decision.as_dict(),
                      throughput=throughput, gpu_train=gpu_train,
                      gpu_validation=gpu_validation, spent_usd=round(spent, 3),
                      stop=bool(trainer.stop), stop_reason=state["stop_reason"])
        with epochs_log.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, default=str) + "\n")
        emit("EPOCH", dict(
            epoch=epoch, fitness=round(fitness, 5),
            mask_map50_95=metrics.get("metrics/mAP50-95(M)"),
            mask_map50=metrics.get("metrics/mAP50(M)"),
            box_map50_95=metrics.get("metrics/mAP50-95(B)"),
            action=decision.action, reason=decision.reason, decays=decision.decays,
            lr=rates[0] if rates else None, smoothed=decision.smoothed,
            best_epoch=decision.best_epoch, since_best=decision.epochs_since_best,
            images_per_second=round(throughput["images_per_second"], 2),
            data_wait=round(throughput["data_wait_fraction"], 3),
            gpu_util=gpu_train.get("utilization_mean"), gpu_util_p10=gpu_train.get("utilization_p10"),
            power_fraction=gpu_train.get("power_fraction"), vram_mib=gpu_train.get("memory_max_mib"),
            epoch_seconds=round(now - state["epoch_wall"], 1), spent_usd=round(spent, 3),
            stop_reason=state["stop_reason"]), events_log)

    def on_train_end(trainer):
        reason = state["stop_reason"] or (
            "epoch_ceiling" if trainer.epoch + 1 >= trainer.epochs else "ultralytics")
        summary = dict(
            stop_reason=reason, epochs_completed=len(controller.history),
            controller=controller.state(),
            ultralytics_best_fitness=scalar(getattr(trainer, "best_fitness", None)),
            final_validation=state["final_validation"],
            weights=dict(best=str(trainer.best), last=str(trainer.last)),
            stages=state["stage"], batch=state["batch"],
            batch_reductions=state["batch_reductions"], price_per_hour=args.price_per_hour,
            spent_usd_at_train_end=round((time.time() - args.pod_start_epoch) / 3600.0
                                         * args.price_per_hour, 3))
        (artifacts / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
        emit("STAGE", dict(name="train_end", stop_reason=reason,
                           epochs=summary["epochs_completed"],
                           best_fitness=summary["ultralytics_best_fitness"]), events_log)

    model = YOLO(args.model)
    for event, callback in (("on_train_start", on_train_start),
                            ("on_train_epoch_start", on_train_epoch_start),
                            ("on_train_batch_start", on_train_batch_start),
                            ("on_train_batch_end", on_train_batch_end),
                            ("on_train_epoch_end", on_train_epoch_end),
                            ("on_fit_epoch_end", on_fit_epoch_end),
                            ("on_train_end", on_train_end)):
        model.add_callback(event, callback)

    exit_code = 0
    try:
        model.train(
            data=args.data, imgsz=args.imgsz, batch=args.batch, workers=args.workers,
            epochs=args.epoch_ceiling, patience=100000, cache=args.cache,
            optimizer="AdamW", lr0=args.lr0, lrf=1.0, cos_lr=False, momentum=0.937,
            weight_decay=args.weight_decay, warmup_epochs=args.warmup_epochs,
            warmup_bias_lr=0.0, warmup_momentum=0.8, nbs=64,
            amp=True, deterministic=False, seed=args.seed, rect=False, multi_scale=False,
            mosaic=0.0, close_mosaic=0, mixup=0.0, copy_paste=0.0,
            degrees=15.0, translate=0.1, scale=0.5, shear=0.0, perspective=0.0,
            flipud=0.5, fliplr=0.5, hsv_h=0.0, hsv_s=0.0, hsv_v=0.4,
            overlap_mask=True, mask_ratio=4, dropout=0.0,
            val=True, plots=False, save=True, save_period=-1,
            project=args.project, name=args.name, exist_ok=True, verbose=True)
    except Exception as error:
        exit_code = 1
        (artifacts / "error.json").write_text(json.dumps(dict(
            error=str(error), traceback=traceback.format_exc()), indent=2))
        emit("ERROR", dict(error=str(error)[:500]), events_log)
        last = run_dir / "weights" / "last.pt"
        if last.exists():
            snapshot(last, artifacts / "stage_error_last.pt")
    finally:
        monitor.stop()

    if uploader is not None:
        uploader.drain()
        files = artifact_files(final=True)
        for path in sorted(artifacts.glob("stage_error_*.pt")):
            files[path.name] = path
        try:
            upload = publish(files, args.upload_dataset, args.upload_title,
                             f"final ({'error' if exit_code else 'complete'})", staging)
            upload["verification"] = verify(args.upload_dataset, upload["files"], timeout=300)
            (artifacts / "final_upload.json").write_text(json.dumps(upload, indent=2))
            emit("UPLOAD", dict(dataset=args.upload_dataset, message="final",
                                bytes=upload["bytes"], files=len(upload["files"]),
                                verification=upload["verification"]), events_log)
            if not upload["verification"]["verified"] and exit_code == 0:
                exit_code = 5
        except Exception as error:
            emit("UPLOAD_ERROR", dict(message="final", error=str(error)[:500]), events_log)
            exit_code = exit_code or 6
    emit("SUMMARY", dict(exit_code=exit_code, artifacts=str(artifacts)), events_log)
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
