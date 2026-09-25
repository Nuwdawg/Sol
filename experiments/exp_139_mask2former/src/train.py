"""exp139: a properly trained Mask2Former for solar filament instances.

Every change corrects something read or measured about the previous run
(kaggle/exp046_mask2_train/train.py, docs/ACTIVE_EXPERIMENT.md):

| | exp046/exp058 | exp139 |
|---|---|---|
| length | ~4 epochs of a planned 50, cut by the session limit | a schedule sized to the budget before it starts, annealed to zero |
| backbone | swin-tiny, COCO instance | swin-small, COCO instance |
| input | whole image shrunk to 1024: stride-8 masks, cap IoU 0.771 | native pixels: stride-4 masks, cap 0.914 |
| loss | a custom Hungarian reimplementation | transformers 4.57.6's own Mask2Former loss |
| batch | 1 | 8 crop-equivalents |
| monitoring | none during training | validation PQ on the exp033 path every few epochs |

Optimiser: Mask2Former's `build_optimizer` - AdamW lr 1e-4, weight decay 0.05,
backbone lr x0.1, no decay on norms, embeddings and position tables, full-model
gradient clipping at norm 0.01. Schedule: linear warmup, then cosine to zero over
exactly the steps the run will take.

Why the schedule is sized in the run (CLAUDE.md section 2): exp_010's cosine was
sized for 400 epochs, stopped at 62, and never left its peak. Here the first
`--probe-steps` optimiser steps measure throughput while warmup is still running,
so no cosine value has been applied yet. The run then fixes the number of epochs
the budget affords, capped at `--epochs`, logs it as SCHEDULE, and refuses to run
below `--min-epochs` rather than quietly train a weaker model.

Validation during training: inference runs on the GPU between epochs; the CPU
sweep runs in a background process so training does not wait for it.
"""
from __future__ import annotations

import argparse
import json
import math
import queue
import subprocess
import sys
import threading
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from experiments.exp_033_inference.src.sweep import records_for
from experiments.exp_139_mask2former.src.data import (ExtentBatchSampler, FilamentSamples,
                                                      collate, gpu_pixels, load_records, read_gray)
from experiments.exp_139_mask2former.src.evaluate import cache_validation, event
from experiments.exp_139_mask2former.src.fast_ops import (install_batched_loss,
                                                          install_deformable_kernel,
                                                          verify_deformable_kernel)
from experiments.exp_139_mask2former.src.model import (backbone_checkpointing, build_model,
                                                       parameter_groups, predict)


def schedule(step: int, warmup: int, total: int) -> float:
    if step < warmup:
        return (step + 1) / warmup
    progress = min(1.0, (step - warmup) / max(1, total - warmup))
    return 0.5 * (1.0 + math.cos(math.pi * progress))


def atomic_save(path: Path, state: dict) -> None:
    temporary = path.with_suffix(".partial")
    torch.save(state, temporary)
    temporary.replace(path)


class Uploader(threading.Thread):
    """Serialises uploads so training never waits on the network (exp097 pattern)."""

    def __init__(self, dataset: str, title: str, staging: Path, log: Path):
        super().__init__(daemon=True, name="kaggle-uploader")
        self.dataset, self.title, self.staging, self.log = dataset, title, staging, log
        self.jobs: queue.Queue = queue.Queue()

    def submit(self, files: dict, message: str) -> None:
        self.jobs.put((dict(files), message))

    def run(self) -> None:
        from experiments.exp_097_runpod_yolo.src.kaggle_sync import publish
        while True:
            files, message = self.jobs.get()
            try:
                upload = publish(files, self.dataset, self.title, message, self.staging)
                event("UPLOAD", self.log, message=message, bytes=upload["bytes"],
                      files=len(upload["files"]), seconds=upload["seconds"])
            except Exception as error:   # a failed mid-run upload must not kill training
                event("UPLOAD_ERROR", self.log, message=message, error=str(error)[:400])
            finally:
                self.jobs.task_done()

    def drain(self) -> None:
        self.jobs.join()


def parse() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--images", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=50, help="upper bound")
    parser.add_argument("--min-epochs", type=int, default=25, help="only with --epoch-policy fit")
    parser.add_argument("--epoch-policy", choices=("fixed", "fit"), default="fixed",
                        help="fixed: train every epoch or stop at the speed check; fit: shrink to the budget")
    parser.add_argument("--crop", type=int, default=1024)
    parser.add_argument("--crops-per-record", type=int, default=2)
    parser.add_argument("--whole-share", type=float, default=0.25)
    parser.add_argument("--no-rotate90", action="store_true")
    parser.add_argument("--batch", type=int, default=8, help="crops per optimiser step")
    parser.add_argument("--micro-batch", type=int, default=0,
                        help="crops per forward pass; must divide --batch; 0 means --batch")
    parser.add_argument("--checkpoint", choices=("none", "whole", "all"), default="none",
                        help="backbone activation checkpointing: never, whole frames only, always")
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.05)
    parser.add_argument("--backbone-multiplier", type=float, default=0.1)
    parser.add_argument("--clip", type=float, default=0.01)
    parser.add_argument("--warmup", type=int, default=500)
    parser.add_argument("--probe-steps", type=int, default=40)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--msda", choices=("pytorch", "hub"), default="pytorch",
                        help="deformable attention: HF's PyTorch loop or the hub CUDA kernel")
    parser.add_argument("--msda-revision", default=None, help="pin the hub kernel revision")
    parser.add_argument("--no-batched-loss", action="store_true",
                        help="use HF's loss with one CPU copy per image per layer")
    parser.add_argument("--no-preload", action="store_true",
                        help="decode photographs on demand instead of once into RAM")
    parser.add_argument("--model", choices=("pretrained", "tiny"), default="pretrained")
    parser.add_argument("--max-steps", type=int, default=0, help="smoke tests only")
    parser.add_argument("--eval-every", type=int, default=5, help="0 turns in-run scoring off")
    parser.add_argument("--fold", choices=("train", "all"), default="train",
                        help="train: the canonical 974 records; all: every record, no held-out set")
    parser.add_argument("--save-epochs", type=int, nargs="*", default=[],
                        help="epochs whose weights are kept when in-run scoring is off")
    parser.add_argument("--sanity-photographs", type=int, default=0,
                        help="score this many canonical validation photographs at the end; with "
                             "--fold all they are in the training data, so it only detects a broken run")
    parser.add_argument("--eval-limit", type=int, default=0, help="smoke tests only")
    parser.add_argument("--sweep-processes", type=int, default=4)
    parser.add_argument("--memory-probe", action="store_true",
                        help="run a few steps on the densest samples of both extents and exit")
    parser.add_argument("--price-per-hour", type=float, default=0.0)
    parser.add_argument("--budget-usd", type=float, default=0.0)
    parser.add_argument("--reserve-usd", type=float, default=1.0,
                        help="held back for the final evaluation and upload")
    parser.add_argument("--billing-start", type=float, default=0.0,
                        help="unix seconds when billing began, so boot time counts")
    parser.add_argument("--resume", action="store_true", help="continue from output/last.pt")
    parser.add_argument("--no-final-d4", action="store_true",
                        help="skip the D4 evaluation of the selected checkpoint")
    parser.add_argument("--upload-dataset", default=None)
    parser.add_argument("--upload-title", default="sol exp139 mask2former weights")
    return parser.parse_args()


def capture_loss_dict(model) -> dict:
    """Wrap `get_loss` so every forward leaves its per-term losses behind, before summing."""
    captured: dict = {}
    original = model.get_loss

    def wrapper(loss_dict):
        captured["dict"] = loss_dict
        return original(loss_dict)

    model.get_loss = wrapper
    return captured


def step_shares(classes, num_queries: int, eos: float) -> tuple[float, float]:
    """A micro-batch's share of its step: masks, and cross-entropy weight.

    transformers normalises each forward pass by exactly these two quantities - the mask
    and dice losses by the number of target masks, the class loss by the sum of class
    weights (1 per matched query, `no_object_weight` per unmatched one). Scaling each
    micro-batch by its share therefore reproduces the whole step's loss exactly.
    """
    masks = sum(len(item) for item in classes)
    weight = sum(len(item) + eos * (num_queries - len(item)) for item in classes)
    return float(masks), float(weight)


def step_losses(model, pending: list, device, args, captured: dict) -> float:
    """Forward, scale and backward every micro-batch of one optimiser step."""
    counts = [step_shares(classes, model.config.num_queries, model.config.no_object_weight)
              for _, _, classes, _, _ in pending]
    total_masks = max(1e-12, sum(count[0] for count in counts))
    total_weight = max(1e-12, sum(count[1] for count in counts))
    value = 0.0
    for (pixels, masks, classes, photometric, extent), (micro_masks, micro_weight) in zip(pending, counts):
        if args.checkpoint != "none":
            backbone_checkpointing(model, checkpoint_for(args.checkpoint, extent))
        pixels = gpu_pixels(pixels, photometric, device)
        masks = [item.to(device, non_blocking=True).float() for item in masks]
        classes = [item.to(device, non_blocking=True) for item in classes]
        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=args.bf16):
            model(pixel_values=pixels, mask_labels=masks, class_labels=classes)
        mask_share, class_share = micro_masks / total_masks, micro_weight / total_weight
        loss = sum(term * (class_share if "cross_entropy" in key else mask_share)
                   for key, term in captured["dict"].items())
        loss.backward()
        value += float(loss.detach())
    return value


def checkpoint_for(mode: str, extent: int) -> bool:
    from experiments.exp_139_mask2former.src.data import NATIVE
    return mode == "all" or (mode == "whole" and extent >= NATIVE)


def memory_probe(args, model, optimiser, dataset, sampler, device, log) -> None:
    """Worst case for both micro-batch kinds: the most instance-dense records.

    It runs the optimiser step too, so Adam's two moment buffers are in the peak.
    """
    from experiments.exp_139_mask2former.src.data import NATIVE
    dense = sorted(range(len(dataset.records)), key=lambda i: -len(dataset.records[i]["rles"]))
    model.train()
    for extent, size in ((args.crop, sampler.micro), (NATIVE, sampler.whole_micro)):
        backbone_checkpointing(model, checkpoint_for(args.checkpoint, extent))
        torch.cuda.reset_peak_memory_stats() if device.type == "cuda" else None
        for repeat in range(2):
            items = [dataset[(dense[j % len(dense)], extent, 0, j, True, 1)] for j in range(size)]
            pixels, masks, classes, _, photometric = collate(items)
            pixels = gpu_pixels(pixels, photometric, device)
            masks = [m.to(device).float() for m in masks]
            classes = [c.to(device) for c in classes]
            with torch.autocast(device.type, dtype=torch.bfloat16, enabled=args.bf16):
                loss = model(pixel_values=pixels, mask_labels=masks, class_labels=classes).loss
            loss.backward()
            optimiser.step()
            optimiser.zero_grad(set_to_none=True)
        peak = torch.cuda.max_memory_allocated() / 2**30 if device.type == "cuda" else 0.0
        total = torch.cuda.get_device_properties(0).total_memory / 2**30 if device.type == "cuda" else 0.0
        event("PROBE", log, extent=extent, batch=size, checkpoint=checkpoint_for(args.checkpoint, extent),
              peak_gib=round(peak, 2), device_gib=round(total, 2),
              instances=[len(m) for m in masks], loss=round(loss.item(), 4))


def main() -> int:
    args = parse()
    args.output.mkdir(parents=True, exist_ok=True)
    log = args.output / "events.log"
    torch.manual_seed(2026)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    started = time.time()
    billing_start = args.billing_start or started

    def spent() -> float:
        return (time.time() - billing_start) / 3600 * args.price_per_hour

    records = load_records(args.annotations, args.fold)
    loading_started = time.time()
    dataset = FilamentSamples(records, args.images, rotate90=not args.no_rotate90,
                              preload=not args.no_preload)
    event("LOADER", log, preload=not args.no_preload, seconds=round(time.time() - loading_started, 1),
          instances=sum(len(pieces) for pieces in dataset._pieces))
    sampler = ExtentBatchSampler(len(records), args.crop, args.batch, args.whole_share,
                                 micro=args.micro_batch or args.batch,
                                 crops_per_record=args.crops_per_record)
    model = build_model(args.model).to(device)

    # Speed-ups with identical mathematics; each is proven equal before it is used.
    if args.msda != "pytorch":
        if device.type != "cuda":
            raise SystemExit("the deformable attention kernel needs a CUDA device")
        try:
            kernel = install_deformable_kernel(args.msda, args.msda_revision)
            rows = verify_deformable_kernel(device)
            event("KERNEL", log, **kernel, verification=rows)
            kernel_ok = all(row["passed"] for row in rows)
        except Exception as error:
            event("KERNEL", log, source=args.msda, error=f"{type(error).__name__}: {error}"[:1000])
            kernel_ok = False
        if not kernel_ok:
            # HF's own PyTorch loop is the reference the kernel was tested against, so this
            # keeps the mathematics and costs only speed; the speed check then re-prices the run.
            install_deformable_kernel("pytorch")
            event("KERNEL_FALLBACK", log, source="pytorch",
                  message="kernel failed its check or did not load; training uses HF's PyTorch loop")
    if not args.no_batched_loss:
        install_batched_loss(model)
    capture = capture_loss_dict(model)

    groups, counts = parameter_groups(model, args.lr, args.weight_decay, args.backbone_multiplier)
    # The fused AdamW kernel applies the same update as the default implementation.
    optimiser = torch.optim.AdamW(groups, lr=args.lr, betas=(0.9, 0.999), fused=device.type == "cuda")
    base_rates = [group["lr"] for group in optimiser.param_groups]
    backbone = model.config.backbone_config
    event("MODEL", log, model=args.model, device=str(device), parameters=counts,
          queries=model.config.num_queries, drop_path=getattr(backbone, "drop_path_rate", None),
          aux_loss=model.config.use_auxiliary_loss, weights=dict(
              class_=model.config.class_weight, mask=model.config.mask_weight,
              dice=model.config.dice_weight, no_object=model.config.no_object_weight),
          bf16=args.bf16, gpu=torch.cuda.get_device_name(0) if device.type == "cuda" else None)

    if args.memory_probe:
        memory_probe(args, model, optimiser, dataset, sampler, device, log)
        return 0

    loader = DataLoader(dataset, batch_sampler=sampler, num_workers=args.workers,
                        collate_fn=collate, persistent_workers=args.workers > 0,
                        pin_memory=device.type == "cuda", prefetch_factor=4 if args.workers else None)
    steps_per_epoch = sampler.steps
    epochs = args.epochs
    total_steps = steps_per_epoch * epochs
    step, start_epoch, sized = 0, 0, False
    last = args.output / "last.pt"
    if args.resume and last.is_file():
        state = torch.load(last, map_location="cpu", weights_only=False)
        model.load_state_dict(state["model"])
        optimiser.load_state_dict(state["optimiser"])
        step, start_epoch = int(state["step"]), int(state["epoch"]) + 1
        epochs, total_steps, sized = int(state["epochs"]), int(state["total_steps"]), True
        event("RESUME", log, step=step, epoch=start_epoch, epochs=epochs)
    event("DATA", log, records=len(records), micro_batches_per_epoch=len(sampler),
          steps_per_epoch=steps_per_epoch, crop=args.crop, whole_share=args.whole_share,
          batch=sampler.batch, micro=sampler.micro, whole_batch=sampler.whole_batch,
          whole_micro=sampler.whole_micro, checkpoint=args.checkpoint,
          rotate90=not args.no_rotate90)

    val_records, _ = records_for(args.annotations)
    uploader = None
    if args.upload_dataset:
        uploader = Uploader(args.upload_dataset, args.upload_title, args.output / "staging", log)
        uploader.start()
    sweeps: dict[int, subprocess.Popen] = {}
    inference_seconds: list[float] = []
    stop_reason = "completed"

    def save_checkpoint(epoch_number: int) -> Path:
        checkpoint = args.output / f"model_e{epoch_number:03d}.pt"
        atomic_save(checkpoint, dict(model=model.state_dict(), config=model.config.to_dict(),
                                     epoch=epoch_number, step=step, crop=args.crop))
        return checkpoint

    def evaluate_now(epoch_number: int) -> None:
        save_checkpoint(epoch_number)
        model.eval()
        # Floor 0.05 equals the sweep's lowest threshold. A candidate's score is its
        # class probability times a mask score <= 1, so nothing below the floor could
        # ever be emitted by the single-view sweep; skipping it only skips encoding.
        cache, seconds = cache_validation(model, val_records, args.images, device, args.bf16,
                                          limit=args.eval_limit, floor=0.05)
        model.train()
        inference_seconds.append(seconds)
        cache_path = args.output / f"e{epoch_number:03d}_cache.json"
        cache_path.write_text(json.dumps(cache))
        launch_sweep(epoch_number)
        event("EVAL_STARTED", log, epoch=epoch_number, inference_seconds=round(seconds, 1),
              candidates=sum(len(entries) for entries in cache.values()))

    def completed_reports() -> dict[int, dict]:
        """Every finished sweep on disk, including those from before a restart."""
        reports = {}
        for path in sorted(args.output.glob("e[0-9][0-9][0-9]_sweep.json")):
            epoch_number = int(path.name[1:4])
            process = sweeps.get(epoch_number)
            if process is None or process.poll() is not None:
                reports[epoch_number] = json.loads(path.read_text())
        return reports

    def launch_sweep(epoch_number: int) -> None:
        command = [sys.executable, "-m", "experiments.exp_139_mask2former.src.evaluate",
                   "--annotations", str(args.annotations), "--output", str(args.output),
                   "--cache", str(args.output / f"e{epoch_number:03d}_cache.json"),
                   "--tag", f"e{epoch_number:03d}", "--processes", str(args.sweep_processes),
                   "--log", str(log)]
        if args.eval_limit:
            command += ["--limit", str(args.eval_limit)]
        sweeps[epoch_number] = subprocess.Popen(command)

    # A restart can land between an evaluation's inference and its sweep.
    for cache_path in sorted(args.output.glob("e[0-9][0-9][0-9]_cache.json")):
        epoch_number = int(cache_path.name[1:4])
        if not (args.output / f"e{epoch_number:03d}_sweep.json").is_file():
            launch_sweep(epoch_number)
            event("SWEEP_RELAUNCHED", log, epoch=epoch_number)

    def artifact_files(final: bool) -> dict:
        files = {"events.log": log}
        if (args.output / "sanity_sweep.json").is_file():
            files["sanity_sweep.json"] = args.output / "sanity_sweep.json"
        reports = completed_reports()
        for epoch_number in reports:
            files[f"e{epoch_number:03d}_sweep.json"] = args.output / f"e{epoch_number:03d}_sweep.json"
        if reports:
            best = max(reports, key=lambda e: reports[e]["best"]["pq"])
            files[f"model_e{best:03d}.pt"] = args.output / f"model_e{best:03d}.pt"
        checkpoints = sorted(args.output.glob("model_e[0-9][0-9][0-9].pt"))
        if final:
            # Without sweep reports (a full-data run has no held-out set) every kept
            # checkpoint leaves the pod, because none of them can be ranked here.
            files.update({path.name: path for path in (checkpoints if not reports else checkpoints[-1:])})
        return files

    window_started, window_steps, probe_started = time.time(), 0, None
    model.train()
    epoch = start_epoch
    for epoch in range(start_epoch, epochs):
        if epoch >= epochs:
            break
        sampler.set_epoch(epoch)
        running, epoch_steps = 0.0, 0
        optimiser.zero_grad(set_to_none=True)
        pending: list = []
        for pixels, masks, classes, (step_end, parts, extent), photometric in loader:
            pending.append((pixels, masks, classes, photometric, extent))
            if not step_end:
                continue
            # A step's micro-batches are collected first so each one can be weighted by its
            # exact share of the step (`step_losses`), which makes any micro-batch size give
            # the same loss and gradients as one big batch.
            value = step_losses(model, pending, device, args, capture)
            if not np.isfinite(value):
                raise SystemExit(f"non-finite loss at step {step}: {value}")
            running += value
            pending = []
            epoch_steps += 1
            if args.clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
            factor = schedule(step, args.warmup, total_steps)
            for group, rate in zip(optimiser.param_groups, base_rates):
                group["lr"] = rate * factor
            optimiser.step()
            optimiser.zero_grad(set_to_none=True)
            step += 1
            window_steps += 1
            if step == 5 and probe_started is None:
                probe_started = time.time()

            if not sized and probe_started is not None and step >= 5 + args.probe_steps:
                seconds_per_step = (time.time() - probe_started) / args.probe_steps
                model.eval()
                probe_names = sorted(val_records)[:2]
                inference_started = time.time()
                for name in probe_names:
                    predict(model, read_gray(args.images / name), device, args.bf16)
                photographs = args.eval_limit or len(val_records)
                eval_seconds = (time.time() - inference_started) / len(probe_names) * photographs * 1.3
                model.train()
                affordable, projected = epochs, None
                if args.budget_usd and args.price_per_hour:
                    available = args.budget_usd - args.reserve_usd - spent()
                    per_epoch = steps_per_epoch * seconds_per_step / 3600 * args.price_per_hour
                    per_eval = eval_seconds / 3600 * args.price_per_hour
                    # --eval-every 0 (the full-data run) means no in-run evaluations at all; the
                    # division below killed that run at step 25 (exp142, 2026-09-17).
                    def evaluations_for(total: int) -> int:
                        return math.ceil(total / args.eval_every) + 1 if args.eval_every else 0
                    projected = spent() + (epochs - epoch) * per_epoch + \
                        evaluations_for(epochs) * per_eval + args.reserve_usd
                    affordable = 0
                    for candidate in range(epochs, 0, -1):
                        evaluations = evaluations_for(candidate)
                        if (candidate - epoch) * per_epoch + evaluations * per_eval <= available:
                            affordable = candidate
                            break
                event("SCHEDULE", log, policy=args.epoch_policy, requested_epochs=args.epochs,
                      affordable_epochs=affordable, projected_total_usd=round(projected, 2) if projected else None,
                      seconds_per_step=round(seconds_per_step, 3), eval_seconds=round(eval_seconds, 1),
                      spent_usd=round(spent(), 2), budget_usd=args.budget_usd,
                      gpu_gib=round(torch.cuda.max_memory_allocated() / 2**30, 2) if device.type == "cuda" else 0)
                if args.epoch_policy == "fixed":
                    # Every epoch or none: a run the budget cannot finish stops here, a few
                    # minutes in, instead of being cut short hours later.
                    if affordable < epochs:
                        event("ERROR", log, stage="schedule",
                              message=f"all {epochs} epochs are projected to cost {projected:.2f} USD, "
                                      f"over the {args.budget_usd:.2f} USD budget; stopped before spending more")
                        return 3
                else:
                    if affordable < args.min_epochs:
                        event("ERROR", log, stage="schedule",
                              message=f"budget affords {affordable} epochs, below --min-epochs {args.min_epochs}")
                        return 3
                    epochs, total_steps = affordable, steps_per_epoch * affordable
                sized = True

            if step % 25 == 0 or step == 1:
                elapsed = time.time() - window_started
                event("STEP", log, step=step, epoch=epoch, loss=round(running / max(1, epoch_steps), 4),
                      lr=round(optimiser.param_groups[-1]["lr"], 8),
                      seconds_per_step=round(elapsed / max(1, window_steps), 3),
                      remaining_hours=round(elapsed / max(1, window_steps) * (total_steps - step) / 3600, 2),
                      spent_usd=round(spent(), 2),
                      gpu_gib=round(torch.cuda.max_memory_allocated() / 2**30, 2) if device.type == "cuda" else 0)
                window_started, window_steps = time.time(), 0
            if args.max_steps and step >= args.max_steps:
                stop_reason = "max_steps"
                break
            if args.budget_usd and args.price_per_hour and spent() >= args.budget_usd - args.reserve_usd:
                stop_reason = "budget"
                break

        epoch_number = epoch + 1
        atomic_save(last, dict(model=model.state_dict(), optimiser=optimiser.state_dict(), step=step,
                               epoch=epoch, epochs=epochs, total_steps=total_steps,
                               config=model.config.to_dict()))
        event("EPOCH", log, epoch=epoch_number, epochs=epochs, step=step,
              loss=round(running / max(1, epoch_steps), 4),
              hours=round((time.time() - started) / 3600, 3), spent_usd=round(spent(), 2))
        final = stop_reason != "completed" or epoch_number >= epochs
        evaluating = args.eval_every > 0 and (epoch_number % args.eval_every == 0 or final)
        if evaluating:
            evaluate_now(epoch_number)
        elif epoch_number in args.save_epochs or final:
            # Training on every record leaves no held-out set to score, so checkpoints are
            # kept on a fixed list of epochs instead of on validation PQ.
            save_checkpoint(epoch_number)
            event("CHECKPOINT", log, epoch=epoch_number, step=step)
        if uploader is not None and not final and (evaluating or epoch_number in args.save_epochs):
            uploader.submit(artifact_files(final=False), f"epoch {epoch_number}")
        if final:
            break

    if args.sanity_photographs:
        # A full-data run has trained on these photographs, so this number cannot measure
        # quality. It only shows that the model still finds filaments at all.
        model.eval()
        cache, seconds = cache_validation(model, val_records, args.images, device, args.bf16,
                                          limit=args.sanity_photographs, floor=0.05)
        cache_path = args.output / "sanity_cache.json"
        cache_path.write_text(json.dumps(cache))
        subprocess.run([sys.executable, "-m", "experiments.exp_139_mask2former.src.evaluate",
                        "--annotations", str(args.annotations), "--output", str(args.output),
                        "--cache", str(cache_path), "--tag", "sanity", "--limit",
                        str(args.sanity_photographs), "--processes", str(args.sweep_processes),
                        "--log", str(log)], check=False)
        report_path = args.output / "sanity_sweep.json"
        best = json.loads(report_path.read_text())["best"] if report_path.is_file() else None
        event("SANITY", log, in_sample=args.fold == "all", photographs=args.sanity_photographs,
              best=best, inference_seconds=round(seconds, 1))

    for process in sweeps.values():
        process.wait()
    reports = completed_reports()
    exit_code = 0
    if reports:
        best = max(reports, key=lambda e: reports[e]["best"]["pq"])
        event("BEST", log, epoch=best, row=reports[best]["best"],
              curve={e: round(r["best"]["pq"], 5) for e, r in sorted(reports.items())})
    elif args.eval_every > 0:
        event("ERROR", log, stage="evaluation", message="no sweep report completed")
        exit_code = 4
    else:
        event("NO_SCORING", log, message="in-run scoring is off; every kept checkpoint is uploaded",
              epochs_kept=sorted(int(path.name[7:10]) for path in args.output.glob("model_e[0-9][0-9][0-9].pt")))
    def final_upload(message: str) -> int:
        from experiments.exp_097_runpod_yolo.src.kaggle_sync import publish, verify
        files = artifact_files(final=True)
        for path in sorted(args.output.glob("best_e[0-9][0-9][0-9]_d4_sweep.json")):
            files[path.name] = path
        try:
            upload = publish(files, args.upload_dataset, args.upload_title, message,
                             args.output / "staging")
            upload["verification"] = verify(args.upload_dataset, upload["files"], timeout=600)
            event("UPLOAD", log, message=message, bytes=upload["bytes"], files=len(upload["files"]),
                  verification=upload["verification"])
            return 0 if upload["verification"]["verified"] else 5
        except Exception as error:
            event("UPLOAD_ERROR", log, message=message, error=str(error)[:500])
            return 6

    # Weights leave the pod before anything optional runs: the budget watchdog stops
    # the pod at the cap, and a stopped pod must not take the only copy with it.
    if uploader is not None:
        uploader.drain()
        exit_code = exit_code or final_upload(f"final ({stop_reason})")

    if reports and not args.no_final_d4:
        # D4 of the selected checkpoint: eight views of every photograph, so about
        # eight single-view evaluations. Skipped, loudly, if the budget cannot hold it.
        estimate = 8 * 1.3 * inference_seconds[-1] / 3600 * args.price_per_hour if inference_seconds else 0.0
        if args.budget_usd and args.price_per_hour and spent() + estimate + 0.25 > args.budget_usd:
            event("D4_SKIPPED", log, spent_usd=round(spent(), 2), estimate_usd=round(estimate, 2),
                  budget_usd=args.budget_usd)
        else:
            state = torch.load(args.output / f"model_e{best:03d}.pt", map_location="cpu",
                               weights_only=False)
            model.load_state_dict(state["model"])
            model.eval()
            cache, seconds = cache_validation(model, val_records, args.images, device, args.bf16,
                                              d4=True, limit=args.eval_limit)
            tag = f"best_e{best:03d}_d4"
            cache_path = args.output / f"{tag}_cache.json"
            cache_path.write_text(json.dumps(cache))
            command = [sys.executable, "-m", "experiments.exp_139_mask2former.src.evaluate",
                       "--annotations", str(args.annotations), "--output", str(args.output),
                       "--cache", str(cache_path), "--tag", tag, "--d4",
                       "--processes", str(args.sweep_processes), "--log", str(log)]
            if args.eval_limit:
                command += ["--limit", str(args.eval_limit)]
            sweep_rc = subprocess.run(command).returncode
            event("D4_DONE", log, epoch=best, inference_seconds=round(seconds, 1), sweep_exit=sweep_rc,
                  spent_usd=round(spent(), 2))
            if uploader is not None and sweep_rc == 0:
                exit_code = exit_code or final_upload("final with D4 report")
    event("SUMMARY", log, stop_reason=stop_reason, steps=step, epochs=epochs,
          hours=round((time.time() - started) / 3600, 3), spent_usd=round(spent(), 2),
          exit_code=exit_code)
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
