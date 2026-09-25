"""exp139 equality tests for the speed-ups. Each check compares new code with the code it replaces.

    python -m experiments.exp_139_mask2former.tests.test_equivalence --suite cpu ...
    python -m experiments.exp_139_mask2former.tests.test_equivalence --suite gpu ...

cpu: the batched loss against HF's loss (tiny random model: all losses and gradients
     identical to the bit), and the fast loader against the first loader (masks
     identical, pixels equal to float32 rounding).
gpu: the loader again on real images; the hub deformable-attention kernel against a
     float64 run of HF's PyTorch loop; the batched loss against HF's loss with HF run twice
     as the control for CUDA's non-deterministic backward; the full model with the kernel
     against the full model without it, again with a control; peak memory; step timing.

Results go to one JSON file. The process exits non-zero if an equality check fails.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from experiments.exp_139_mask2former.src.data import NATIVE, FilamentSamples, collate, gpu_pixels, load_records

_DATASETS: dict = {}


def loader_check(annotations: Path, images: Path, crops: int, wholes: int, preload: bool) -> dict:
    from experiments.exp_139_mask2former.tests.reference_data import ReferenceSamples
    records = load_records(annotations, "train")
    started = time.time()
    fast = FilamentSamples(records, images, preload=preload)
    build_seconds = time.time() - started
    _DATASETS[(str(annotations), str(images), preload)] = fast
    rng = np.random.default_rng(7)
    keys = [(int(rng.integers(len(records))), 1024, int(rng.integers(50)), int(rng.integers(10**6)), True, 1)
            for _ in range(crops)]
    keys += [(int(rng.integers(len(records))), NATIVE, int(rng.integers(50)), int(rng.integers(10**6)), True, 1)
             for _ in range(wholes)]
    report = dict(keys=len(keys), build_seconds=round(build_seconds, 1))
    for augment in (True, False):
        reference = ReferenceSamples(records, images, augment=augment)
        fast.augment = augment
        mask_mismatch, meta_mismatch, worst = 0, 0, 0.0
        timing = dict(reference=0.0, fast=0.0)
        for key in keys:
            started = time.time()
            ref_pixels, ref_masks, ref_classes, ref_meta = reference[key]
            timing["reference"] += time.time() - started
            started = time.time()
            patch, masks, classes, meta, photometric = fast[key]
            timing["fast"] += time.time() - started
            if not (torch.equal(ref_masks, masks) and torch.equal(ref_classes, classes)):
                mask_mismatch += 1
            if tuple(ref_meta) != tuple(meta):
                meta_mismatch += 1
            pixels = gpu_pixels(patch[None], [photometric], torch.device("cpu"))[0]
            worst = max(worst, float((pixels - ref_pixels).abs().max()))
        report["augment" if augment else "plain"] = dict(
            mask_mismatches=mask_mismatch, metadata_mismatches=meta_mismatch,
            max_abs_pixel_difference=worst,
            milliseconds_per_sample=dict(reference=round(1000 * timing["reference"] / len(keys), 1),
                                         fast=round(1000 * timing["fast"] / len(keys), 1)))
    fast.augment = True
    report["passed"] = all(report[name]["mask_mismatches"] == 0 and report[name]["metadata_mismatches"] == 0
                           and report[name]["max_abs_pixel_difference"] < 1e-4 for name in ("augment", "plain"))
    return report


def sample_batch(annotations: Path, images: Path, size: int, extent: int, device: torch.device, preload: bool):
    key = (str(annotations), str(images), preload)
    if key not in _DATASETS:
        _DATASETS[key] = FilamentSamples(load_records(annotations, "train"), images, preload=preload)
    dataset = _DATASETS[key]
    records = dataset.records
    dense = sorted(range(len(records)), key=lambda i: -len(records[i]["rles"]))
    items = [dataset[(dense[j], extent, 0, j, True, 1)] for j in range(size)]
    pixels, masks, classes, _, photometric = collate(items)
    return (gpu_pixels(pixels, photometric, device), [m.to(device).float() for m in masks],
            [c.to(device) for c in classes])


def _gradients(model, batch, autocast_dtype, seed: int = 99):
    pixels, masks, classes = batch
    model.zero_grad(set_to_none=True)
    torch.manual_seed(seed)
    with torch.autocast(pixels.device.type, dtype=autocast_dtype or torch.float32,
                        enabled=autocast_dtype is not None):
        loss = model(pixel_values=pixels, mask_labels=masks, class_labels=classes).loss
    loss.backward()
    return float(loss), {n: p.grad.detach().float().clone() for n, p in model.named_parameters() if p.grad is not None}


def _relative(first: dict, second: dict) -> tuple[float, float]:
    values = []
    for name, reference in first.items():
        scale = float(reference.abs().max())
        if scale > 0:
            values.append(float((second[name] - reference).abs().max()) / scale)
    return max(values), float(np.median(values))


def model_kernel_check(model, batch, revision: str | None, autocast_dtype) -> dict:
    """Full model: HF's deformable attention twice (control), then the hub kernel."""
    from experiments.exp_139_mask2former.src import fast_ops
    fast_ops.install_deformable_kernel("pytorch")
    loss_a, grads_a = _gradients(model, batch, autocast_dtype)
    loss_b, grads_b = _gradients(model, batch, autocast_dtype)
    fast_ops.install_deformable_kernel("hub", revision)
    loss_c, grads_c = _gradients(model, batch, autocast_dtype)
    control_worst, control_median = _relative(grads_a, grads_b)
    kernel_worst, kernel_median = _relative(grads_a, grads_c)
    return dict(loss_hf=loss_a, loss_hf_again=loss_b, loss_kernel=loss_c,
                loss_relative_difference=abs(loss_c - loss_a) / max(abs(loss_a), 1e-12),
                control_gradient_worst=control_worst, control_gradient_median=control_median,
                kernel_gradient_worst=kernel_worst, kernel_gradient_median=kernel_median)


def memory_check(model, annotations, images, device, autocast_dtype, preload) -> list[dict]:
    rows = []
    for extent, size in ((1024, 1), (1024, 2), (1536, 1)):
        model.zero_grad(set_to_none=True)
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        try:
            batch = sample_batch(annotations, images, size, extent, device, preload)
            with torch.autocast(device.type, dtype=autocast_dtype, enabled=autocast_dtype is not None):
                loss = model(pixel_values=batch[0], mask_labels=batch[1], class_labels=batch[2]).loss
            loss.backward()
            rows.append(dict(extent=extent, batch=size, peak_gib=round(torch.cuda.max_memory_allocated() / 2**30, 3),
                             instances=[len(m) for m in batch[1]]))
            del batch, loss
        except torch.OutOfMemoryError:
            rows.append(dict(extent=extent, batch=size, peak_gib=None, error="out of memory"))
        model.zero_grad(set_to_none=True)
    return rows


def profile_steps(model, annotations, images, device, autocast_dtype, preload, steps: int = 8) -> dict:
    """Seconds per training step (batch 2 crops) and the top CUDA operations."""
    from torch.profiler import ProfilerActivity, profile
    batch = sample_batch(annotations, images, 2, 1024, device, preload)
    optimiser = torch.optim.AdamW(model.parameters(), lr=1e-7, fused=True)

    def step():
        with torch.autocast(device.type, dtype=autocast_dtype, enabled=autocast_dtype is not None):
            loss = model(pixel_values=batch[0], mask_labels=batch[1], class_labels=batch[2]).loss
        loss.backward()
        optimiser.step()
        optimiser.zero_grad(set_to_none=True)
        return float(loss)

    for _ in range(3):
        step()
    torch.cuda.synchronize()
    started = time.time()
    for _ in range(steps):
        step()
    torch.cuda.synchronize()
    seconds = (time.time() - started) / steps
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as trace:
        step()
        torch.cuda.synchronize()
    table = trace.key_averages().table(sort_by="cuda_time_total", row_limit=12)
    del optimiser
    return dict(seconds_per_step=round(seconds, 3), top_cuda_ops=table)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--suite", choices=("cpu", "gpu"), required=True)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--images", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--crops", type=int, default=200)
    parser.add_argument("--wholes", type=int, default=30)
    parser.add_argument("--no-preload", action="store_true")
    parser.add_argument("--skip-loader", action="store_true")
    parser.add_argument("--revision", default=None)
    args = parser.parse_args()
    report: dict = dict(suite=args.suite, torch=torch.__version__)
    preload = not args.no_preload
    failed = []

    def save():
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2, default=str))

    def record(key: str, value) -> None:
        report[key] = value
        print(key.upper(), json.dumps(value, default=str)[:3000], flush=True)
        if isinstance(value, dict) and value.get("passed") is False:
            failed.append(key)
        save()

    if not args.skip_loader:
        record("loader", loader_check(args.annotations, args.images, args.crops, args.wholes, preload))

    from experiments.exp_139_mask2former.src import fast_ops
    from experiments.exp_139_mask2former.src.model import build_model
    if args.suite == "cpu":
        device = torch.device("cpu")
        model = build_model("tiny").to(device).train()
        batch = sample_batch(args.annotations, args.images, 2, 1024, device, preload)
        for label, dtype in (("fp32", None), ("bf16", torch.bfloat16)):
            record(f"batched_loss_{label}", fast_ops.verify_batched_loss(model, *batch, dtype))
    else:
        device = torch.device("cuda")
        report["gpu"] = torch.cuda.get_device_name(0)
        # T4 has no native bf16; fp16 autocast has the same memory footprint.
        mixed = torch.bfloat16 if torch.cuda.get_device_capability(0)[0] >= 8 else torch.float16
        report["mixed_precision"] = str(mixed)
        try:
            started = time.time()
            description = fast_ops.install_deformable_kernel("hub", args.revision)
            load_seconds = round(time.time() - started, 1)
            rows = fast_ops.verify_deformable_kernel(device)
            record("kernel_hub", dict(description=description, load_seconds=load_seconds, verification=rows,
                                      passed=all(row["passed"] for row in rows)))
        except Exception as error:
            record("kernel_hub", dict(passed=False, error=f"{type(error).__name__}: {error}"[:3000]))
        kernel_ok = bool(report["kernel_hub"].get("passed"))
        fast_ops.install_deformable_kernel("pytorch")
        model = build_model("pretrained").to(device).train()
        batch = sample_batch(args.annotations, args.images, 1, 1024, device, preload)
        record("batched_loss_fp32", fast_ops.verify_batched_loss(model, *batch, None))
        record("batched_loss_mixed", fast_ops.verify_batched_loss(model, *batch, mixed))
        fast_ops.restore_hf_loss(model)
        if kernel_ok:
            record("model_kernel_fp32", model_kernel_check(model, batch, args.revision, None))
            record("model_kernel_mixed", model_kernel_check(model, batch, args.revision, mixed))
        del batch
        for source in ["pytorch"] + (["hub"] if kernel_ok else []):
            fast_ops.install_deformable_kernel(source, args.revision)
            record(f"memory_{source}", memory_check(model, args.annotations, args.images, device, mixed, preload))
        variants = [("pytorch", False)] + ([("hub", False), ("hub", True)] if kernel_ok else [])
        for source, batched in variants:
            fast_ops.install_deformable_kernel(source, args.revision)
            fast_ops.install_batched_loss(model) if batched else fast_ops.restore_hf_loss(model)
            record(f"profile_{source}_{'batched' if batched else 'hf'}_loss",
                   profile_steps(model, args.annotations, args.images, device, mixed, preload))
    report["failed"] = failed
    save()
    print("RESULT", json.dumps(dict(failed=failed)), flush=True)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
