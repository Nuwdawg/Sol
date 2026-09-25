"""Exercise crowded real training records, optimizer state and EMA validation.

Run each invocation in its own process. This is a resource probe, not a PQ run.
Uses the prepared canonical dataset and an explicit existing checkpoint.
"""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import time

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True)
    parser.add_argument("--weights", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--imgsz", type=int, default=2048)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--chunk-size", type=int, default=8)
    parser.add_argument("--device", default="0")
    parser.add_argument("--steps", type=int, default=3)
    parser.add_argument('--criterion',choices=['chunked','roi'],default='chunked')
    args = parser.parse_args()
    if args.steps < 2 or args.batch < 1:
        parser.error("at least two optimizer steps and a positive batch are required")
    output = Path(args.output)
    if output.exists():
        parser.error("output directory already exists; keep previous probe evidence")
    if not Path(args.weights).is_file():
        parser.error("an existing checkpoint is required")

    import torch
    from ultralytics.models.yolo.segment import SegmentationTrainer
    from ultralytics.utils.torch_utils import autocast
    from experiments.exp_032_chunked.src.train import install_criterion
    from ultralytics import YOLO

    expected = dict(imgsz=args.imgsz, batch=args.batch, epochs=1, mask_ratio=1,
                    overlap_mask=True, optimizer="AdamW", cos_lr=True,
                    mosaic=0., mixup=0., copy_paste=0., freeze=0)
    trainer = SegmentationTrainer(overrides=dict(
        **expected, model=args.weights, data=args.data, device=args.device,
        project=str(output.parent), name=output.name, workers=0, plots=False,
        seed=2026, lr0=5e-4, hsv_h=0., hsv_s=0., hsv_v=.15, fliplr=.5,
        flipud=.5, degrees=15., cache=False,
    ))
    trainer.add_callback("on_pretrain_routine_end", lambda t: install_criterion(t, args.chunk_size, expected, True, args.criterion))
    trainer._setup_train()
    trainer.epoch = 0  # Normally initialized by the training loop, bypassed by this probe.
    dataset = trainer.train_loader.dataset
    crowded = sorted(range(len(dataset)), key=lambda i: len(dataset.labels[i]["cls"]), reverse=True)
    report = dict(status="running", device=str(trainer.device),
                  gpu=torch.cuda.get_device_name(trainer.device) if trainer.device.type == "cuda" else None,
                  settings=expected, criterion=args.criterion, steps=[], amp_skips=[])
    report_path = output / "probe.json"

    def save():
        report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    def sync():
        if trainer.device.type == "cuda":
            torch.cuda.synchronize()

    save()
    if trainer.device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    trainer.model.train()
    trainer.optimizer.zero_grad()
    for step in range(args.steps):
        indices = [crowded[(step * args.batch + j) % len(crowded)] for j in range(args.batch)]
        start = time.perf_counter()
        batch = trainer.preprocess_batch(dataset.collate_fn([dataset[i] for i in indices]))
        for attempt in range(17):
            scale_before = trainer.scaler.get_scale()
            with autocast(trainer.amp):
                loss, trainer.loss_items = trainer.model(batch)
                loss = loss.sum()
            if not torch.isfinite(loss):
                report.update(status="failed", reason="nonfinite forward loss", loss=float(loss.detach()))
                save()
                raise RuntimeError("non-finite forward loss")
            trainer.scaler.scale(loss).backward()
            bad = [name for name,p in trainer.model.named_parameters()
                   if p.grad is not None and not torch.isfinite(p.grad).all()]
            if bad and not trainer.amp:
                raise RuntimeError('non-finite gradients with AMP disabled')
            trainer.optimizer_step()  # GradScaler skips overflowing updates and reduces scale.
            if not bad:
                break
            report['amp_skips'].append(dict(step=step, attempt=attempt,
                loss=float(loss.detach()), scale_before=scale_before,
                scale_after=trainer.scaler.get_scale(), bad_gradient_layers=bad))
            save()
            print('AMP_SKIP',json.dumps(report['amp_skips'][-1]),flush=True)
            if not trainer.amp or trainer.scaler.get_scale() >= scale_before:
                raise RuntimeError('non-finite gradients without loss-scale recovery')
        else:
            raise RuntimeError('no finite gradients after 17 loss-scale attempts')
        sync()
        report["steps"].append(dict(
            seconds=time.perf_counter()-start, loss=float(loss.detach()),
            images=batch["im_file"], instances=len(batch["cls"]),
            original_instances=[len(dataset.labels[i]["cls"]) for i in indices],
            mask_shape=list(batch["masks"].shape),
            loss_scale=trainer.scaler.get_scale(),
        ))
        save()
        print('PROBE_STEP',json.dumps(report['steps'][-1]),flush=True)
    if not trainer.optimizer.state:
        raise RuntimeError("optimizer state was not allocated")
    # Rebuild the most crowded batch through EMA, including FP16 on GPU,
    # exactly as the training validator does. This is a memory check, not scoring.
    ema = trainer.ema.ema
    ema = ema.half() if trainer.amp else ema.float()
    ema.eval()
    batch = trainer.preprocess_batch(dataset.collate_fn([dataset[i] for i in crowded[:args.batch]]))
    batch["img"] = batch["img"].half() if trainer.amp else batch["img"].float()
    with torch.inference_mode():
        val_loss, _ = ema.loss(batch)
    sync()
    report['ema_fp16_loss'] = val_loss.tolist()
    report['ema_nonfinite_state'] = [name for name,tensor in ema.state_dict().items()
                                    if not torch.isfinite(tensor).all()]
    ema.float()
    batch['img'] = batch['img'].float()
    with torch.inference_mode():
        fp32_loss, _ = ema.loss(batch)
    report['ema_fp32_loss'] = fp32_loss.tolist()
    report['cuda_peak_bytes'] = torch.cuda.max_memory_allocated() if trainer.device.type == 'cuda' else None
    save()
    print('EMA_DIAGNOSTIC',json.dumps({k:v for k,v in report.items() if k.startswith('ema_')}),flush=True)
    if not torch.isfinite(fp32_loss).all():
        raise RuntimeError("non-finite FP32 EMA validation loss")
    from torch.utils.data import DataLoader, Subset
    val_dataset = trainer.test_loader.dataset
    val_indices = sorted(range(len(val_dataset)),
                         key=lambda i:len(val_dataset.labels[i]['cls']), reverse=True)[:4]
    trainer.validator.dataloader = DataLoader(Subset(val_dataset,val_indices),
        batch_size=2, num_workers=0, collate_fn=val_dataset.collate_fn)
    val_metrics = trainer.validator(trainer)
    if not all(torch.isfinite(torch.tensor(v)) for v in val_metrics.values()):
        raise RuntimeError('nonfinite actual FP32 validator metrics')
    report['fp32_validator_metrics'] = val_metrics
    report['fp32_validator_records'] = [val_dataset.im_files[i] for i in val_indices]
    save()
    checkpoint = output / "probe_checkpoint.pt"
    restored = YOLO(args.weights)
    restored.model = ema.float()
    restored.ckpt = {'train_args': dict(vars(trainer.args)), 'ema': None}
    restored.save(checkpoint)
    del restored
    restored = YOLO(checkpoint)
    if restored.task != 'segment':
        raise RuntimeError('checkpoint lost segmentation task metadata')
    heldout = trainer.test_loader.dataset.im_files[0]
    predictions = restored.predict(heldout, imgsz=args.imgsz, device=args.device, verbose=False)
    if len(predictions) != 1 or not torch.isfinite(predictions[0].boxes.data).all():
        raise RuntimeError("checkpoint reload/held-out prediction failed")
    report.update(status="complete", ema_loss=fp32_loss.tolist(),
                  fp16_validation_passed=bool(torch.isfinite(val_loss).all()),
                  reloaded_checkpoint=str(checkpoint), heldout_image=heldout,
                  optimizer_states=len(trainer.optimizer.state),
                  cuda_peak_bytes=torch.cuda.max_memory_allocated() if trainer.device.type == "cuda" else None)
    save()
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
