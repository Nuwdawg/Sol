"""Explicit opt-in to the chunked criterion; historical experiments stay intact."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess

# Before torch/CUDA imports. There is no automatic configuration fallback.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")


def install_criterion(trainer, chunk_size, expected, fp32_validation=False, criterion_name='chunked'):
    import torch
    import ultralytics
    from shared.chunked_mask_loss import ChunkedSegmentationLoss
    criterion_type = ChunkedSegmentationLoss
    if criterion_name == 'roi':
        from shared.roi_mask_loss import ROISegmentationLoss
        criterion_type = ROISegmentationLoss
    elif criterion_name != 'chunked':
        raise ValueError(criterion_name)

    if ultralytics.__version__ != "8.4.0":
        raise RuntimeError("exp_032 requires ultralytics==8.4.0; rerun parity tests before upgrading")
    resolved = vars(trainer.args)
    for key, value in expected.items():
        if resolved[key] != value:
            raise RuntimeError(f"configuration changed: {key}={resolved[key]!r}, expected {value!r}")
    trainer.model.criterion = criterion_type(trainer.model, chunk_size)
    # Validation computes loss on EMA, not on the live training model. Leaving
    # this at upstream would move the same OOM to the first validation epoch.
    if trainer.ema is not None:
        trainer.ema.ema.criterion = criterion_type(trainer.ema.ema, chunk_size)
    if fp32_validation:
        from copy import copy
        from ultralytics.models.yolo.segment import SegmentationValidator

        class FP32Validation(SegmentationValidator):
            def __call__(self, trainer=None, model=None):
                if trainer is None:
                    self.args.half = False
                    return super().__call__(trainer, model)
                training_amp = trainer.amp
                try:
                    # Upstream forcibly derives validation precision from AMP.
                    trainer.amp = False
                    return super().__call__(trainer, model)
                finally:
                    trainer.amp = training_amp

        trainer.validator = FP32Validation(trainer.test_loader, save_dir=trainer.save_dir,
                                          args=copy(trainer.args), _callbacks=trainer.callbacks)
    metadata = {
        "criterion": type(trainer.model.criterion).__name__, "chunk_size": chunk_size,
        "torch": torch.__version__, "ultralytics": ultralytics.__version__,
        "device": str(next(trainer.model.parameters()).device), "resolved": resolved,
        "source_sha256": hashlib.sha256(Path(__file__).parents[3].joinpath("shared/chunked_mask_loss.py").read_bytes()).hexdigest(),
        "git_commit": os.environ.get("SOL_SOURCE_REVISION") or subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "trainer_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "validation_precision": "fp32" if fp32_validation else "upstream",
        "roi_sha256": hashlib.sha256(Path(__file__).parents[3].joinpath('shared/roi_mask_loss.py').read_bytes()).hexdigest() if criterion_name=='roi' else None,
    }
    Path(trainer.save_dir, "resolved_experiment.json").write_text(
        json.dumps(metadata, indent=2, default=str) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True)
    parser.add_argument("--weights", required=True)
    parser.add_argument("--project", required=True)
    parser.add_argument("--name", default="chunked")
    parser.add_argument("--imgsz", type=int, default=2048)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--chunk-size", type=int, default=8)
    parser.add_argument("--device", default="0")
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--lr0", type=float, default=5e-4)
    parser.add_argument("--fp32-validation", action="store_true")
    parser.add_argument('--criterion', choices=['chunked','roi'], default='chunked')
    parser.add_argument("--time-hours", type=float, default=None,
                        help="Ultralytics measured-time schedule; includes automatic epoch-count adjustment")
    args = parser.parse_args()
    if min(args.imgsz, args.batch, args.epochs, args.chunk_size) < 1:
        parser.error("image size, batch, epochs and chunk size must be positive")
    if args.time_hours is not None and args.time_hours <= 0:
        parser.error('time budget must be positive')
    if args.device != "cpu" and not args.device.isdecimal():
        parser.error("use cpu or one CUDA device index; distributed training is not supported")
    if not Path(args.weights).is_file():
        parser.error("weights must be an explicit local checkpoint, with no download fallback")
    if Path(args.project, args.name).exists():
        parser.error("run directory exists; use a new name to preserve previous evidence")

    from ultralytics import YOLO
    expected = dict(imgsz=args.imgsz, batch=args.batch, epochs=args.epochs,
                    mask_ratio=1, overlap_mask=True, optimizer="AdamW", cos_lr=True, warmup_bias_lr=0.0,
                    mosaic=0.0, mixup=0.0, copy_paste=0.0, freeze=0)
    if args.time_hours is not None:
        expected['time'] = args.time_hours
    model = YOLO(args.weights)
    model.add_callback("on_pretrain_routine_end", lambda trainer: install_criterion(trainer, args.chunk_size, expected, args.fp32_validation, args.criterion))
    model.train(
        **expected, data=args.data, project=args.project, name=args.name,
        device=args.device, workers=args.workers, lr0=args.lr0, lrf=.01,
        warmup_epochs=2, patience=args.epochs, save_period=1,
        hsv_h=0.0, hsv_s=0.0, hsv_v=.15, fliplr=.5, flipud=.5, degrees=15,
        cache=False, seed=2026, plots=False,
    )


if __name__ == "__main__":
    main()
