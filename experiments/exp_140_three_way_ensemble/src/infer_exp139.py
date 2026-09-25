"""exp140 GPU step: exp139's candidates on validation and test, exactly as its submission made them.

Epoch 40 checkpoint (SHA256 checked against the pod's manifest), hub deformable-attention
kernel verified against float64 before use, float32 single-view inference, candidate floor
0.02. Validation candidates must reproduce the submission kernel's PQ 0.47052 before the
ensemble trusts them.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from experiments.exp_033_inference.src.sweep import records_for
from experiments.exp_139_mask2former.src.data import read_gray
from experiments.exp_139_mask2former.src.evaluate import cache_validation, load_checkpoint
from experiments.exp_139_mask2former.src.fast_ops import install_deformable_kernel, verify_deformable_kernel
from experiments.exp_139_mask2former.src.model import predict_photograph
from experiments.exp_139_mask2former.src.submit import sha256


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--train-images", type=Path, required=True)
    parser.add_argument("--test-images", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--epoch", type=int, default=40)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda")
    checkpoint = args.weights / f"model_e{args.epoch:03d}.pt"
    manifest = json.loads((args.weights / "manifest.json").read_text())
    if sha256(checkpoint) != manifest[checkpoint.name]["sha256"]:
        raise SystemExit(f"{checkpoint.name} differs from the pod's manifest")
    install_deformable_kernel("hub")
    rows = verify_deformable_kernel(device)
    if not all(row["passed"] for row in rows):
        install_deformable_kernel("pytorch")
        print("KERNEL_FALLBACK pytorch", flush=True)
    model, _ = load_checkpoint(checkpoint, device)
    records, _ = records_for(args.annotations)
    cache, seconds = cache_validation(model, records, args.train_images, device, bf16=False, floor=0.02)
    (args.output / "exp139_val_cache.json").write_text(json.dumps(cache))
    print(f"EXP139_VALIDATION photographs={len(cache)} seconds={seconds:.1f}", flush=True)
    test = sorted(args.test_images.glob("*.jpeg"))
    predictions = {}
    for position, path in enumerate(test, start=1):
        predictions[path.name] = predict_photograph(model, read_gray(path), device, False, d4=False, floor=0.02)
        if position % 30 == 0:
            print(f"EXP139_TEST {position}/{len(test)}", flush=True)
    (args.output / "exp139_test_cache.json").write_text(json.dumps(predictions))
    print(f"EXP139_TEST_DONE photographs={len(predictions)}", flush=True)


if __name__ == "__main__":
    main()
