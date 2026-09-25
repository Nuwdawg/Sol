"""Write the test submission from a checkpoint, and optionally submit it, on the pod itself.

No validation and no sweep: the emission rule is passed in (exp139's validated
confidence 0.85, no erosion, minimum area 100, single view, float32). Overlap and
emptiness are still checked, because a rejected submission costs a slot.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
from pathlib import Path

import numpy as np
import torch
from pycocotools import mask as mu

from experiments.exp_139_mask2former.src.data import NATIVE, read_gray
from experiments.exp_139_mask2former.src.evaluate import event, load_checkpoint
from experiments.exp_139_mask2former.src.model import predict_photograph
from experiments.exp_139_mask2former.src.submit import emit


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--test-images", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--conf", type=float, default=0.85)
    parser.add_argument("--grow", type=int, default=0)
    parser.add_argument("--min-area", type=int, default=100)
    parser.add_argument("--msda", choices=("pytorch", "hub"), default="hub")
    parser.add_argument("--submit", action="store_true")
    parser.add_argument("--competition", default="filament-segmentation-2026")
    parser.add_argument("--message", default="")
    parser.add_argument("--log", type=Path, default=None)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.msda != "pytorch" and device.type == "cuda":
        from experiments.exp_139_mask2former.src.fast_ops import (install_deformable_kernel,
                                                                  verify_deformable_kernel)
        install_deformable_kernel(args.msda)
        if not all(row["passed"] for row in verify_deformable_kernel(device)):
            install_deformable_kernel("pytorch")
            event("KERNEL_FALLBACK", args.log, source="pytorch")
    model, state = load_checkpoint(args.checkpoint, device)
    test = sorted(args.test_images.glob("*.jpeg"))
    if len(test) != 180:
        raise SystemExit(f"expected 180 test photographs, found {len(test)}")
    destination = args.output / f"submission_{args.checkpoint.stem}.csv"
    written = 0
    with destination.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["filament_id", "segmentation_rle"])
        for position, path in enumerate(test, start=1):
            entries = predict_photograph(model, read_gray(path), device, False, d4=False, floor=0.02)
            claimed = np.zeros((NATIVE, NATIVE), np.uint8)
            for index, rle in enumerate(emit(entries, args.conf, args.grow, args.min_area), start=1):
                mask = mu.decode(rle)
                if not mask.sum() or (claimed & mask).any():
                    raise SystemExit(f"bad instance for {path.stem}")
                claimed |= mask
                writer.writerow([f"{path.stem}_{index}", rle["counts"]])
                written += 1
            if position % 60 == 0:
                print(f"EMIT {position}/180 rows={written}", flush=True)
    digest = hashlib.sha256(destination.read_bytes()).hexdigest().upper()
    event("EMIT_DONE", args.log, checkpoint=args.checkpoint.name, epoch=state.get("epoch"),
          rows=written, sha256=digest, rule=dict(conf=args.conf, grow=args.grow, min_area=args.min_area))
    if args.submit:
        message = args.message or (f"exp142 full-data Mask2Former {args.checkpoint.stem}, single view fp32, "
                                   f"conf{args.conf} grow{args.grow} area{args.min_area}")
        result = subprocess.run(["kaggle", "competitions", "submit", args.competition,
                                 "-f", str(destination), "-m", message],
                                capture_output=True, text=True, timeout=1800)
        event("SUBMITTED", args.log, checkpoint=args.checkpoint.name, returncode=result.returncode,
              output=(result.stdout + result.stderr).strip()[-300:])
        return result.returncode
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
