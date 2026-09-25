"""Write the test submission for exp141's best eight-view fusion.

exp141 measured this configuration on validation at PQ 0.47002, against 0.47052 for the
same checkpoint with one view. The difference is inside the noise, so this file exists to
put the variant on the leaderboard, not because validation promised a gain. It reuses
exp141's own `raw_views` and `fused_cache`, so the deployed rule is the validated rule.

    python -m experiments.exp_141_soup_and_tta.src.emit_fusion \
        --checkpoint model_e040.pt --test-images .../test_images --output .
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from pycocotools import mask as mu

from experiments.exp_139_mask2former.src.data import NATIVE
from experiments.exp_139_mask2former.src.evaluate import event, load_checkpoint
from experiments.exp_139_mask2former.src.submit import emit
from experiments.exp_141_soup_and_tta.src.soup_tta import fused_cache, raw_views


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--test-images", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    # exp141's winning row: eight views, link 0.60, vote 0.375, exp119 support, conf 0.70.
    parser.add_argument("--link", type=float, default=0.6)
    parser.add_argument("--vote", type=float, default=0.375)
    parser.add_argument("--support", default="exp119")
    parser.add_argument("--conf", type=float, default=0.7)
    parser.add_argument("--grow", type=int, default=0)
    parser.add_argument("--min-area", type=int, default=100)
    parser.add_argument("--msda", choices=("pytorch", "hub"), default="hub")
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
    names = sorted(path.name for path in args.test_images.glob("*.jpeg"))
    if len(names) != 180:
        raise SystemExit(f"expected 180 test photographs, found {len(names)}")
    records = {name: None for name in names}
    views = raw_views(model, records, args.test_images, device, floor=0.02)
    cache = fused_cache(views, records, list(range(8)), args.link, args.vote, args.support)

    destination = args.output / f"submission_fusion_{args.checkpoint.stem}.csv"
    written = 0
    with destination.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["filament_id", "segmentation_rle"])
        for name in names:
            claimed = np.zeros((NATIVE, NATIVE), np.uint8)
            stem = Path(name).stem
            for index, rle in enumerate(emit(cache[name], args.conf, args.grow, args.min_area), start=1):
                binary = mu.decode(rle)
                if not binary.sum() or (claimed & binary).any():
                    raise SystemExit(f"bad instance for {stem}")
                claimed |= binary
                writer.writerow([f"{stem}_{index}", rle["counts"]])
                written += 1
    digest = hashlib.sha256(destination.read_bytes()).hexdigest().upper()
    event("EMIT_DONE", args.log, checkpoint=args.checkpoint.name, epoch=state.get("epoch"),
          rows=written, sha256=digest, csv=destination.name,
          rule=dict(views=8, link=args.link, vote=args.vote, support=args.support,
                    conf=args.conf, grow=args.grow, min_area=args.min_area))
    print("FUSION_CSV " + json.dumps(dict(name=destination.name, rows=written, sha256=digest)), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
