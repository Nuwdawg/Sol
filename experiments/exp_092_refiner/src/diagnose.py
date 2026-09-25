"""Locate where exp092's crop-level gains disappear before canonical PQ.

Training reported 52 validation pairs crossing IoU 0.5 and 12 falling below it,
but the exact sweep moved one true positive. Two explanations were already
measured and refuted (window truncation; fragments of matched truth). What is
left is the measurement chain itself, so this scores the model's own output one
step at a time for every (proposal, annotation record) pair:

- crop_seed:      the seed mask at crop resolution against the crop target.
                  Compared with the native seed IoU it shows whether resampling
                  to 256 inflates IoU, which would make the training monitor's
                  crossings partly an artifact.
- crop_plain:     the refined mask exactly as training measured it.
- native_plain:   pasted back to 2048 without test-time augmentation.
- native_tta:     pasted back with the four-flip average used in deployment.
- native_tta_040: as above at the sweep's best threshold, 0.40.

Each column changes one thing, so the lossy step is identified directly.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from pycocotools import mask as mu
import torch

from experiments.exp_033_inference.src.sweep import records_for
from experiments.exp_092_refiner.src.crops import channels_for, resample, window
from experiments.exp_092_refiner.src.evaluate import load_model

NATIVE = 2048
VARIANTS = ("crop_seed", "crop_plain", "native_plain", "native_tta", "native_tta_040")


@torch.no_grad()
def probability(model, stack: np.ndarray, device, tta: bool) -> np.ndarray:
    tensor = torch.from_numpy((stack.astype(np.float32) / 255.0 - .5) / .25)[None].to(device)
    views = [()] if not tta else [(), (2,), (3,), (2, 3)]
    total = None
    with torch.autocast(device_type=device.type, dtype=torch.float16,
                        enabled=device.type == "cuda"):
        for dims in views:
            viewed = torch.flip(tensor, dims=dims) if dims else tensor
            value = torch.sigmoid(model(viewed))
            if dims:
                value = torch.flip(value, dims=dims)
            total = value if total is None else total + value
    return (total / len(views))[0, 0].float().cpu().numpy()


def iou(a: np.ndarray, b: np.ndarray) -> float:
    union = np.logical_or(a, b).sum()
    return float(np.logical_and(a, b).sum() / union) if union else 0.


def native_iou(prob: np.ndarray, box, threshold: float, truth: np.ndarray) -> float:
    x0, y0, side = box
    full = np.zeros((NATIVE, NATIVE), bool)
    full[y0:y0 + side, x0:x0 + side] = cv2.resize(
        prob, (side, side), interpolation=cv2.INTER_LINEAR) >= threshold
    return iou(full, truth)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--emission", type=Path, required=True)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--images", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--size", type=int, default=256)
    parser.add_argument("--context", type=float, default=1.8)
    args = parser.parse_args()
    cv2.setNumThreads(1)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model, state = load_model(args.checkpoint, device)
    emission = json.loads(args.emission.read_text())
    records, split = records_for(args.annotations)
    args.output.mkdir(parents=True, exist_ok=True)

    rows = []
    for position, name in enumerate(sorted(emission), start=1):
        entries = emission[name]
        if not entries:
            continue
        image = cv2.imread(str(args.images / name), cv2.IMREAD_GRAYSCALE)
        assert image is not None, name
        planes = channels_for(image)
        rles = [entry["rle"] for entry in entries]
        seeds = [mu.decode(rle) for rle in rles]
        boxes = [window(seed, args.size, args.context) for seed in seeds]
        stacks = [np.stack([resample(plane, box, args.size, cv2.INTER_AREA) for plane in planes]
                           + [resample(seed * 255, box, args.size, cv2.INTER_NEAREST)])
                  for seed, box in zip(seeds, boxes)]
        plain = [probability(model, stack, device, False) for stack in stacks]
        averaged = [probability(model, stack, device, True) for stack in stacks]
        for _, truth in records[name]:
            if not truth:
                continue
            table = np.asarray(mu.iou(rles, truth, [0] * len(truth)))
            for i in range(len(rles)):
                g = int(table[i].argmax())
                seed_iou = float(table[i, g])
                if seed_iou < .2:
                    continue
                target = mu.decode(truth[g]).astype(bool)
                crop_target = resample(target.astype(np.uint8) * 255, boxes[i], args.size,
                                       cv2.INTER_NEAREST) > 127
                rows.append(dict(
                    photo=name, instance=i, seed=seed_iou,
                    crop_seed=iou(stacks[i][3] > 127, crop_target),
                    crop_plain=iou(plain[i] > .5, crop_target),
                    native_plain=native_iou(plain[i], boxes[i], .5, target),
                    native_tta=native_iou(averaged[i], boxes[i], .5, target),
                    native_tta_040=native_iou(averaged[i], boxes[i], .40, target)))
        if position % 20 == 0 or position == len(emission):
            print(f"diagnose {position}/{len(emission)} pairs={len(rows)}", flush=True)

    seed = np.asarray([row["seed"] for row in rows])
    summary = dict(pairs=len(rows), checkpoint_epoch=int(state.get("epoch", -1)),
                   native_seed_mean=float(seed.mean()),
                   native_seed_above_half=int((seed > .5).sum()))
    for variant in VARIANTS:
        value = np.asarray([row[variant] for row in rows])
        summary[variant] = dict(
            mean=float(value.mean()),
            mean_minus_native_seed=float((value - seed).mean()),
            above_half=int((value > .5).sum()),
            crossed_up=int(((seed <= .5) & (value > .5)).sum()),
            crossed_down=int(((seed > .5) & (value <= .5)).sum()))
    (args.output / "pairs.json").write_text(json.dumps(rows))
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
