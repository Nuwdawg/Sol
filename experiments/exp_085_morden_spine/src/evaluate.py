"""Evaluate a trained mask/spine checkpoint on the canonical PQ split."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
from pycocotools import mask as mask_util
import torch

from experiments.exp_004_spine_tpu.src.decompose import connected_components, decompose
from experiments.exp_033_inference.src.sweep import encode, pq_rle, records_for
from experiments.exp_085_morden_spine.src.model import MordenSpineNet
from experiments.exp_085_morden_spine.src.prepare import image_channels
from shared.utils import aggregate_pq


NATIVE = 2048


def predict(model, image: np.ndarray, device: torch.device, size: int, tta: bool) -> tuple[np.ndarray, np.ndarray]:
    features = image_channels(image, size).astype(np.float32) / 255.0
    tensor = torch.from_numpy((features - 0.5) / 0.25)[None].to(device)
    views = [()] if not tta else [(), (2,), (3,), (2, 3)]
    mask_sum = spine_sum = None
    with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.float16):
        for dims in views:
            viewed = torch.flip(tensor, dims=dims) if dims else tensor
            mask_logits, spine_logits = model(viewed)
            mask = torch.sigmoid(mask_logits)
            spine = torch.sigmoid(spine_logits)
            if dims:
                mask, spine = torch.flip(mask, dims=dims), torch.flip(spine, dims=dims)
            mask_sum = mask if mask_sum is None else mask_sum + mask
            spine_sum = spine if spine_sum is None else spine_sum + spine
    mask = (mask_sum / len(views))[0, 0].float().cpu().numpy()
    spine = (spine_sum / len(views))[0, 0].float().cpu().numpy()
    return (
        cv2.resize(mask, (NATIVE, NATIVE), interpolation=cv2.INTER_LINEAR),
        cv2.resize(spine, (NATIVE, NATIVE), interpolation=cv2.INTER_LINEAR),
    )


def rles(instances: list[np.ndarray]) -> list[dict]:
    return [encode(mask) for mask in instances]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--annotations", required=True)
    parser.add_argument("--images", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--tta", action="store_true")
    parser.add_argument("--probabilities-only", action="store_true",
                        help="persist fields for external fusion without the slow standalone rule sweep")
    args = parser.parse_args()

    output = Path(args.output)
    probability_dir = output / "probabilities"
    probability_dir.mkdir(parents=True, exist_ok=True)
    saved = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    model = MordenSpineNet(base_channels=int(saved["base_channels"]))
    model.load_state_dict(saved["model"], strict=True)
    device = torch.device(args.device)
    model = model.to(device).eval()
    size = int(saved["image_size"])
    records, split = records_for(args.annotations)

    probabilities = {}
    for index, name in enumerate(sorted(records), start=1):
        raw = cv2.imread(str(Path(args.images) / name), cv2.IMREAD_GRAYSCALE)
        if raw is None or raw.shape != (NATIVE, NATIVE):
            raise ValueError(f"invalid image {name}")
        mask, spine = predict(model, raw, device, size, args.tta)
        # Quantized maps keep the terminal artifact small and allow CPU-only
        # fusion experiments after the checkpoint is downloaded.
        mask8 = np.rint(np.clip(mask, 0, 1) * 255).astype(np.uint8)
        spine8 = np.rint(np.clip(spine, 0, 1) * 255).astype(np.uint8)
        cv2.imwrite(str(probability_dir / f"{Path(name).stem}_mask.png"), mask8)
        cv2.imwrite(str(probability_dir / f"{Path(name).stem}_spine.png"), spine8)
        probabilities[name] = (mask8, spine8, raw > 10)
        if index % 10 == 0 or index == len(records):
            print(f"inference {index}/{len(records)}", flush=True)

    if args.probabilities_only:
        report = {
            "status": "probabilities_complete",
            "checkpoint_sha256": hashlib.sha256(Path(args.checkpoint).read_bytes()).hexdigest(),
            "checkpoint_epoch": saved["epoch"],
            "checkpoint_validation": saved["validation"],
            "split": split.summary(),
            "tta": args.tta,
            "photographs": len(probabilities),
        }
        (output / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        print("PROBABILITIES_COMPLETE " + json.dumps(report, sort_keys=True), flush=True)
        return

    rules = []
    for mask_threshold in (0.25, 0.35, 0.45):
        for min_area in (150, 300):
            rules.append(("components", mask_threshold, None, 1, min_area))
            for spine_threshold in (0.25, 0.35, 0.45):
                for closing in (3, 7):
                    rules.append(("spine", mask_threshold, spine_threshold, closing, min_area))

    results = []
    for rule_index, (method, mask_threshold, spine_threshold, closing, min_area) in enumerate(rules, start=1):
        scores = []
        predicted_count = 0
        mask_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        spine_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (closing, closing))
        for name, passes in records.items():
            mask8, spine8, disk = probabilities[name]
            mask = ((mask8 >= round(mask_threshold * 255)) & disk).astype(np.uint8)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, mask_kernel)
            if method == "components":
                instances = connected_components(mask, min_area=min_area)
            else:
                spine = ((spine8 >= round(spine_threshold * 255)) & (mask > 0)).astype(np.uint8)
                spine = cv2.morphologyEx(spine, cv2.MORPH_CLOSE, spine_kernel)
                instances = decompose(mask, spine, min_area=min_area, min_spine_area=8)
            predicted = rles(instances)
            predicted_count += len(predicted)
            for _, truth in passes:
                scores.append(pq_rle(predicted, truth))
        metric = aggregate_pq(scores)
        row = {
            "method": method,
            "mask_threshold": mask_threshold,
            "spine_threshold": spine_threshold,
            "spine_closing": closing,
            "min_area": min_area,
            "predicted_instances_across_photos": predicted_count,
            **metric,
        }
        results.append(row)
        print(f"rule {rule_index}/{len(rules)} " + json.dumps(row, sort_keys=True), flush=True)

    results.sort(key=lambda row: row["pq"], reverse=True)
    report = {
        "status": "complete",
        "checkpoint_sha256": hashlib.sha256(Path(args.checkpoint).read_bytes()).hexdigest(),
        "checkpoint_epoch": saved["epoch"],
        "checkpoint_validation": saved["validation"],
        "split": split.summary(),
        "tta": args.tta,
        "best": results[0],
        "rules": results,
    }
    (output / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("BEST " + json.dumps(results[0], sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
