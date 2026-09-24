"""Score submission instances with the exp085 semantic and spine fields.

The pruning rule was validated against 8-bit probability PNGs, so the deployment
path quantizes to uint8 and reads the value back before measuring anything. The
optional verification pass recomputes the stored validation features from the
same checkpoint and refuses to continue unless they reproduce. That is the cheap
invariant which catches a deployment that silently differs from what was tuned.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import csv
import json
from pathlib import Path

import cv2
import numpy as np
from pycocotools import mask as mu
import torch

from experiments.exp_085_morden_spine.src.evaluate import predict
from experiments.exp_085_morden_spine.src.model import MordenSpineNet
from scripts.audit_morden_fusion import mask_features

NATIVE = 2048
CENTRE = 1024.0


def quantize(probability: np.ndarray) -> np.ndarray:
    """Reproduce the validated PNG round trip exactly."""
    stored = np.rint(np.clip(probability, 0, 1) * 255).astype(np.uint8)
    return stored.astype(np.float32) / 255.0


def load_submission(path: Path) -> dict[str, list[dict]]:
    grouped = defaultdict(list)
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            stem = row["filament_id"].rsplit("_", 1)[0]
            grouped[stem].append(dict(filament_id=row["filament_id"],
                                      rle=dict(size=[NATIVE, NATIVE],
                                               counts=row["segmentation_rle"])))
    return dict(grouped)


def load_model(checkpoint: Path, device: torch.device):
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    model = MordenSpineNet(base_channels=int(state["base_channels"]))
    model.load_state_dict(state["model"])
    return model.to(device).eval(), state


def measure(model, image_path: Path, instances: list[dict], device, size: int,
            tta: bool) -> list[dict]:
    image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if image is None or image.shape != (NATIVE, NATIVE):
        raise ValueError(f"cannot read {image_path}")
    semantic, spine = predict(model, image, device, size, tta)
    semantic, spine = quantize(semantic), quantize(spine)
    rows = []
    for entry in instances:
        mask = mu.decode(entry["rle"])
        ys, xs = np.nonzero(mask)
        rows.append(dict(
            filament_id=entry.get("filament_id"),
            area=int(mask.sum()),
            radius=float(np.hypot(xs.mean() - CENTRE, ys.mean() - CENTRE) / CENTRE),
            **mask_features(mask, semantic, spine)))
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--images", type=Path, required=True)
    parser.add_argument("--submission", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--size", type=int, default=None,
                        help="defaults to the checkpoint's own image_size")
    parser.add_argument("--tta", action="store_true")
    parser.add_argument("--verify-features", type=Path, default=None)
    parser.add_argument("--verify-images", type=Path, default=None)
    parser.add_argument("--verify-tolerance", type=float, default=0.02)
    args = parser.parse_args()
    cv2.setNumThreads(1)
    device = torch.device(args.device)
    model, state = load_model(args.checkpoint, device)
    if args.size is None:
        args.size = int(state["image_size"])
    print("checkpoint epoch", state.get("epoch"), "size", args.size, flush=True)
    args.output.mkdir(parents=True, exist_ok=True)

    verification = None
    if args.verify_features is not None:
        stored = json.loads(args.verify_features.read_text())
        worst, checked = 0.0, 0
        for position, (name, entries) in enumerate(sorted(stored.items()), start=1):
            produced = measure(model, args.verify_images / name, entries, device,
                               args.size, args.tta)
            for original, fresh in zip(entries, produced):
                for key in ("semantic_mean", "semantic_p75", "semantic_coverage_035",
                            "spine_mean", "spine_p90"):
                    worst = max(worst, abs(original[key] - fresh[key]))
                    checked += 1
            if position % 20 == 0:
                print(f"verify {position}/{len(stored)} worst={worst:.5f}", flush=True)
        verification = dict(photographs=len(stored), comparisons=checked,
                            worst_absolute_difference=worst,
                            tolerance=args.verify_tolerance)
        (args.output / "verification.json").write_text(json.dumps(verification, indent=2))
        print("VERIFY", json.dumps(verification), flush=True)
        if worst > args.verify_tolerance:
            raise SystemExit(f"validation features did not reproduce: {worst}")

    submission = load_submission(args.submission)
    features = {}
    for position, name in enumerate(sorted(submission), start=1):
        features[name] = measure(model, args.images / f"{name}.jpeg", submission[name],
                                 device, args.size, args.tta)
        if position % 20 == 0 or position == len(submission):
            (args.output / "submission_features.json").write_text(json.dumps(features))
            print(f"score {position}/{len(submission)} "
                  f"instances={sum(len(v) for v in features.values())}", flush=True)
    (args.output / "submission_features.json").write_text(json.dumps(features))
    summary = dict(status="complete", photographs=len(features),
                   instances=sum(len(v) for v in features.values()),
                   checkpoint_epoch=int(state.get("epoch", -1)),
                   base_channels=int(state["base_channels"]),
                   tta=bool(args.tta), size=args.size, verification=verification)
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
