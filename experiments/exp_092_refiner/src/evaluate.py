"""Apply the refiner to the retained emission and score exact canonical PQ.

Refinement changes boundaries only. Every instance that enters is an instance
the baseline already emitted, so the instance count cannot rise and no new
object is invented. Refined masks can overlap, so they are repainted in the
original score order with the same claim rule the emission used.

The unchanged baseline is scored first and must reproduce 0.4558638687 exactly.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from pycocotools import mask as mu
import torch

from experiments.exp_033_inference.src.sweep import encode, pq_rle, records_for
from experiments.exp_092_refiner.src.crops import channels_for, resample, window
from experiments.exp_092_refiner.src.model import RefinerNet
from shared.utils import aggregate_pq

NATIVE = 2048
BASELINE_PQ = 0.4558638687292126


def load_model(checkpoint: Path, device: torch.device):
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    model = RefinerNet(base=int(state["base"]))
    model.load_state_dict(state["model"])
    return model.to(device).eval(), state


@torch.no_grad()
def refine_photo(model, image: np.ndarray, entries: list[dict], device, size: int,
                 context: float, tta: bool) -> list[np.ndarray]:
    planes = channels_for(image)
    outputs = []
    for entry in entries:
        seed = mu.decode(entry["rle"])
        box = window(seed, size, context)
        stack = np.stack([resample(plane, box, size, cv2.INTER_AREA) for plane in planes]
                         + [resample(seed * 255, box, size, cv2.INTER_NEAREST)])
        tensor = torch.from_numpy((stack.astype(np.float32) / 255.0 - .5) / .25)[None].to(device)
        views = [()] if not tta else [(), (2,), (3,), (2, 3)]
        total = None
        with torch.autocast(device_type=device.type, dtype=torch.float16,
                            enabled=device.type == "cuda"):
            for dims in views:
                viewed = torch.flip(tensor, dims=dims) if dims else tensor
                probability = torch.sigmoid(model(viewed))
                if dims:
                    probability = torch.flip(probability, dims=dims)
                total = probability if total is None else total + probability
        crop = (total / len(views))[0, 0].float().cpu().numpy()
        x0, y0, side = box
        full = np.zeros((NATIVE, NATIVE), np.float32)
        full[y0:y0 + side, x0:x0 + side] = cv2.resize(crop, (side, side),
                                                      interpolation=cv2.INTER_LINEAR)
        outputs.append(full)
    return outputs


def repaint(entries: list[dict], probabilities: list[np.ndarray], threshold: float,
            guard: float, min_area: int) -> list[dict]:
    """Threshold, fall back when the refinement disagrees, then claim pixels once."""
    claimed = np.zeros((NATIVE, NATIVE), np.uint8)
    order = sorted(range(len(entries)), key=lambda i: -float(entries[i].get("score", 0.)))
    painted = [None] * len(entries)
    for index in order:
        seed = mu.decode(entries[index]["rle"])
        candidate = (probabilities[index] >= threshold).astype(np.uint8)
        if candidate.sum() < min_area:
            candidate = seed
        else:
            overlap = float((candidate & seed).sum())
            union = float(((candidate | seed) > 0).sum())
            if union <= 0 or overlap / union < guard:
                candidate = seed
        remaining = (candidate & (1 - claimed)).astype(np.uint8)
        if int(remaining.sum()) < min_area:
            remaining = (seed & (1 - claimed)).astype(np.uint8)
        if int(remaining.sum()) < min_area:
            continue
        claimed |= remaining
        painted[index] = encode(remaining)
    return [rle for rle in painted if rle is not None]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--emission", type=Path, required=True)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--images", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--size", type=int, default=256)
    parser.add_argument("--context", type=float, default=1.8)
    parser.add_argument("--min-area", type=int, default=150)
    parser.add_argument("--tta", action="store_true")
    args = parser.parse_args()
    cv2.setNumThreads(1)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model, state = load_model(args.checkpoint, device)
    emission = json.loads(args.emission.read_text())
    records, split = records_for(args.annotations)
    assert set(emission) == set(records)
    args.output.mkdir(parents=True, exist_ok=True)

    baseline = aggregate_pq([pq_rle([row["rle"] for row in emission[name]], truth)
                             for name, passes in records.items() for _, truth in passes])
    assert abs(baseline["pq"] - BASELINE_PQ) < 1e-9, baseline
    print("baseline", json.dumps(baseline), flush=True)

    cache = {}
    for position, name in enumerate(sorted(emission), start=1):
        image = cv2.imread(str(args.images / name), cv2.IMREAD_GRAYSCALE)
        assert image is not None, name
        cache[name] = refine_photo(model, image, emission[name], device, args.size,
                                   args.context, args.tta)
        if position % 20 == 0 or position == len(emission):
            print(f"refine {position}/{len(emission)}", flush=True)

    results = []
    for threshold in (.35, .40, .45, .50, .55, .60):
        for guard in (.0, .20, .30, .40, .50):
            metrics = []
            for name, passes in records.items():
                predicted = repaint(emission[name], cache[name], threshold, guard,
                                    args.min_area)
                metrics.extend(pq_rle(predicted, truth) for _, truth in passes)
            row = dict(threshold=threshold, guard=guard, **aggregate_pq(metrics))
            results.append(row)
            print(json.dumps(row), flush=True)
    results.sort(key=lambda row: -row["pq"])
    report = dict(status="complete", split=split.summary(), baseline=baseline,
                  checkpoint_epoch=int(state.get("epoch", -1)),
                  checkpoint_metrics=state.get("metrics"), tta=bool(args.tta),
                  best=results[0], results=results)
    (args.output / "report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(dict(baseline=baseline, best=results[0]), indent=2), flush=True)


if __name__ == "__main__":
    main()
