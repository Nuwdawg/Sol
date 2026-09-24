"""Build a test submission with frozen U-Net support for weak D4 proposals."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import cv2
import numpy as np
from pycocotools import mask as mu
import torch

from experiments.exp_033_inference.src.cached_tta import fuse
from scripts.evaluate_semantic_complement import (
    build_model,
    emit_with_baseline,
    emit_yolo,
    predict_probability,
)
from shared.utils import check_no_overlap


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--views", type=Path, nargs=2, required=True)
    parser.add_argument("--images", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    cv2.setNumThreads(1)
    assert torch.cuda.is_available()
    device = torch.device("cuda")
    print("ACCELERATOR", torch.cuda.get_device_name(0), flush=True)

    model = build_model()
    model.load_state_dict(torch.load(args.checkpoint, map_location="cpu", weights_only=True))
    model = model.to(device).eval()

    views = {}
    for path in args.views:
        shard = json.loads(path.read_text())
        assert not views.keys() & shard.keys()
        views.update(shard)
    expected = {path.name for path in args.images.glob("*.jpeg")}
    assert len(views) == 180 and set(views) == expected

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    rows, counts, changes = [], {}, {}
    for position, name in enumerate(sorted(views), 1):
        entries = fuse(views[name], link=.5, vote=.5)
        baseline = emit_yolo(entries, set(), kernel)
        probability = predict_probability(model, args.images / name, device)
        selected = set()
        for index, entry in enumerate(entries):
            if not (.26 <= entry["score"] < .35):
                continue
            mask = mu.decode(entry["rle"])
            small = cv2.resize(mask, (1024, 1024), interpolation=cv2.INTER_NEAREST).astype(bool)
            values = probability[small]
            if len(values) and float(values.mean()) >= .80:
                selected.add(index)
        claimed = (mu.decode(mu.merge(baseline)).astype(bool) if baseline
                   else np.zeros((2048, 2048), dtype=bool))
        emitted = emit_with_baseline(baseline, claimed, entries, selected, kernel)
        counts[name] = len(emitted)
        if len(emitted) != len(baseline):
            changes[name] = dict(baseline=len(baseline), semantic=len(emitted),
                                 selected=len(selected))
        for index, rle in enumerate(emitted, 1):
            rows.append(dict(filament_id=f"{Path(name).stem}_{index}",
                             segmentation_rle=rle["counts"]))
        if position % 10 == 0:
            print(f"INFERENCE {position}/{len(views)}", flush=True)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["filament_id", "segmentation_rle"])
        writer.writeheader()
        writer.writerows(rows)
    check_no_overlap(str(args.output))
    report = dict(status="complete", rule=dict(confidence=.26, semantic_mean=.80),
                  photographs=len(views), rows=len(rows), changed_photographs=len(changes),
                  changes=changes, instance_counts=counts)
    args.output.with_suffix(".json").write_text(json.dumps(report, indent=2))
    print(json.dumps({key: value for key, value in report.items()
                      if key not in ("changes", "instance_counts")}), flush=True)


if __name__ == "__main__":
    main()
