"""Refine an existing submission's masks and prove the rule matches validation.

Order matters here. The kernel refines the validation emission first and asserts
the exact canonical PQ that was chosen on validation. Only then does it touch
the test CSV. `CLAUDE.md` section 5 records the deployment that quietly differed
from its validation and was caught only by counting rows; this makes that
failure impossible rather than unlikely.

Boundaries change; instances do not. The output row count is compared against
the input and the masks are checked pixel-disjoint before anything is written.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import csv
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
from pycocotools import mask as mu
import torch

from experiments.exp_033_inference.src.sweep import pq_rle, records_for
from experiments.exp_092_refiner.src.evaluate import (BASELINE_PQ, load_model,
                                                      refine_photo, repaint)
from shared.utils import aggregate_pq

NATIVE = 2048


def load_submission(path: Path) -> dict[str, list[dict]]:
    grouped = defaultdict(list)
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            stem = row["filament_id"].rsplit("_", 1)[0]
            grouped[stem].append(dict(
                filament_id=row["filament_id"], score=1.0 / (1 + len(grouped[stem])),
                rle=dict(size=[NATIVE, NATIVE], counts=row["segmentation_rle"])))
    return dict(grouped)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--emission", type=Path, required=True)
    parser.add_argument("--submission", type=Path, required=True)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--train-images", type=Path, required=True)
    parser.add_argument("--test-images", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--threshold", type=float, required=True)
    parser.add_argument("--guard", type=float, required=True)
    parser.add_argument("--expected-pq", type=float, required=True)
    parser.add_argument("--min-area", type=int, default=150)
    parser.add_argument("--size", type=int, default=256)
    parser.add_argument("--context", type=float, default=1.8)
    parser.add_argument("--tta", action="store_true")
    args = parser.parse_args()
    cv2.setNumThreads(1)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model, state = load_model(args.checkpoint, device)
    args.output.mkdir(parents=True, exist_ok=True)

    emission = json.loads(args.emission.read_text())
    records, split = records_for(args.annotations)
    assert set(emission) == set(records)
    unchanged = aggregate_pq([pq_rle([row["rle"] for row in emission[name]], truth)
                              for name, passes in records.items() for _, truth in passes])
    assert abs(unchanged["pq"] - BASELINE_PQ) < 1e-9, unchanged
    print("baseline", json.dumps(unchanged), flush=True)

    metrics = []
    for position, name in enumerate(sorted(emission), start=1):
        image = cv2.imread(str(args.train_images / name), cv2.IMREAD_GRAYSCALE)
        assert image is not None, name
        probabilities = refine_photo(model, image, emission[name], device, args.size,
                                     args.context, args.tta)
        predicted = repaint(emission[name], probabilities, args.threshold, args.guard,
                            args.min_area)
        metrics.extend(pq_rle(predicted, truth) for _, truth in records[name])
        if position % 25 == 0 or position == len(emission):
            print(f"validate {position}/{len(emission)}", flush=True)
    replay = aggregate_pq(metrics)
    print("validation replay", json.dumps(replay), flush=True)
    (args.output / "validation_replay.json").write_text(json.dumps(replay, indent=2))
    assert abs(replay["pq"] - args.expected_pq) < 1e-6, (
        f"deployment rule does not reproduce validation: {replay['pq']} vs {args.expected_pq}")

    submission = load_submission(args.submission)
    rows, changed = [], 0
    for position, name in enumerate(sorted(submission), start=1):
        entries = submission[name]
        image = cv2.imread(str(args.test_images / f"{name}.jpeg"), cv2.IMREAD_GRAYSCALE)
        assert image is not None, name
        probabilities = refine_photo(model, image, entries, device, args.size,
                                     args.context, args.tta)
        painted = repaint(entries, probabilities, args.threshold, args.guard, args.min_area)
        assert len(painted) == len(entries), (name, len(painted), len(entries))
        accumulator = np.zeros((NATIVE, NATIVE), np.int16)
        for entry, rle in zip(entries, painted):
            accumulator += mu.decode(rle)
            if rle["counts"] != entry["rle"]["counts"]:
                changed += 1
            rows.append(dict(filament_id=entry["filament_id"],
                             segmentation_rle=rle["counts"]))
        assert int(accumulator.max()) <= 1, f"overlapping masks in {name}"
        if position % 25 == 0 or position == len(submission):
            print(f"deploy {position}/{len(submission)} rows={len(rows)}", flush=True)

    destination = args.output / "refined_submission.csv"
    with destination.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["filament_id", "segmentation_rle"])
        writer.writeheader()
        writer.writerows(rows)
    summary = dict(status="complete", split=split.summary(), baseline=unchanged,
                   validation_replay=replay, expected_pq=args.expected_pq,
                   threshold=args.threshold, guard=args.guard, tta=bool(args.tta),
                   checkpoint_epoch=int(state.get("epoch", -1)),
                   input_rows=sum(len(v) for v in submission.values()),
                   output_rows=len(rows), changed_masks=changed,
                   sha256=hashlib.sha256(destination.read_bytes()).hexdigest().upper())
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
