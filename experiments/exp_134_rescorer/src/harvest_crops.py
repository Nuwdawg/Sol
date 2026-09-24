"""exp134: crops for the rescorer, labelled with the question PQ actually asks.

The target is the 172 validation truths the detector *found* and scored below the
0.35 confidence floor - the largest measured bucket, worth +0.054 PQ if fully
recovered (validation 0.5048, public 0.43-0.44).

Why this is not exp025 repeated
-------------------------------
exp025 built the same crops and failed, and CLAUDE.md section 4 records exactly
why: it labelled a crop positive if **any** annotator drew a filament there. PQ
scores each record against **one** annotator. The model reached 72% accuracy at
"would somebody draw this" and 17% at "did this person draw it", and the 55-point
gap was the labelling decision, not the model.

So the unit here is **(photograph, record, proposal)**, not (photograph, proposal).
A proposal is positive for a record when it matches that record's truth at
IoU >= 0.5. The same crop is legitimately positive for one annotator and negative
for another, which is what the metric rewards. exp025's stored artifacts cannot be
reused: they "retain union labels and no record/candidate mapping".

Measured on exp091's training harvest under the corrected rule:
  12,081 examples, 5,530 positive (45.8%)
  sub-threshold only: 5,281 examples, 1,021 positive (19.3%)
versus 25.1% on the validation sub-threshold pool, so the transfer assumption holds.
Under the union rule it would have been 54% - a different problem entirely.

No inference happens here. exp091 already ran the detector over all 601 training
photographs on a T4 (5,410 s, parity 1.0 against the emission path), so this reads
cached RLEs and costs only CPU.
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
from pycocotools import mask as mu

from experiments.exp_025_verifier.src.harvest import crop_of
from shared.data_split import assert_disjoint, make_split

FULL = 2048
MATCH = 0.5
BASE_CONF = 0.35


def truth_rles(annotations: list[dict]) -> list[dict]:
    out = []
    for annotation in annotations:
        segmentation = annotation.get("segmentation")
        if not segmentation:
            continue
        if isinstance(segmentation, list):
            rings = [r for r in segmentation if len(r) >= 6]
            if not rings:
                continue
            rle = mu.merge(mu.frPyObjects(rings, FULL, FULL))
        elif isinstance(segmentation["counts"], list):
            rle = mu.frPyObjects(segmentation, FULL, FULL)
        else:
            rle = segmentation
        out.append(dict(size=rle["size"],
                        counts=rle["counts"].decode("ascii")
                        if isinstance(rle["counts"], bytes) else rle["counts"]))
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidates", type=Path, required=True,
                        help="exp091 train_candidates.json, or the validation pool")
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--images", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--fold", choices=("train", "val"), required=True)
    parser.add_argument("--val-fraction", type=float, default=0.15)
    parser.add_argument("--sub-threshold-only", action="store_true",
                        help="keep only proposals below the 0.35 floor, which is the bucket")
    args = parser.parse_args()
    cv2.setNumThreads(2)
    args.output.mkdir(parents=True, exist_ok=True)

    coco = json.loads(args.annotations.read_text(encoding="utf-8"))
    meta = {record["id"]: record for record in coco["images"]}
    by_image = defaultdict(list)
    for annotation in coco["annotations"]:
        by_image[annotation["image_id"]].append(annotation)

    split = make_split(str(args.annotations), val_fraction=args.val_fraction)
    assert_disjoint(split)
    image_ids = split.train_image_ids if args.fold == "train" else split.val_image_ids
    print(split.summary(), flush=True)

    pool = json.loads(args.candidates.read_text())
    pool = {Path(name).name: entries for name, entries in pool.items()}

    per_photograph = defaultdict(list)
    for image_id in image_ids:
        per_photograph[meta[image_id]["file_name"]].append(image_id)

    crops, masks, labels, index = [], [], [], []
    photographs = kept = skipped = 0
    for photograph, records in sorted(per_photograph.items()):
        entries = pool.get(photograph)
        if not entries:
            skipped += 1
            continue
        image = cv2.imread(str(args.images / photograph), cv2.IMREAD_GRAYSCALE)
        if image is None:
            skipped += 1
            continue
        photographs += 1
        selected = [(float(e["score"]), e["rle"]) for e in entries
                    if not args.sub_threshold_only or float(e["score"]) < BASE_CONF]
        if not selected:
            continue
        rles = [rle for _, rle in selected]
        decoded = [mu.decode(rle) for rle in rles]
        for image_id in records:
            truths = truth_rles(by_image.get(image_id, []))
            if not truths:
                continue
            ious = np.asarray(mu.iou(rles, truths, [0] * len(truths)))
            best = ious.max(axis=1)
            for position, (score, _) in enumerate(selected):
                patch = crop_of(image, decoded[position])
                if patch is None:
                    continue
                crops.append(patch[0])
                masks.append(patch[1])
                labels.append(int(best[position] >= MATCH))
                # image_id is a STRING in this release, shaped
                # "<annotator/batch>-<photograph stem>", e.g. "030101-20110109104734Ch".
                # An int() cast here crashed the first run; the id is an identifier,
                # not a number, and nothing downstream needs it to be one.
                index.append(dict(photograph=photograph, image_id=str(image_id),
                                  proposal=position, score=round(score, 6),
                                  best_iou=round(float(best[position]), 4)))
                kept += 1
        del decoded

    crops = np.asarray(crops, dtype=np.uint8)
    masks = np.asarray(masks, dtype=np.uint8)
    labels = np.asarray(labels, dtype=np.uint8)
    np.savez_compressed(args.output / f"{args.fold}_crops.npz",
                        image=crops, mask=masks, label=labels)
    (args.output / f"{args.fold}_index.json").write_text(json.dumps(index))

    positive = int(labels.sum())
    per_photo = defaultdict(int)
    for row, label in zip(index, labels):
        per_photo[row["photograph"]] += int(label)
    counts = np.array(sorted(per_photo.values()))
    summary = dict(fold=args.fold, photographs=photographs, skipped=skipped,
                   examples=int(len(labels)), positive=positive,
                   positive_rate=round(positive / max(len(labels), 1), 4),
                   sub_threshold_only=args.sub_threshold_only,
                   positives_per_photograph=dict(
                       zero=int((counts == 0).sum()), median=int(np.median(counts)) if counts.size else 0,
                       p90=int(np.percentile(counts, 90)) if counts.size else 0,
                       max=int(counts.max()) if counts.size else 0))
    (args.output / f"{args.fold}_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    print(f"\ncrops {crops.shape} masks {masks.shape} -> {args.output}")


if __name__ == "__main__":
    main()
