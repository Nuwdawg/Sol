"""Cache (proposal, annotation record) crop pairs as memory-mapped arrays."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from experiments.exp_092_refiner.src.crops import build_examples, truths_by_photo
from shared.data_split import assert_disjoint, make_split


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--proposals", type=Path, required=True)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--images", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--split", choices=("train", "val"), default="train")
    parser.add_argument("--size", type=int, default=256)
    parser.add_argument("--context", type=float, default=1.8)
    parser.add_argument("--floor", type=float, default=0.2)
    args = parser.parse_args()
    cv2.setNumThreads(1)

    split = make_split(str(args.annotations))
    assert_disjoint(split)
    proposals = json.loads(args.proposals.read_text())
    wanted = set(split.train_stems if args.split == "train" else split.val_stems)
    names = sorted(name for name in proposals if name in wanted)
    assert names, f"no {args.split} photographs in {args.proposals}"
    truths = truths_by_photo(args.annotations, set(names))

    args.output.mkdir(parents=True, exist_ok=True)
    features_path = args.output / "features.npy"
    targets_path = args.output / "targets.npy"
    # Exact upper bound on pairs, so the cache can never silently truncate.
    capacity = sum(len(proposals[name]) * max(1, len(truths.get(name, [])))
                   for name in names)
    assert capacity, "no proposal/record pairs are possible"
    print(f"capacity {capacity} pairs over {len(names)} photographs", flush=True)
    features = np.lib.format.open_memmap(
        features_path, mode="w+", dtype=np.uint8,
        shape=(capacity, 4, args.size, args.size))
    targets = np.lib.format.open_memmap(
        targets_path, mode="w+", dtype=np.uint8,
        shape=(capacity, args.size, args.size))
    meta, written = [], 0
    for position, name in enumerate(names, start=1):
        image = cv2.imread(str(args.images / name), cv2.IMREAD_GRAYSCALE)
        assert image is not None, name
        examples = build_examples(image, proposals[name], truths.get(name, []),
                                  args.size, args.context, args.floor)
        if not examples:
            continue
        assert written + len(examples) <= capacity, (written, len(examples), capacity)
        for example in examples:
            features[written] = example["features"]
            targets[written] = example["target"]
            meta.append(dict(photo=name, seed_iou=example["seed_iou"],
                             score=example["score"], box=list(example["box"])))
            written += 1
        if position % 25 == 0 or position == len(names):
            print(f"prepare {position}/{len(names)} examples={written}", flush=True)
    assert written, "no examples were produced"
    features.flush(); targets.flush()
    del features, targets
    # Trim the memory maps to exactly what was written.
    for path, shape in ((features_path, (written, 4, args.size, args.size)),
                        (targets_path, (written, args.size, args.size))):
        data = np.load(path, mmap_mode="r")[:written]
        trimmed = np.lib.format.open_memmap(path.with_suffix(".tmp.npy"), mode="w+",
                                            dtype=np.uint8, shape=shape)
        trimmed[:] = data
        trimmed.flush(); del trimmed, data
        path.unlink(); path.with_suffix(".tmp.npy").rename(path)
    summary = dict(status="complete", split=args.split, photographs=len(names),
                   examples=written, size=args.size, context=args.context,
                   floor=args.floor,
                   mean_seed_iou=float(np.mean([row["seed_iou"] for row in meta])))
    (args.output / "meta.json").write_text(json.dumps(meta))
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
