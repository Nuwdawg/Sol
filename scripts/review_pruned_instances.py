"""Native-resolution review of instances a pruning rule would remove.

Validation has annotations, so each removal is labelled by what it actually
cost: a removal that matched an annotator record is a real filament lost. The
selection is deterministic and deliberately adversarial. It shows the largest,
median and smallest removals, and the removals with the highest ground-truth
IoU, which are the most expensive mistakes the rule makes.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from pycocotools import mask as mu
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from experiments.exp_033_inference.src.sweep import records_for


def crop_box(mask, margin=120):
    ys, xs = np.nonzero(mask)
    side = int(min(2048, max(360, max(xs.max() - xs.min(), ys.max() - ys.min()) + margin)))
    x0 = int(np.clip((int(xs.max()) + int(xs.min()) - side) // 2, 0, 2048 - side))
    y0 = int(np.clip((int(ys.max()) + int(ys.min()) - side) // 2, 0, 2048 - side))
    return x0, y0, side


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--removed", type=Path, required=True)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--images", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    cv2.setNumThreads(1)
    features = json.loads(args.features.read_text())
    removed = json.loads(args.removed.read_text())
    records, _ = records_for(args.annotations)
    args.output.mkdir(parents=True, exist_ok=True)

    by_area = sorted(removed, key=lambda r: -r["area"])
    by_iou = sorted(removed, key=lambda r: -r["best_iou"])
    picks, seen = [], set()
    for label, row in ([("largest", by_area[0]), ("median", by_area[len(by_area) // 2]),
                        ("smallest", by_area[-1])]
                       + [(f"costliest_{i+1}", by_iou[i]) for i in range(3)]
                       + [("cleanest", by_iou[-1])]):
        key = (row["stem"], row["index"])
        if key in seen:
            continue
        seen.add(key)
        picks.append((label, row))

    fig, axes = plt.subplots(len(picks), 3, figsize=(13, 4.3 * len(picks)), squeeze=False)
    manifest = []
    for (label, row), axs in zip(picks, axes):
        stem = row["stem"]
        mask = mu.decode(features[stem][row["index"]]["rle"])
        x0, y0, side = crop_box(mask)
        image = cv2.imread(str(args.images / stem), cv2.IMREAD_GRAYSCALE)
        assert image is not None and image.shape == (2048, 2048), stem
        crop = image[y0:y0 + side, x0:x0 + side]
        lo, hi = np.percentile(crop, (1, 99))
        truth = np.zeros((2048, 2048), np.uint8)
        for _, rles in records[stem]:
            for rle in rles:
                truth |= mu.decode(rle)
        panels = [("Observation", None),
                  (f"Removed mask (area {row['area']})", mask),
                  (f"All annotator masks (IoU {row['best_iou']:.2f})", truth)]
        for ax, (title, overlay) in zip(axs, panels):
            ax.imshow(crop, cmap="gray", vmin=lo, vmax=hi, interpolation="nearest")
            if overlay is not None:
                ax.contour(overlay[y0:y0 + side, x0:x0 + side], levels=[.5],
                           colors=["#ff3b30" if overlay is mask else "#34c759"], linewidths=1.2)
            ax.set_title(f"{label}: {title}", fontsize=9)
            ax.axis("off")
        manifest.append(dict(label=label, **row, crop=[x0, y0, side]))
    fig.tight_layout()
    fig.savefig(args.output / "removed_instances.png", dpi=110)
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps(dict(panels=len(picks), output=str(args.output)), indent=2), flush=True)


if __name__ == "__main__":
    main()
