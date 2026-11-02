"""exp200w: shape post-processing of a submission's masks, no fitting, judged per record.

The truth masks are single polygons rasterised by pycocotools (every annotation's `area` field
equals pycocotools' pixel count; no other rasteriser matches). The predicted masks are pixel sets
from upsampled logits and votes: stair-stepped boundaries, occasional holes, stray islands. Each
variant below maps one predicted mask to another with no constants learnt from labels:
  holes:   fill holes;
  islands: keep the largest connected component (8-connectivity);
  polygon: trace the outer boundary through boundary-pixel centres (cv2, no simplification) and
           rasterise it with pycocotools, as the truth was (sheds about half a boundary pixel);
  poly_e:  the same, simplified with approxPolyDP at epsilon e px.
Instances are re-painted in their original order, so the file stays non-overlapping; an instance
that shrinks below 100 px is dropped, as the emission rule does.

    python -m experiments.exp_200_test_truth.src.shape_post --annotations ... --predictions val_predictions.json
        --key exp169 --output shape_post.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from pycocotools import mask as mu
from scipy import ndimage

from experiments.exp_033_inference.src.sweep import records_for
from experiments.exp_200_test_truth.src.hypotheses import as_bytes, match, total

SIZE = 2048
MIN_AREA = 100


def polygon(mask: np.ndarray, epsilon: float) -> np.ndarray:
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    rings = []
    for c in contours:
        if epsilon > 0:
            c = cv2.approxPolyDP(c, epsilon, True)
        if len(c) >= 3:
            rings.append(c.reshape(-1).astype(float).tolist())
    if not rings:
        return np.zeros_like(mask)
    return mu.decode(mu.merge(mu.frPyObjects(rings, SIZE, SIZE)))


def transform(mask: np.ndarray, kind: str) -> np.ndarray:
    if kind == "holes":
        return ndimage.binary_fill_holes(mask).astype(np.uint8)
    if kind == "islands":
        labels, n = ndimage.label(mask, structure=np.ones((3, 3)))
        if n <= 1:
            return mask
        sizes = ndimage.sum(mask, labels, range(1, n + 1))
        return (labels == 1 + int(np.argmax(sizes))).astype(np.uint8)
    if kind.startswith("poly"):
        return polygon(mask, float(kind.split("_")[1]) if "_" in kind else 0.0)
    raise ValueError(kind)


def apply(preds: list[dict], kind: str | None) -> list[dict]:
    """Transform each instance in order, giving way to the pixels earlier instances claimed."""
    if kind is None:
        return preds
    claimed = np.zeros((SIZE, SIZE), np.uint8)
    out = []
    for rle in preds:
        mask = transform(mu.decode(rle), kind) & (1 - claimed)
        if int(mask.sum()) < MIN_AREA:
            continue
        claimed |= mask
        out.append(mu.encode(np.asfortranarray(mask)))
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True, help="hypotheses.py's validation predictions")
    parser.add_argument("--key", default="exp169")
    parser.add_argument("--kinds", nargs="+", default=["holes", "islands", "poly", "poly_0.5", "poly_1.0", "poly_1.5"])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    records, _ = records_for(args.annotations)
    names = sorted(records)
    passes = {n: [[as_bytes(t) for t in truth] for _, truth in sorted(records[n])] for n in names}
    rng = np.random.default_rng(2026)
    order = rng.permutation(len(names))
    folds = [set(names[i] for i in order[:len(names) // 2]), set(names[i] for i in order[len(names) // 2:])]
    preds = {n: [as_bytes(r) for r in json.loads(args.predictions.read_text())[args.key][n]] for n in names}
    rows = []
    for kind in [None] + args.kinds:
        rows_by = {n: [match(apply(preds[n], kind), t) for t in passes[n]] for n in names}
        whole = total([r for n in names for r in rows_by[n]])
        halves = [total([r for n in f for r in rows_by[n]])["pq"] for f in folds]
        rows.append(dict(kind=kind, pq=whole["pq"], sq=whole["sq"], rq=whole["rq"], tp=whole["tp"], fp=whole["fp"],
                         fn=whole["fn"], halves=halves))
        print(json.dumps(dict(kind=kind, pq=round(whole["pq"], 5), sq=round(whole["sq"], 4), tp=whole["tp"],
                              fp=whole["fp"], fn=whole["fn"], halves=[round(h, 5) for h in halves])), flush=True)
    args.output.write_text(json.dumps(rows, indent=1))


if __name__ == "__main__":
    main()
