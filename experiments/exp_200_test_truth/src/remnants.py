"""exp200ab: two more learning-free signals on exp169's instances, judged per record.

remnants: painting gives an instance only the pixels higher-ranked instances left. One that keeps
  under a fraction f of its own vote mask is mostly a duplicate of a neighbour; exp169 drops it only
  below 100 px. Drop it when remaining / own area < f (f 0.3, 0.5, 0.7).
layers:   every exp139 full-view candidate stores its class probability after each of the decoder's
  10 layers. Replace the final-layer probability in the score by the mean of the last k layers
  (k 3, 5, 10): a filament the decoder believed throughout outranks one that jumped at the end. The
  score is then rebuilt as exp169's: (p_k x m)^0.3 x mean^0.7, threshold chosen by exp168's folds.
Both run with exp169's painting (descending score) and with small instances first.

    python -m experiments.exp_200_test_truth.src.remnants --annotations ... --cache exp147_val_cache.json
        --ingredients val_ingredients.pkl --output remnants.json
"""
from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import numpy as np
from pycocotools import mask as mu

from experiments.exp_033_inference.src.sweep import records_for
from experiments.exp_149_evidence_fusion.src import evidence as E
from experiments.exp_168_joint_tuning.src.joint import scores
from experiments.exp_200_test_truth.src.hypotheses import as_bytes, match, total
from experiments.exp_200_test_truth.src.paint_order import ORDERS

THRESHOLDS = tuple(round(t, 3) for t in np.arange(0.60, 0.761, 0.01))


def paint_fraction(entries, key, fraction: float, min_area: int = 100) -> list[dict]:
    claimed = np.zeros((E.SIZE, E.SIZE), np.uint8)
    out = []
    for score, crop in sorted(entries, key=key):
        if crop.mask.size == 0:
            continue
        h, w = crop.mask.shape
        window = claimed[crop.y0:crop.y0 + h, crop.x0:crop.x0 + w]
        remaining = crop.mask & (1 - window)
        kept = int(remaining.sum())
        if kept < min_area or kept < fraction * int(crop.mask.sum()):
            continue
        window |= remaining
        full = np.zeros((E.SIZE, E.SIZE), np.uint8)
        full[crop.y0:crop.y0 + h, crop.x0:crop.x0 + w] = remaining
        out.append(mu.encode(np.asfortranarray(full)))
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--ingredients", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    records, _ = records_for(args.annotations)
    names = sorted(records)
    passes = {n: [[as_bytes(t) for t in truth] for _, truth in sorted(records[n])] for n in names}
    rng = np.random.default_rng(2026)
    order = rng.permutation(len(names))
    folds = [set(names[i] for i in order[:len(names) // 2]), set(names[i] for i in order[len(names) // 2:])]
    prep = pickle.loads(args.ingredients.read_bytes())
    cache = E.load_cache(args.cache)

    def evaluate(entries_by, key, fraction):
        by = {n: [match(paint_fraction(entries_by[n], key, fraction), t) for t in passes[n]] for n in names}
        return (total([r for n in names for r in by[n]])["pq"], [total([r for n in f for r in by[n]])["pq"] for f in folds])

    rows = []
    base = {n: [(float(s), c) for s, c in zip(scores(prep[n], 4.0, 0.3), prep[n]["crops"]) if s >= 0.68] for n in names}
    for order_name in ("score_desc", "area_asc"):
        for fraction in (0.0, 0.3, 0.5, 0.7):
            whole, halves = evaluate(base, ORDERS[order_name], fraction)
            rows.append(dict(test="remnants", order=order_name, fraction=fraction, pq=whole, halves=halves))
            print(json.dumps(dict(test="remnants", order=order_name, fraction=fraction, pq=round(whole, 5),
                                  halves=[round(h, 5) for h in halves])), flush=True)
    for k in (3, 5, 10):
        for t in THRESHOLDS:
            entries_by = {}
            for n in names:
                p = prep[n]
                layer_p = np.array([np.mean(c["layers"][-k:]) for c in cache[n]["full"]])
                m = np.array([c["m"] for c in cache[n]["full"]])
                q = dict(p, s=layer_p * m)
                entries_by[n] = [(float(s), c) for s, c in zip(scores(q, 4.0, 0.3), p["crops"]) if s >= t]
            whole, halves = evaluate(entries_by, ORDERS["score_desc"], 0.0)
            rows.append(dict(test="layers", k=k, threshold=t, pq=whole, halves=halves))
        best = max((r for r in rows if r.get("test") == "layers" and r["k"] == k), key=lambda r: r["pq"])
        print(json.dumps(dict(test="layers", k=k, threshold=best["threshold"], pq=round(best["pq"], 5),
                              halves=[round(h, 5) for h in best["halves"]])), flush=True)
    args.output.write_text(json.dumps(rows, indent=1))


if __name__ == "__main__":
    main()


def paint_absorb(entries, key, fraction: float, min_area: int = 100) -> list[dict]:
    """Like paint_fraction, but a remnant's leftover pixels join the instance it overlaps most.

    A candidate that keeps less than `fraction` of its own vote mask is usually the full-length
    version of a filament whose other part a higher-ranked instance already holds. Dropping it
    truncates the filament; here its unclaimed pixels are added to that instance instead.
    """
    owner = np.zeros((E.SIZE, E.SIZE), np.int32)          # 0 = unclaimed, i + 1 = instance i
    masks = []
    for score, crop in sorted(entries, key=key):
        if crop.mask.size == 0:
            continue
        h, w = crop.mask.shape
        window = owner[crop.y0:crop.y0 + h, crop.x0:crop.x0 + w]
        free = crop.mask.astype(bool) & (window == 0)
        kept = int(free.sum())
        own = int(crop.mask.sum())
        if kept >= fraction * own and kept >= min_area:
            masks.append(None)
            window[free] = len(masks)
            continue
        overlapped = window[crop.mask.astype(bool) & (window > 0)]
        if fraction > 0 and overlapped.size and kept:
            window[free] = np.bincount(overlapped).argmax()
    out = []
    for i in range(1, int(owner.max()) + 1):
        m = (owner == i).astype(np.uint8)
        if int(m.sum()) >= min_area:
            out.append(mu.encode(np.asfortranarray(m)))
    return out
