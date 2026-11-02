"""exp200z: who gets the contested pixels when exp169's vote masks overlap, judged per record.

exp169 paints its kept instances (score >= 0.68, area >= 100) in descending score: an instance
keeps only the pixels no higher-scoring instance claimed. The vote masks of neighbouring
candidates overlap, so the order decides boundaries between touching filaments and which small
remnants fall under the area floor. Orders tried (no constant fitted): descending score (exp169),
ascending score, ascending area (small instances keep their pixels), descending area.

    python -m experiments.exp_200_test_truth.src.paint_order --annotations ... --ingredients val_ingredients.pkl
        --output paint_order.json
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

EXP169 = 0.48894859995182627


def paint(entries: list[tuple[float, "E.Crop"]], key, min_area: int = 100) -> list[dict]:
    claimed = np.zeros((E.SIZE, E.SIZE), np.uint8)
    out = []
    for score, crop in sorted(entries, key=key):
        if crop.mask.size == 0:
            continue
        h, w = crop.mask.shape
        window = claimed[crop.y0:crop.y0 + h, crop.x0:crop.x0 + w]
        remaining = crop.mask & (1 - window)
        if int(remaining.sum()) < min_area:
            continue
        window |= remaining
        full = np.zeros((E.SIZE, E.SIZE), np.uint8)
        full[crop.y0:crop.y0 + h, crop.x0:crop.x0 + w] = remaining
        out.append(mu.encode(np.asfortranarray(full)))
    return out


ORDERS = {
    "score_desc": lambda e: -e[0],
    "score_asc": lambda e: e[0],
    "area_asc": lambda e: int(e[1].mask.sum()),
    "area_desc": lambda e: -int(e[1].mask.sum()),
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--ingredients", type=Path, required=True, help="tune.py's exp169 ingredients")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    records, _ = records_for(args.annotations)
    names = sorted(records)
    passes = {n: [[as_bytes(t) for t in truth] for _, truth in sorted(records[n])] for n in names}
    rng = np.random.default_rng(2026)
    order = rng.permutation(len(names))
    folds = [set(names[i] for i in order[:len(names) // 2]), set(names[i] for i in order[len(names) // 2:])]
    prep = pickle.loads(args.ingredients.read_bytes())
    kept = {n: [(float(s), c) for s, c in zip(scores(prep[n], 4.0, 0.3), prep[n]["crops"]) if s >= 0.68] for n in names}
    rows = []
    for name, key in ORDERS.items():
        by = {n: [match(paint(kept[n], key), t) for t in passes[n]] for n in names}
        whole = total([r for n in names for r in by[n]])
        halves = [total([r for n in f for r in by[n]])["pq"] for f in folds]
        if name == "score_desc" and abs(whole["pq"] - EXP169) > 1e-9:
            raise SystemExit(f"exp169 not reproduced: {whole['pq']}")
        rows.append(dict(order=name, pq=whole["pq"], sq=whole["sq"], tp=whole["tp"], fp=whole["fp"], fn=whole["fn"],
                         halves=halves))
        print(json.dumps(dict(order=name, pq=round(whole["pq"], 5), sq=round(whole["sq"], 4), tp=whole["tp"],
                              fp=whole["fp"], fn=whole["fn"], halves=[round(h, 5) for h in halves])), flush=True)
    args.output.write_text(json.dumps(rows, indent=1))


if __name__ == "__main__":
    main()
