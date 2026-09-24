"""exp122: admit candidates as single instances, not as fragments.

exp121 found the deployable verifier earns +0.000480 against a +0.020822 ceiling,
and the instance counts explain why. Admitting 35 candidates created 55 instances:
painting gives every pixel to the highest-scoring mask covering it, so a
sub-threshold candidate lying partly under an existing mask is carved into pieces,
and each piece is scored as its own instance. Measured multipliers were 1.49 to
1.63 across rules and 1.63 for exp118's oracle itself.

That turns an effective precision of 20/35 = 0.571 into 20/55 = 0.364, which sits
just above the 0.350 break-even and therefore earns almost nothing.

The repair is mechanical and reads no truth: for each admitted candidate, in
descending score, subtract the pixels already claimed, keep only the largest
surviving connected component, and admit that single piece. A candidate then
contributes one instance or none.

Three admission geometries are compared under the same gates:

- `whole`: the carved remainder admitted as one instance, disconnected or not, so
  the multiplier is 1.0 by construction.
- `largest`: only the biggest surviving component is admitted.
- `all_pieces`: every surviving component above the area floor is admitted. This is
  the control, because it reproduces exp121's fragmenting behaviour.

This deviates from the frozen emission path by construction, which is the point,
so every variant is rescored by `aggregate_pq` after global rematching and the
`admit nothing` row must still reproduce the baseline exactly.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from pycocotools import mask as mu

from experiments.exp_033_inference.src.sweep import encode, pq_rle, records_for
from shared.utils import aggregate_pq, paint_panoptic

KERNEL = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
BASELINE_PQ = 0.4558638687292126
BASE_CONF = 0.35
MATCH = 0.5
CEILING = 0.020822


def prepare(entries, grow, min_area):
    prepared = []
    for entry in entries:
        mask = mu.decode(entry["rle"])
        if grow < 0:
            mask = cv2.erode(mask, KERNEL, iterations=-grow)
        if int(mask.sum()) < min_area:
            continue
        prepared.append((float(entry["score"]), mask))
    return prepared


def score_all(per_record):
    rows = [pq_rle(predicted, truths) for predicted, truths in per_record]
    aggregate = aggregate_pq(rows)
    aggregate["iou_sum"] = float(sum(row["iou_sum"] for row in rows))
    return aggregate


def per_truth_best(predicted, truths):
    if not predicted or not truths:
        return np.zeros(len(truths))
    return np.asarray(mu.iou(predicted, truths, [0] * len(truths))).max(axis=0)


def admit(base_masks, candidates, mode, min_area):
    """Add candidates beside an existing emission without re-painting it.

    `base_masks` are the already-emitted instances and are never modified, which is
    what keeps crossings-down at zero. Candidates are processed in descending score
    so a higher-scoring candidate claims contested pixels first.
    """
    claimed = np.zeros(base_masks[0].shape, np.uint8) if base_masks else None
    for mask in base_masks:
        claimed = mask.copy() if claimed is None else (claimed | mask)
    added = []
    for _, mask in sorted(candidates, key=lambda pair: -pair[0]):
        if claimed is None:
            claimed = np.zeros(mask.shape, np.uint8)
        remainder = ((mask > 0) & (claimed == 0)).astype(np.uint8)
        if int(remainder.sum()) < min_area:
            continue
        if mode == "whole":
            pieces = [remainder]
        else:
            count, labels = cv2.connectedComponents(remainder, connectivity=8)
            pieces = [(labels == index).astype(np.uint8) for index in range(1, count)]
            pieces = [piece for piece in pieces if int(piece.sum()) >= min_area]
            if mode == "largest" and pieces:
                pieces = [max(pieces, key=lambda piece: int(piece.sum()))]
        for piece in pieces:
            added.append(piece)
            claimed |= piece
    return added


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pool", type=Path,
                        default=Path("data/exp033_cached_tta/candidates_0.5_0.5.json"))
    parser.add_argument("--features", type=Path,
                        default=Path("data/exp120_verifier/candidates.json"))
    parser.add_argument("--output", type=Path, default=Path("data/exp122_fragment_free"))
    parser.add_argument("--grow", type=int, default=-1)
    parser.add_argument("--min-area", type=int, default=300)
    args = parser.parse_args()
    cv2.setNumThreads(2)
    args.output.mkdir(parents=True, exist_ok=True)
    annotations = next(Path(".").rglob("MAGFiLO_1.0_Annotations_kaggle2026_train.json"))
    records, _ = records_for(annotations)
    pool = json.loads(args.pool.read_text())
    scored = json.loads(args.features.read_text())
    index = {(row["photograph"], round(row["score"], 9)): row for row in scored}

    values = {name: np.array([r[name] for r in scored], float) for name in ("score", "limb")}
    cuts = {name: {q: float(np.quantile(values[name], q)) for q in
                   (0.50, 0.70, 0.80, 0.90, 0.95)} for name in values}

    def pair(quantile_score, quantile_limb):
        score_cut, limb_cut = cuts["score"][quantile_score], cuts["limb"][quantile_limb]
        return lambda row: row["score"] >= score_cut and row["limb"] <= limb_cut

    gates = [
        ("admit nothing", lambda row: False),
        ("score q0.90 + limb q0.70", pair(0.90, 0.70)),
        ("score q0.80 + limb q0.70", pair(0.80, 0.70)),
        ("score q0.80 + limb q0.50", pair(0.80, 0.50)),
        ("score q0.70 + limb q0.50", pair(0.70, 0.50)),
        ("score q0.95 alone", lambda row, c=cuts["score"][0.95]: row["score"] >= c),
        ("score q0.90 alone", lambda row, c=cuts["score"][0.90]: row["score"] >= c),
    ]

    prepared = {name: prepare(pool[name], args.grow, args.min_area) for name in records}
    base_rle, base_masks, base_records = {}, {}, []
    for name, passes in records.items():
        emitted = [encode(mask) for _, mask, _ in paint_panoptic(
            [(s, m) for s, m in prepared[name] if s >= BASE_CONF], min_area=args.min_area)]
        base_rle[name] = emitted
        base_masks[name] = [mu.decode(rle) for rle in emitted]
        for _, truths in passes:
            base_records.append((emitted, truths))
    baseline = score_all(base_records)
    assert abs(baseline["pq"] - BASELINE_PQ) < 1e-9, baseline["pq"]
    print(f"baseline PQ {baseline['pq']:.10f}  TP/FP/FN "
          f"{baseline['tp']}/{baseline['fp']}/{baseline['fn']}")
    print(f"ceiling +{CEILING:.6f} (exp118 oracle admit)\n")

    print(f"{'gate':28} {'mode':11} {'gated':>6} {'inst':>5} {'mult':>5} "
          f"{'PQ':>10} {'delta':>10} {'up':>4} {'down':>5} {'eff prec':>9}")
    report = dict(experiment_id="exp122", baseline_pq=baseline["pq"], ceiling=CEILING, rows=[])
    for label, gate in gates:
        for mode in ("whole", "largest", "all_pieces"):
            gated = instances = 0
            per_record = []
            for name, passes in records.items():
                chosen = []
                for s, m in prepared[name]:
                    if s >= BASE_CONF:
                        continue
                    row = index.get((name, round(s, 9)))
                    if row is not None and gate(row):
                        chosen.append((s, m))
                gated += len(chosen)
                extra = admit(base_masks[name], chosen, mode, args.min_area) if chosen else []
                instances += len(extra)
                emitted = base_rle[name] + [encode(piece) for piece in extra]
                for _, truths in passes:
                    per_record.append((emitted, truths))
            result = score_all(per_record)
            up = down = 0
            for (before, truths), (after, _) in zip(base_records, per_record):
                was, now = per_truth_best(before, truths), per_truth_best(after, truths)
                up += int(((was < MATCH) & (now >= MATCH)).sum())
                down += int(((was >= MATCH) & (now < MATCH)).sum())
            delta = result["pq"] - baseline["pq"]
            multiplier = instances / gated if gated else 0.0
            precision = up / instances if instances else 0.0
            print(f"{label:28} {mode:11} {gated:>6} {instances:>5} {multiplier:>5.2f} "
                  f"{result['pq']:>10.6f} {delta:>+10.6f} {up:>4} {down:>5} {precision:>9.3f}")
            report["rows"].append(dict(gate=label, mode=mode, gated=gated,
                                       instances=instances, multiplier=round(multiplier, 3),
                                       pq=result["pq"], delta_pq=round(delta, 6),
                                       crossings_up=up, crossings_down=down,
                                       effective_precision=round(precision, 4)))
            if label == "admit nothing" and mode == "whole":
                assert abs(result["pq"] - BASELINE_PQ) < 1e-9, result["pq"]

    best = max(report["rows"], key=lambda row: row["delta_pq"])
    (args.output / "report.json").write_text(json.dumps(report, indent=2))
    print(f"\nbest: {best['gate']} / {best['mode']} at {best['delta_pq']:+.6f} "
          f"({100 * best['delta_pq'] / CEILING:.1f}% of ceiling)")
    print("exp121's best was +0.000480. A large gain here means fragmentation was the cost.")


if __name__ == "__main__":
    main()
