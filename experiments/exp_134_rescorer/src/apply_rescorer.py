"""Admit sub-threshold proposals by the rescorer's judgement, and score it officially.

The rescorer answers one question per crop: is there a real filament under this
proposal? This turns that ordering into an emission and measures it with the frozen
metric, because average precision is a surrogate and PQ is not.

Admission rules swept
---------------------
- `global`: the top K proposals across the whole split.
- `per_photograph`: at most N per photograph, above a probability floor.

The second exists because the positives are sparse and concentrated: 23 of 106
validation photographs contain **no** admissible sub-threshold proposal at all,
the median is 1 and the 90th percentile is 4. A global top-K therefore spends most
of its budget on photographs that contain nothing to find, which is the same
imbalance that made the hand-feature verifier worthless in exp118-127.

Mechanics kept identical to the measured baseline
-------------------------------------------------
An admitted proposal is eroded 1 px, carved against pixels already claimed by the
base emission, and dropped if what survives is under the 300 px floor - exactly the
emission the 0.4558638687 baseline uses. Without the carve an admitted duplicate
would be scored as a fresh false positive rather than dissolving, which is how
exp122's "fragmentation" phantom arose.

Reported in the pre-registered order: proposal recall, buckets, native IoU, then PQ.
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import torch
from pycocotools import mask as mu

from experiments.exp_025_verifier.src.model import Verifier
from experiments.exp_033_inference.src.sweep import encode, pq_rle, records_for
from shared.utils import aggregate_pq, paint_panoptic

KERNEL = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
BASELINE_PQ = 0.4558638687292126
BASE_CONF = 0.35
MIN_AREA = 300
SHAPE = (2048, 2048)


def score_crops(checkpoint: Path, crops: Path, batch: int = 256) -> np.ndarray:
    payload = np.load(crops)
    images, masks = payload["image"], payload["mask"]
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    model = Verifier(width=state.get("width", 24))
    model.load_state_dict(state["state"])
    model.eval()
    print(f"checkpoint epoch {state.get('epoch')} held AP {state.get('held_ap'):.4f}")
    out = []
    with torch.no_grad():
        for start in range(0, len(images), batch):
            image = images[start:start + batch].astype(np.float32) / 255.0
            mask = masks[start:start + batch].astype(np.float32)
            if mask.max() > 1.0:
                mask = mask / 255.0
            stacked = torch.from_numpy(np.stack([image, mask], axis=1))
            out.append(torch.sigmoid(model(stacked)).numpy())
    return np.concatenate(out)


def score_all(per_record):
    rows = [pq_rle(predicted, truths) for predicted, truths in per_record]
    aggregate = aggregate_pq(rows)
    aggregate["iou_sum"] = float(sum(row["iou_sum"] for row in rows))
    return aggregate


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--crops", type=Path, required=True)
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--pool", type=Path,
                        default=Path("data/exp033_cached_tta/candidates_0.5_0.5.json"))
    parser.add_argument("--annotations", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    cv2.setNumThreads(2)
    args.output.mkdir(parents=True, exist_ok=True)

    index = json.loads(args.index.read_text())
    probability = score_crops(args.checkpoint, args.crops)
    labels = np.load(args.crops)["label"].astype(int)
    if len(index) != len(probability):
        raise SystemExit(f"index {len(index)} rows, scores {len(probability)}")
    print(f"scored {len(probability)} crops, {labels.sum()} positive "
          f"({labels.mean():.1%})")

    # One score per (photograph, proposal): the crop is identical across records of
    # the same photograph, so collapse by taking the max over records.
    best: dict[tuple[str, int], float] = {}
    for row, value in zip(index, probability):
        key = (row["photograph"], row["proposal"])
        best[key] = max(best.get(key, 0.0), float(value))
    print(f"distinct (photograph, proposal) pairs: {len(best)}")

    records, split = records_for(args.annotations)
    pool = json.loads(args.pool.read_text())

    base_rle, base_claimed, base_records, sub_index = {}, {}, [], {}
    for photograph, passes in records.items():
        entries = pool[photograph]
        kept, sub = [], []
        for position, entry in enumerate(entries):
            mask = cv2.erode(mu.decode(entry["rle"]), KERNEL, iterations=1)
            if float(entry["score"]) >= BASE_CONF:
                if int(mask.sum()) >= MIN_AREA:
                    kept.append((float(entry["score"]), mask))
            else:
                sub.append((position, float(entry["score"]), mask))
        emitted = [encode(m) for _, m, _ in paint_panoptic(kept, min_area=MIN_AREA)]
        base_rle[photograph] = emitted
        claimed = np.zeros(SHAPE, np.uint8)
        for rle in emitted:
            claimed |= mu.decode(dict(size=rle["size"],
                                      counts=rle["counts"].encode("ascii")))
        base_claimed[photograph] = claimed
        sub_index[photograph] = sub
        for _, truths in passes:
            base_records.append((emitted, truths))
    baseline = score_all(base_records)
    assert abs(baseline["pq"] - BASELINE_PQ) < 1e-9, baseline["pq"]
    print(f"baseline PQ {baseline['pq']:.10f}  TP/FP/FN "
          f"{baseline['tp']}/{baseline['fp']}/{baseline['fn']}\n")

    ranked = sorted(best.items(), key=lambda kv: -kv[1])
    rules = [("admit nothing", dict(kind="none"))]
    for k in (25, 50, 75, 100, 150, 200):
        rules.append((f"global top {k}", dict(kind="global", k=k)))
    for cap in (1, 2, 3):
        for floor in (0.5, 0.7, 0.9):
            rules.append((f"per photograph <={cap} at p>={floor}",
                          dict(kind="per_photograph", cap=cap, floor=floor)))

    print(f"{'rule':34} {'admitted':>9} {'PQ':>11} {'delta':>10} {'TP':>5} {'FP':>5} {'prec':>6}")
    report = dict(experiment_id="exp134", baseline_pq=baseline["pq"],
                  crops=len(probability), rules=[])
    for label, spec in rules:
        chosen: dict[str, list[int]] = defaultdict(list)
        if spec["kind"] == "global":
            for (photograph, position), _ in ranked[:spec["k"]]:
                chosen[photograph].append(position)
        elif spec["kind"] == "per_photograph":
            grouped = defaultdict(list)
            for (photograph, position), value in ranked:
                if value >= spec["floor"]:
                    grouped[photograph].append(position)
            for photograph, positions in grouped.items():
                chosen[photograph] = positions[:spec["cap"]]

        admitted, per_record = 0, []
        for photograph, passes in records.items():
            extra = []
            if chosen.get(photograph):
                claimed = base_claimed[photograph].copy()
                wanted = set(chosen[photograph])
                # Carve in RESCORER order, not detector order. When two admitted
                # proposals overlap, whichever is carved first keeps the contested
                # pixels, so the model's confidence must decide that - ordering by
                # the detector score would let the thing we trust less win.
                for position, _, mask in sorted(
                        sub_index[photograph],
                        key=lambda row: -best.get((photograph, row[0]), 0.0)):
                    if position not in wanted:
                        continue
                    remainder = ((mask > 0) & (claimed == 0)).astype(np.uint8)
                    if int(remainder.sum()) < MIN_AREA:
                        continue
                    extra.append(encode(remainder))
                    claimed |= remainder
            emitted = base_rle[photograph] + extra
            admitted += len(extra) * len(passes)
            for _, truths in passes:
                per_record.append((emitted, truths))
        result = score_all(per_record)
        delta = result["pq"] - baseline["pq"]
        gained = result["tp"] - baseline["tp"]
        precision = gained / admitted if admitted else 0.0
        print(f"{label:34} {admitted:>9} {result['pq']:>11.6f} {delta:>+10.6f} "
              f"{result['tp']:>5} {result['fp']:>5} {precision:>6.3f}")
        report["rules"].append(dict(label=label, admitted_records=admitted,
                                    pq=result["pq"], delta_pq=round(delta, 6),
                                    tp=result["tp"], fp=result["fp"],
                                    crossings_up=gained, precision=round(precision, 4)))

    best_rule = max(report["rules"], key=lambda row: row["delta_pq"])
    # Measured ceiling, not the bucket size. exp132 put the confidence-floor bucket
    # at 172, but only 168 are reachable by a sub-threshold proposal, and all 168
    # clear IoU 0.5 on the RAW mask the crops carry - so the classifier is not
    # handicapped by the representation. exp076 independently found the same 168 on
    # 2026-09-15 by a different code path.
    report["ceiling"] = dict(reachable_truths=168, bucket=172,
                             pq_if_all_recovered=0.5038,
                             note="168 at mean IoU 0.62 -> TP 1028 / FP 397 / FN 297")
    (args.output / "apply_report.json").write_text(json.dumps(report, indent=2))
    print(f"\nbest: {best_rule['label']} at {best_rule['delta_pq']:+.6f} "
          f"-> PQ {best_rule['pq']:.6f}")
    print(f"ceiling if all 168 reachable truths were recovered: PQ 0.5038")
    print(f"share of ceiling achieved: "
          f"{100 * best_rule['delta_pq'] / (0.5038 - BASELINE_PQ):.1f}%")
    print(f"public needs validation 0.472 for 0.40 and about 0.51 for 0.44.")


if __name__ == "__main__":
    main()
