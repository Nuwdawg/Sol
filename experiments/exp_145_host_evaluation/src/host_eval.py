"""Score exp139 epoch 40 on validation with the host's own self-evaluation notebook.

The organisers publish `azimahmadzadeh/self-evaluation-notebook`. It takes a ground-truth
frame and a prediction frame in the submission format and reports the Panoptic Quality,
the IoU and Dice distributions and the one-to-many / many-to-one relations: the
quantitative items of the competition rubric. Its code cells run here verbatim, in their
own namespace. This file only prepares the two frames and saves what the notebook
computes, so the report quotes the judges' own numbers, not a re-implementation of them.

Test labels are private, so the frames cover the canonical validation split (seed 2026,
split by photograph). Predictions use exp139's deployed rule: one view, float32,
confidence 0.85, no erosion, minimum area 100.

    python -m experiments.exp_145_host_evaluation.src.host_eval --checkpoint model_e040.pt \
        --annotations .../MAGFiLO_1.0_Annotations_kaggle2026_train.json \
        --train-images .../train_images --notebook self_evaluation_notebook.ipynb \
        --weights <exp139 weights folder> --output out/
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")

import numpy as np
import pandas as pd
import torch
from pycocotools import mask as mu

from experiments.exp_033_inference.src.sweep import evaluate_thresholds, records_for
from experiments.exp_139_mask2former.src.data import NATIVE, read_gray
from experiments.exp_139_mask2former.src.evaluate import event, load_checkpoint
from experiments.exp_139_mask2former.src.model import predict_photograph
from experiments.exp_139_mask2former.src.submit import emit

RULE = dict(conf=0.85, grow=0, min_area=100)


def counts_text(rle: dict) -> str:
    counts = rle["counts"]
    return counts.decode("ascii") if isinstance(counts, bytes) else counts


def host_namespace(notebook: Path) -> tuple[dict, str]:
    """Every code cell of the host notebook, executed in order, in a fresh namespace."""
    cells = json.loads(notebook.read_text(encoding="utf-8"))["cells"]
    namespace: dict = {}
    for cell in cells:
        if cell["cell_type"] == "code":
            exec(compile("".join(cell["source"]), str(notebook), "exec"), namespace)
    return namespace, hashlib.sha256(notebook.read_bytes()).hexdigest()


def predict_validation(model, records: dict, images: Path, device) -> tuple[dict, dict, list]:
    cache, emitted, seconds = {}, {}, []
    for position, name in enumerate(sorted(records), start=1):
        gray = read_gray(images / name)
        if device.type == "cuda":
            torch.cuda.synchronize()
        started = time.perf_counter()
        entries = predict_photograph(model, gray, device, False, d4=False, floor=0.02)
        kept = list(emit(entries, RULE["conf"], RULE["grow"], RULE["min_area"]))
        if device.type == "cuda":
            torch.cuda.synchronize()
        seconds.append(time.perf_counter() - started)
        claimed = np.zeros((NATIVE, NATIVE), np.uint8)
        for rle in kept:
            binary = mu.decode(rle)
            if not binary.sum() or (claimed & binary).any():
                raise SystemExit(f"bad instance for {name}")
            claimed |= binary
        cache[name], emitted[name] = entries, [counts_text(rle) for rle in kept]
        if position % 20 == 0:
            print(f"PREDICT {position}/{len(records)}", flush=True)
    return cache, emitted, seconds


def frames(records: dict, emitted: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Both frames in the submission format the notebook reads.

    Ground truth: `<annotator>-<photograph>_<k>`, so the notebook's split on `_` yields
    the annotator-image and its split on `-` yields the photograph. Predictions:
    `<photograph>_<k>`, exactly as submitted.
    """
    gt_rows, pred_rows = [], []
    for name in sorted(records):
        stem = Path(name).stem
        for index, counts in enumerate(emitted[name], start=1):
            pred_rows.append((f"{stem}_{index}", counts))
        for image_id, rles in records[name]:
            if str(image_id).split("-", 1)[1] != stem:
                raise SystemExit(f"annotation record {image_id} does not name photograph {stem}")
            for index, rle in enumerate(rles, start=1):
                gt_rows.append((f"{image_id}_{index}", counts_text(rle)))
    columns = ["filament_id", "segmentation_rle"]
    return pd.DataFrame(gt_rows, columns=columns), pd.DataFrame(pred_rows, columns=columns)


def notebook_measures(overlap_df: pd.DataFrame, gt_df: pd.DataFrame, pred_df: pd.DataFrame) -> dict:
    """The quantities the notebook plots, read from its own overlap table.

    TP/FP/FN follow `get_pq_score` (IoU > 0.5); the relation degrees follow
    `plot_m2n_counts` (any overlap, IoU > 0). Areas come from the same RLEs.
    """
    gt_ids = gt_df["filament_id"].str.split("_", n=1).str[0]
    pred_ids = pred_df["filament_id"].str.split("_", n=1).str[0]
    iou_all, dice_all, tp_iou = [], [], []
    gt_degree, pred_degree, gt_items, pred_items = [], [], [], []
    tp = fp = fn = 0
    for row in overlap_df.itertuples(index=False):
        iou = np.asarray(row.iou_matrix, dtype=float).reshape(row.n_gt, row.n_pred)
        dice = np.asarray(row.dice_matrix, dtype=float).reshape(row.n_gt, row.n_pred)
        photograph = row.annotator_image.split("-", 1)[1]
        gt_rles = gt_df.loc[gt_ids == row.annotator_image, "segmentation_rle"].tolist()
        pred_rles = pred_df.loc[pred_ids == photograph, "segmentation_rle"].tolist()
        size = [NATIVE, NATIVE]
        gt_area = [int(mu.area({"size": size, "counts": c})) for c in gt_rles]
        pred_area = [int(mu.area({"size": size, "counts": c})) for c in pred_rles]
        iou_all.extend(iou[iou > 0].tolist())
        dice_all.extend(dice[dice > 0].tolist())
        if row.n_gt == 0 or row.n_pred == 0:
            fp += row.n_pred if row.n_gt == 0 else 0
            fn += row.n_gt if row.n_pred == 0 else 0
            gt_degree.extend([0] * row.n_gt)
            pred_degree.extend([0] * row.n_pred)
            gt_items.extend(dict(area=a, best_iou=0.0, matched=False) for a in gt_area)
            pred_items.extend(dict(area=a, best_iou=0.0, matched=False) for a in pred_area)
            continue
        hit = iou > 0.5
        tp_iou.extend(iou[hit].tolist())
        tp += int(hit.sum())
        fp += int((hit.sum(axis=0) == 0).sum())
        fn += int((hit.sum(axis=1) == 0).sum())
        touch = iou > 0
        gt_degree.extend(touch.sum(axis=1).astype(int).tolist())
        pred_degree.extend(touch.sum(axis=0).astype(int).tolist())
        gt_items.extend(dict(area=a, best_iou=float(iou[i].max()), matched=bool(hit[i].any()))
                        for i, a in enumerate(gt_area))
        pred_items.extend(dict(area=a, best_iou=float(iou[:, j].max()), matched=bool(hit[:, j].any()))
                          for j, a in enumerate(pred_area))
    denominator = tp + 0.5 * fp + 0.5 * fn
    return dict(
        tp=tp, fp=fp, fn=fn,
        sq=float(np.mean(tp_iou)) if tp_iou else 0.0,
        rq=tp / denominator if denominator else 0.0,
        pq_recomputed=float(np.sum(tp_iou)) / denominator if denominator else 0.0,
        mean_iou_overlapping=float(np.mean(iou_all)) if iou_all else 0.0,
        mean_dice_overlapping=float(np.mean(dice_all)) if dice_all else 0.0,
        iou_overlapping=[round(v, 5) for v in iou_all],
        dice_overlapping=[round(v, 5) for v in dice_all],
        gt_degree={str(k): int(v) for k, v in zip(*np.unique(gt_degree, return_counts=True))},
        pred_degree={str(k): int(v) for k, v in zip(*np.unique(pred_degree, return_counts=True))},
        gt_items=gt_items, pred_items=pred_items)


def epoch_curve(weights: Path) -> list[dict]:
    """Validation PQ per saved epoch at the deployed rule, plus that epoch's best rule."""
    curve = []
    for path in sorted(weights.glob("e[0-9][0-9][0-9]_sweep.json")):
        report = json.loads(path.read_text())
        fixed = [row for row in report["rows"] if abs(row["conf"] - RULE["conf"]) < 1e-9
                 and row["grow"] == RULE["grow"] and row["min_area"] == RULE["min_area"]]
        best = report["best"]
        curve.append(dict(epoch=int(path.name[1:4]),
                          fixed={k: fixed[0][k] for k in ("pq", "sq", "rq", "tp", "fp", "fn")} if fixed else None,
                          best={k: best[k] for k in ("pq", "sq", "rq", "conf", "grow", "min_area")}))
    return curve


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--train-images", type=Path, required=True)
    parser.add_argument("--notebook", type=Path, required=True)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--msda", choices=("pytorch", "hub"), default="hub")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    log = args.output / "events.log"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.msda != "pytorch" and device.type == "cuda":
        from experiments.exp_139_mask2former.src.fast_ops import (install_deformable_kernel,
                                                                  verify_deformable_kernel)
        install_deformable_kernel(args.msda)
        if not all(row["passed"] for row in verify_deformable_kernel(device)):
            install_deformable_kernel("pytorch")
            event("KERNEL_FALLBACK", log, source="pytorch")

    records, _ = records_for(args.annotations)
    loaded = time.perf_counter()
    model, state = load_checkpoint(args.checkpoint, device)
    load_seconds = time.perf_counter() - loaded
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    cache, emitted, seconds = predict_validation(model, records, args.train_images, device)
    peak_gib = torch.cuda.max_memory_allocated() / 2**30 if device.type == "cuda" else 0.0

    ours = evaluate_thresholds(cache, records, [RULE["conf"]], grow=RULE["grow"], min_area=RULE["min_area"])[0]
    gt_df, pred_df = frames(records, emitted)
    gt_df.to_csv(args.output / "validation_ground_truth.csv", index=False)
    pred_df.to_csv(args.output / "validation_predictions.csv", index=False)

    namespace, notebook_sha = host_namespace(args.notebook)
    started = time.perf_counter()
    overlap_df = namespace["get_overlap_df"](gt_df, pred_df)
    host_pq = float(namespace["get_pq_score"](overlap_df))
    host_seconds = time.perf_counter() - started
    import matplotlib.pyplot as plt
    for column, title, label, name in (("iou_matrix", "IoU Distribution", "IoU", "host_iou.png"),
                                       ("dice_matrix", "Dice Distribution", "Dice", "host_dice.png")):
        plt.figure(figsize=(6, 4))
        namespace["plot_distribution"](overlap_df[column], title, label)
        plt.tight_layout(); plt.savefig(args.output / name, dpi=150); plt.close()
    plt.figure(figsize=(8, 4))
    namespace["plot_m2n_counts"](overlap_df, "Many to One and One to Many Distribution")
    plt.tight_layout(); plt.savefig(args.output / "host_m2n.png", dpi=150); plt.close()

    measures = notebook_measures(overlap_df, gt_df, pred_df)
    photographs = sorted(records)
    report = dict(
        checkpoint=args.checkpoint.name, epoch=state.get("epoch"), rule=RULE,
        notebook=dict(source="azimahmadzadeh/self-evaluation-notebook", sha256=notebook_sha,
                      seconds=round(host_seconds, 1)),
        split=dict(seed=2026, photographs=len(photographs), records=sum(len(v) for v in records.values()),
                   gt_filaments=len(gt_df), predicted_filaments=len(pred_df)),
        host_pq=host_pq, our_pq=ours["pq"],
        ours={k: ours[k] for k in ("pq", "sq", "rq", "tp", "fp", "fn")},
        timing=dict(device=torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu",
                    model_load_seconds=round(load_seconds, 2),
                    seconds_per_photograph_median=round(float(np.median(seconds[1:])), 3),
                    seconds_per_photograph_mean=round(float(np.mean(seconds[1:])), 3),
                    first_photograph_seconds=round(seconds[0], 3), peak_gpu_gib=round(peak_gib, 2)),
        per_photograph={name: dict(predicted=len(emitted[name]),
                                   annotators=[len(rles) for _, rles in records[name]])
                        for name in photographs},
        epoch_curve=epoch_curve(args.weights),
        **measures)
    (args.output / "host_eval.json").write_text(json.dumps(report))
    (args.output / "validation_emitted.json").write_text(json.dumps(emitted))
    summary = {k: report[k] for k in ("host_pq", "our_pq", "tp", "fp", "fn", "sq", "rq",
                                      "gt_degree", "pred_degree", "timing", "split")}
    print("HOST_EVAL " + json.dumps(summary), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
