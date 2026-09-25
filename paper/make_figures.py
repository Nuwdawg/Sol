"""Draw every data figure of the technical report and write its numbers as LaTeX macros.

Inputs are measurements, never hand-typed values:

- host_eval.json and validation_emitted.json from exp145 (the host's self-evaluation
  notebook run on exp139 epoch 40 over the canonical validation split);
- annotator_agreement.json (annotator-vs-annotator PQ under the host's matching rule);
- data/submissions.csv (our public-leaderboard history, read with the Kaggle CLI);
- the competition annotations and training photographs, for the overlays.

Colours follow the dataviz reference palette in light mode, validated with its
validate_palette.js: blue #2a78d6, orange #eb6834, aqua #1baf7a on white pass every
check; aqua sits below 3:1, so each series also carries a direct label or legend. On the
grey solar disk no hue reaches 3:1, so overlay lines are drawn over a white halo.

    python paper/make_figures.py --annotations data/MAGFiLO_1.0_Annotations_kaggle2026_train.json \
        --images data/competition/unpacked/MAGFiLO_1.0_Kaggle_2026/train/train_images
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.patheffects as effects
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle
from pycocotools import mask as mu

from experiments.exp_033_inference.src.sweep import records_for

HERE = Path(__file__).resolve().parent
NATIVE = 2048
INK, INK2, MUTED, GRID, AXIS = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
BLUE_LIGHT = "#9ec5f4"          # sequential blue step 200: the "below threshold" bars
COLUMN, TEXT = 3.5, 7.16         # IEEE two-column widths, inches
HALO = [effects.withStroke(linewidth=2.6, foreground="white")]

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 7, "axes.titlesize": 7.5, "axes.labelsize": 7,
    "xtick.labelsize": 6.5, "ytick.labelsize": 6.5, "legend.fontsize": 6.5,
    "axes.edgecolor": AXIS, "axes.linewidth": 0.6, "axes.labelcolor": INK2,
    "xtick.color": MUTED, "ytick.color": MUTED, "xtick.labelcolor": INK2, "ytick.labelcolor": INK2,
    "text.color": INK, "axes.titlecolor": INK, "axes.grid": True, "axes.grid.axis": "y",
    "grid.color": GRID, "grid.linewidth": 0.5, "axes.axisbelow": True,
    "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False,
    "pdf.fonttype": 42, "savefig.bbox": "tight", "savefig.pad_inches": 0.02, "figure.dpi": 150,
})


def decode(counts: str) -> np.ndarray:
    return mu.decode({"size": [NATIVE, NATIVE], "counts": counts})


def photo_pq(truth: list, predicted: list) -> float:
    """The host's rule on one record: IoU > 0.5 pairs are TP, unmatched are FP / FN."""
    if not truth or not predicted:
        return 0.0
    iou = np.asarray(mu.iou(predicted, truth, [0] * len(truth))).T
    hit = iou > 0.5
    fp, fn = int((hit.sum(axis=0) == 0).sum()), int((hit.sum(axis=1) == 0).sum())
    return float(iou[hit].sum()) / (hit.sum() + 0.5 * fp + 0.5 * fn)


def contours(binary: np.ndarray, scale: float = 1.0, offset=(0, 0)) -> list[np.ndarray]:
    found, _ = cv2.findContours(binary.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    return [(c[:, 0, :].astype(float) - np.array(offset)) * scale for c in found if len(c) > 2]


def stretch(gray: np.ndarray) -> np.ndarray:
    """Display-only contrast stretch over the solar disk (1st-99.5th percentile)."""
    disk = gray[gray > 20]
    low, high = np.percentile(disk, [1, 99.5]) if disk.size else (0, 255)
    return np.clip((gray.astype(float) - low) / max(1.0, high - low), 0, 1)


def draw_outline(axis, polygons, colour, width, halo=1.9):
    for polygon in polygons:
        closed = np.vstack([polygon, polygon[:1]])
        axis.plot(closed[:, 0], closed[:, 1], color=colour, linewidth=width, solid_joinstyle="round",
                  path_effects=[effects.withStroke(linewidth=width + halo, foreground="white")])


def draw_fill(axis, polygons, colour, alpha=0.6):
    for polygon in polygons:
        axis.fill(polygon[:, 0], polygon[:, 1], color=colour, alpha=alpha, linewidth=0)


def ring(mask: np.ndarray, pixels: int) -> np.ndarray:
    """The annotator's mask grown by `pixels`, so its outline runs outside the filament
    and never covers the pixels (or the prediction) it marks."""
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * pixels + 1, 2 * pixels + 1))
    return cv2.dilate(mask.astype(np.uint8), kernel)


def figure_qualitative(records: dict, emitted: dict, images: Path, out: Path) -> dict:
    """A median photograph (by per-record PQ), not a best case: full disk plus three zooms."""
    candidates = []
    for name, annotators in records.items():
        if len(annotators) != 1 or len(annotators[0][1]) < 8:
            continue
        truth = [dict(r, counts=r["counts"].decode() if isinstance(r["counts"], bytes) else r["counts"])
                 for r in annotators[0][1]]
        predicted = [{"size": [NATIVE, NATIVE], "counts": c} for c in emitted[name]]
        candidates.append((photo_pq(truth, predicted), name, truth))
    median = float(np.median([c[0] for c in candidates]))
    score, name, truth = min(candidates, key=lambda c: abs(c[0] - median))
    gray = cv2.imread(str(images / name), cv2.IMREAD_GRAYSCALE)
    shown = stretch(gray)
    truth_masks = [mu.decode(r) for r in truth]
    predicted_masks = [decode(c) for c in emitted[name]]

    # Three zoom windows on the largest annotated filaments, without overlap.
    size, windows = 560, []
    for index in np.argsort([-m.sum() for m in truth_masks]):
        ys, xs = np.nonzero(truth_masks[index])
        cx, cy = int(xs.mean()), int(ys.mean())
        x0 = int(np.clip(cx - size // 2, 0, NATIVE - size))
        y0 = int(np.clip(cy - size // 2, 0, NATIVE - size))
        if all(abs(x0 - a) >= size or abs(y0 - b) >= size for a, b in windows):
            windows.append((x0, y0))
        if len(windows) == 3:
            break

    figure, axes = plt.subplots(2, 2, figsize=(COLUMN, COLUMN + 0.28))
    figure.subplots_adjust(wspace=0.04, hspace=0.1, left=0, right=1, top=1, bottom=0.07)
    scale = 512 / NATIVE
    disk = axes[0, 0]
    disk.imshow(cv2.resize(shown, (512, 512), interpolation=cv2.INTER_AREA), cmap="gray", vmin=0, vmax=1)
    for m in truth_masks:
        draw_outline(disk, contours(ring(m, 16), scale), BLUE, 0.55, halo=1.2)
    for m in predicted_masks:
        draw_fill(disk, contours(ring(m, 3), scale), ORANGE, alpha=0.9)
    labels = "bcd"
    for letter, (x0, y0) in zip(labels, windows):
        disk.add_patch(Rectangle((x0 * scale, y0 * scale), size * scale, size * scale,
                                 fill=False, edgecolor="white", linewidth=0.9))
        disk.text(x0 * scale + 4, y0 * scale + 4, letter, color="white", fontsize=7, fontweight="bold",
                  va="top", ha="left", path_effects=[effects.withStroke(linewidth=1.6, foreground=INK)])
    disk.text(6, 506, "(a)", color="white", fontsize=7, fontweight="bold", va="bottom",
              path_effects=[effects.withStroke(linewidth=1.6, foreground=INK)])
    for axis, letter, (x0, y0) in zip([axes[0, 1], axes[1, 0], axes[1, 1]], labels, windows):
        axis.imshow(shown[y0:y0 + size, x0:x0 + size], cmap="gray", vmin=0, vmax=1)
        for m in truth_masks:
            draw_outline(axis, contours(ring(m, 6)[y0:y0 + size, x0:x0 + size]), BLUE, 0.8, halo=1.4)
        for m in predicted_masks:
            draw_fill(axis, contours(m[y0:y0 + size, x0:x0 + size]), ORANGE, alpha=0.5)
        axis.text(8, size - 10, f"({letter})", color="white", fontsize=7, fontweight="bold", va="bottom",
                  path_effects=[effects.withStroke(linewidth=1.6, foreground=INK)])
        axis.set_xlim(0, size); axis.set_ylim(size, 0)
    for axis in axes.ravel():
        axis.set_axis_off()
    figure.legend(handles=[Line2D([], [], color=BLUE, linewidth=1.4, label="annotator (ground truth)"),
                           Patch(facecolor=ORANGE, alpha=0.5, edgecolor=ORANGE, label="prediction")],
                  loc="lower center", ncol=2, bbox_to_anchor=(0.5, -0.005), handlelength=1.6)
    figure.savefig(out / "fig_qualitative.pdf", dpi=400)
    plt.close(figure)

    # Pipeline thumbnails (Fig. 1): the same photograph before and after, 300 px.
    thumb = (cv2.resize(shown, (300, 300), interpolation=cv2.INTER_AREA) * 255).astype(np.uint8)
    cv2.imwrite(str(out / "thumb_input.png"), thumb)
    painted = cv2.cvtColor(thumb, cv2.COLOR_GRAY2BGR).astype(float)
    orange_bgr = np.array([0x34, 0x68, 0xEB], float)
    for m in predicted_masks:
        # Grown by 10 native pixels first, so thin instances survive the 7x reduction.
        small = cv2.resize(ring(m, 10), (300, 300), interpolation=cv2.INTER_AREA) > 0
        painted[small] = 0.2 * painted[small] + 0.8 * orange_bgr
    cv2.imwrite(str(out / "thumb_output.png"), painted.astype(np.uint8))
    return dict(name=name, pq=score, median=median, candidates=len(candidates),
                truth=len(truth_masks), predicted=len(predicted_masks))


def figure_failures(records: dict, emitted: dict, images: Path, out: Path) -> dict:
    """The three error kinds, each picked by a fixed rule, never by eye.

    Over single-annotator validation records, with "touch" meaning IoU > 0 as in the
    notebook's relation plot: (a) the median-area annotated filament that no prediction
    touches, (b) the median-area prediction that touches no annotated filament, and
    (c) the largest annotated filament touched by two or more predictions.
    """
    misses, stray, splits = [], [], []
    for name, annotators in records.items():
        if len(annotators) != 1:
            continue
        truth = annotators[0][1]
        predicted = [{"size": [NATIVE, NATIVE], "counts": c} for c in emitted[name]]
        if not truth:
            continue
        areas_gt = mu.area(truth)
        if not predicted:
            misses.extend((int(a), name, i) for i, a in enumerate(areas_gt))
            continue
        touch = np.asarray(mu.iou(predicted, truth, [0] * len(truth))).T > 0
        areas_pred = mu.area(predicted)
        for i, a in enumerate(areas_gt):
            if touch[i].sum() == 0:
                misses.append((int(a), name, i))
            elif touch[i].sum() >= 2:
                splits.append((int(a), name, i))
        for j, a in enumerate(areas_pred):
            if touch[:, j].sum() == 0:
                stray.append((int(a), name, j))
    median = lambda rows: sorted(rows)[len(rows) // 2]
    picks = [("(a) missed filament", median(misses), "gt"),
             ("(b) false positive", median(stray), "pred"),
             ("(c) split filament", max(splits), "gt")]
    size = 420
    figure, axes = plt.subplots(1, 3, figsize=(COLUMN, COLUMN / 3 + 0.3))
    figure.subplots_adjust(wspace=0.04, left=0, right=1, top=0.86, bottom=0.0)
    chosen = {}
    for axis, (title, (area, name, index), kind) in zip(axes, picks):
        truth = records[name][0][1]
        truth_masks = [mu.decode(r) for r in truth]
        predicted_masks = [decode(c) for c in emitted[name]]
        target = truth_masks[index] if kind == "gt" else predicted_masks[index]
        ys, xs = np.nonzero(target)
        x0 = int(np.clip(xs.mean() - size // 2, 0, NATIVE - size))
        y0 = int(np.clip(ys.mean() - size // 2, 0, NATIVE - size))
        shown = stretch(cv2.imread(str(images / name), cv2.IMREAD_GRAYSCALE))
        axis.imshow(shown[y0:y0 + size, x0:x0 + size], cmap="gray", vmin=0, vmax=1)
        for m in truth_masks:
            draw_outline(axis, contours(ring(m, 6)[y0:y0 + size, x0:x0 + size]), BLUE, 0.8, halo=1.4)
        for m in predicted_masks:
            draw_fill(axis, contours(m[y0:y0 + size, x0:x0 + size]), ORANGE, alpha=0.5)
        axis.set_xlim(0, size); axis.set_ylim(size, 0)
        axis.set_axis_off()
        axis.set_title(title, loc="left", fontsize=6.8, pad=2)
        chosen[title] = dict(photograph=name, area=area)
    figure.savefig(out / "fig_failures.pdf", dpi=400)
    plt.close(figure)
    return dict(chosen=chosen, misses=len(misses), false_positives=len(stray), splits=len(splits))


def histogram(axis, values, threshold, label):
    values = np.asarray(values)
    edges = np.linspace(0, 1, 41)
    counts, _ = np.histogram(values, bins=edges)
    share = counts / counts.sum()
    centres = (edges[:-1] + edges[1:]) / 2
    colours = [BLUE if c > threshold else BLUE_LIGHT for c in centres] if threshold else BLUE
    axis.bar(centres, share, width=(edges[1] - edges[0]) * 0.84, color=colours, linewidth=0)
    mean = values.mean()
    axis.axvline(mean, color=INK, linewidth=0.8, linestyle=(0, (3, 2)))
    handles = [Line2D([], [], color=INK, linewidth=0.8, linestyle=(0, (3, 2)), label=f"mean {mean:.3f}")]
    if threshold:
        handles = [Patch(color=BLUE, label=f"IoU > {threshold}: true positive"),
                   Patch(color=BLUE_LIGHT, label=f"IoU ≤ {threshold}: no match")] + handles
    axis.legend(handles=handles, loc="upper left", handlelength=1.2, borderaxespad=0.1)
    axis.set_xlim(0, 1)
    axis.set_ylim(0, share.max() * 1.25)
    axis.set_xlabel(label)
    axis.set_ylabel("share of pairs")


def figure_host_metrics(host: dict, out: Path):
    figure, axes = plt.subplots(1, 3, figsize=(TEXT, 1.8), gridspec_kw=dict(width_ratios=[1, 1, 1.2]))
    figure.subplots_adjust(wspace=0.3)
    histogram(axes[0], host["iou_overlapping"], 0.5, "IoU of overlapping pairs")
    axes[0].set_title("(a) IoU distribution", loc="left")
    histogram(axes[1], host["dice_overlapping"], None, "Dice of overlapping pairs")
    axes[1].set_title("(b) Dice distribution", loc="left")

    axis = axes[2]
    gt = {int(k): v for k, v in host["gt_degree"].items()}
    pred = {int(k): v for k, v in host["pred_degree"].items()}
    top = max(max(gt), max(pred), 3)
    fold = lambda d: [d.get(k, 0) for k in range(3)] + [sum(v for k, v in d.items() if k >= 3)]
    gt_counts, pred_counts = fold(gt), fold(pred)
    names = ["0", "1", "2", "3+"]
    positions_gt = np.arange(4) + 0.6
    positions_pred = -(np.arange(4) + 0.6)
    for x, value, k in zip(positions_gt, gt_counts, range(4)):
        axis.bar(x, value, width=0.78, color=BLUE, hatch="////" if k == 0 else None,
                 edgecolor="white" if k == 0 else BLUE, linewidth=0)
    for x, value, k in zip(positions_pred, pred_counts, range(4)):
        axis.bar(x, value, width=0.78, color=ORANGE, hatch="////" if k == 0 else None,
                 edgecolor="white" if k == 0 else ORANGE, linewidth=0)
    peak = max(gt_counts + pred_counts)
    for x, value, k in ([(x, v, k) for k, (x, v) in enumerate(zip(positions_gt, gt_counts))] +
                        [(x, v, k) for k, (x, v) in enumerate(zip(positions_pred, pred_counts))]):
        axis.text(x, value + peak * 0.02, str(value), ha="center", va="bottom", fontsize=6, color=INK2)
    # The two hatched bars sit side by side at the centre: one shared label, not two.
    axis.text(0, max(gt_counts[0], pred_counts[0]) + peak * 0.12, "hatched:\nno overlap", ha="center",
              va="bottom", fontsize=6, color=INK2, linespacing=0.95, zorder=4,
              bbox=dict(facecolor="white", edgecolor="none", pad=0.6))
    axis.axvline(0, color=AXIS, linewidth=0.8)
    axis.set_xticks(list(positions_pred) + list(positions_gt))
    axis.set_xticklabels([f"{n}:1" for n in names] + [f"1:{n}" for n in names])
    axis.set_ylim(0, peak * 1.25)
    axis.set_ylabel("count")
    axis.set_xlabel("relation")
    axis.text(-2.2, peak * 1.13, "each prediction", color=INK2, fontsize=6.5, ha="center")
    axis.text(2.2, peak * 1.13, "each GT filament", color=INK2, fontsize=6.5, ha="center")
    axis.set_title("(c) many-to-one / one-to-many", loc="left")
    del top
    figure.savefig(out / "fig_host_metrics.pdf")
    plt.close(figure)


def figure_curve(host: dict, human: float, out: Path) -> dict:
    rows = [r for r in host["epoch_curve"] if r["fixed"]]
    epochs = [r["epoch"] for r in rows]
    figure, axis = plt.subplots(figsize=(COLUMN, 1.75))
    series = [("PQ", "pq", BLUE), ("SQ", "sq", ORANGE), ("RQ", "rq", AQUA)]
    finals = {key: rows[-1]["fixed"][key] for _, key, _ in series}
    # SQ and RQ end within 0.01 of each other: push the higher label up and the lower down.
    nudge = {"sq": 0.011 if finals["sq"] >= finals["rq"] else -0.011, "rq": -0.011 if finals["sq"] >= finals["rq"] else 0.011, "pq": 0.0}
    for label, key, colour in series:
        values = [r["fixed"][key] for r in rows]
        axis.plot(epochs, values, color=colour, linewidth=1.3, marker="o", markersize=3.2,
                  markeredgecolor="white", markeredgewidth=0.6, label=label)
        axis.text(epochs[-1] + 1.2, values[-1] + nudge[key], label, color=INK, fontsize=6.5, va="center")
    axis.axhline(human, color=MUTED, linewidth=0.8, linestyle=(0, (4, 2)))
    axis.text(epochs[0], human + 0.008, f"annotator vs annotator PQ ({human:.3f})", color=INK2, fontsize=6.3)
    chosen = next(r for r in rows if r["epoch"] == 40)
    axis.axvline(40, color=AXIS, linewidth=0.8)
    axis.text(40.6, 0.395, "deployed\nepoch 40", fontsize=6.2, color=INK2, va="bottom")
    axis.set_xlim(epochs[0] - 1, epochs[-1] + 4.5)
    axis.set_xlabel("epoch")
    axis.set_ylabel("validation score")
    axis.legend(loc="lower left", ncol=3, bbox_to_anchor=(0.0, 1.0), handlelength=1.4, borderaxespad=0.2)
    figure.savefig(out / "fig_curve.pdf")
    plt.close(figure)
    best = max(rows, key=lambda r: r["fixed"]["pq"])
    return dict(epoch40=chosen["fixed"], best_epoch=best["epoch"], best=best["fixed"],
                epoch50=next(r for r in rows if r["epoch"] == 50)["fixed"])


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval for a binomial rate: stays inside [0, 1] and behaves at small n."""
    if n == 0:
        return float("nan"), float("nan")
    p = k / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return centre - half, centre + half


def figure_size(host: dict, out: Path) -> dict:
    gt = host["gt_items"]
    pred = host["pred_items"]
    areas = np.array([g["area"] for g in gt])
    edges = np.unique(np.quantile(areas, np.linspace(0, 1, 7)))
    edges[0], edges[-1] = 0, np.inf
    def rate(items):
        a = np.array([i["area"] for i in items]); m = np.array([i["matched"] for i in items])
        rows = []
        for lo, hi in zip(edges[:-1], edges[1:]):
            inside = (a >= lo) & (a < hi)
            k, n = int(m[inside].sum()), int(inside.sum())
            rows.append((k / n if n else np.nan, *wilson(k, n), n))
        return np.array(rows)
    recall, precision = rate(gt), rate(pred)
    def kilo(value: float) -> str:
        return f"{value / 1000:.1f}k"
    labels = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        labels.append(f"<{kilo(hi)}" if lo == 0 else (f">{kilo(lo)}" if np.isinf(hi) else f"{kilo(lo)}-{kilo(hi)}"))
    x = np.arange(len(labels))
    figure, axis = plt.subplots(figsize=(COLUMN, 1.7))
    for rows, colour, label, shift in ((recall, BLUE, "recall (GT found)", -0.07),
                                       (precision, ORANGE, "precision (prediction correct)", 0.07)):
        axis.errorbar(x + shift, rows[:, 0], yerr=[rows[:, 0] - rows[:, 1], rows[:, 2] - rows[:, 0]],
                      color=colour, linewidth=1.3, marker="o", markersize=3.4, markeredgecolor="white",
                      markeredgewidth=0.6, elinewidth=0.8, capsize=1.8, label=label)
    axis.set_xticks(x)
    axis.set_xticklabels(labels)
    axis.set_ylim(0, 1)
    axis.set_xlabel("filament area, pixels at 2048 x 2048 (sextiles of GT area)")
    axis.set_ylabel("rate at IoU > 0.5")
    axis.legend(loc="lower right")
    figure.savefig(out / "fig_size.pdf")
    plt.close(figure)
    return dict(recall_smallest=recall[0, 0], recall_largest=recall[-1, 0],
                recall_smallest_ci=(recall[0, 1], recall[0, 2]), recall_largest_ci=(recall[-1, 1], recall[-1, 2]),
                precision_smallest=precision[0, 0], precision_largest=precision[-1, 0],
                smallest_edge=float(edges[1]), largest_edge=float(edges[-2]), per_bin=int(recall[0, 3]))


def record_sums(truth: list, predicted: list) -> np.ndarray:
    """(sum of TP IoU, TP, FP, FN) for one record under the host's rule."""
    if not truth:
        return np.array([0.0, 0, len(predicted), 0])
    if not predicted:
        return np.array([0.0, 0, 0, len(truth)])
    iou = np.asarray(mu.iou(predicted, truth, [0] * len(truth))).T
    hit = iou > 0.5
    return np.array([float(iou[hit].sum()), int(hit.sum()), int((hit.sum(axis=0) == 0).sum()),
                     int((hit.sum(axis=1) == 0).sum())])


def pooled(sums: np.ndarray) -> tuple:
    iou, tp, fp, fn = sums[..., 0], sums[..., 1], sums[..., 2], sums[..., 3]
    denominator = tp + 0.5 * fp + 0.5 * fn
    return iou / denominator, iou / np.maximum(tp, 1), tp / denominator


def bootstrap(per_photo: np.ndarray, draws: int = 10000, seed: int = 2026) -> dict:
    """Percentile 95% intervals, resampling PHOTOGRAPHS: a photograph's annotation records
    share one prediction set, so they are not independent and must move together."""
    rng = np.random.default_rng(seed)
    picks = rng.integers(0, len(per_photo), size=(draws, len(per_photo)))
    pq, sq, rq = pooled(per_photo[picks].sum(axis=1))
    point = pooled(per_photo.sum(axis=0))
    return {name: dict(point=float(p), low=float(np.percentile(v, 2.5)), high=float(np.percentile(v, 97.5)))
            for name, p, v in (("pq", point[0], pq), ("sq", point[1], sq), ("rq", point[2], rq))}


def uncertainty(records: dict, emitted: dict, host_pq: float) -> dict:
    model, human = [], []
    for name in sorted(records):
        predicted = [{"size": [NATIVE, NATIVE], "counts": c} for c in emitted[name]]
        model.append(sum(record_sums(rles, predicted) for _, rles in records[name]))
        annotators = [rles for _, rles in records[name]]
        if len(annotators) >= 2:
            human.append(sum(record_sums(truth, list(other)) for a, other in enumerate(annotators)
                             for b, truth in enumerate(annotators) if a != b))
    model_ci = bootstrap(np.array(model))
    # The same numbers the host notebook produced, or the intervals describe something else.
    assert abs(model_ci["pq"]["point"] - host_pq) < 1e-9, (model_ci["pq"]["point"], host_pq)
    return dict(model=model_ci, human=bootstrap(np.array(human)), photographs=len(model),
                human_photographs=len(human))


def figure_progress(path: Path, out: Path) -> dict:
    rows = list(csv.DictReader(path.open(encoding="utf-8")))
    colours = {"YOLO-seg": BLUE, "Mask2Former": ORANGE, "Semantic U-Net": AQUA, "Other": MUTED}
    figure, axis = plt.subplots(figsize=(COLUMN, 1.75))
    index = np.array([int(r["index"]) for r in rows])
    score = np.array([float(r["public_pq"]) for r in rows])
    best = np.maximum.accumulate(score)
    axis.step(index, best, where="post", color=INK2, linewidth=0.8, label="best so far")
    for family, colour in colours.items():
        pick = [i for i, r in enumerate(rows) if r["family"] == family]
        probe = [i for i in pick if rows[i]["probe"] == "1"]
        normal = [i for i in pick if rows[i]["probe"] != "1"]
        axis.scatter(index[normal], score[normal], s=16, color=colour, edgecolor="white", linewidth=0.6,
                     zorder=3, label=family)
        if probe:
            axis.scatter(index[probe], score[probe], s=16, facecolor="white", edgecolor=colour, linewidth=0.9, zorder=3)
    axis.axhline(0.35, color=MUTED, linewidth=0.8, linestyle=(0, (4, 2)))
    axis.text(index[-1], 0.343, "host: PQ > 0.35 is \"of great value\"", fontsize=6.2, color=INK2,
              va="top", ha="right")
    top = int(np.argmax(score))
    axis.annotate(f"{score[top]:.2f}", (index[top], score[top]), xytext=(0, 5), textcoords="offset points",
                  ha="center", fontsize=6.5, color=INK)
    axis.set_ylim(0.18, 0.45)
    axis.set_xlabel("submission (chronological)")
    axis.set_ylabel("public PQ")
    handles, _ = axis.get_legend_handles_labels()
    handles.append(Line2D([], [], linestyle="none", marker="o", markersize=4, markerfacecolor="white",
                          markeredgecolor=MUTED, label="diagnostic probe"))
    # Every corner holds data, so the legend sits above the plot area.
    axis.legend(handles=handles, loc="lower left", bbox_to_anchor=(0.0, 1.0), ncol=3, handletextpad=0.2,
                columnspacing=0.9, fontsize=6, borderaxespad=0.2)
    figure.savefig(out / "fig_progress.pdf")
    plt.close(figure)
    return dict(submissions=len(rows), first=float(score[0]), best=float(score.max()),
                families=dict(Counter(r["family"] for r in rows)))


def macros(values: dict) -> str:
    lines = ["% Generated by paper/make_figures.py from measured JSON. Do not edit by hand."]
    for name, value in values.items():
        lines.append(f"\\newcommand{{\\{name}}}{{{value}}}")
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--images", type=Path, required=True)
    parser.add_argument("--host-eval", type=Path, default=HERE / "data" / "host_eval.json")
    parser.add_argument("--emitted", type=Path, default=HERE / "data" / "validation_emitted.json")
    parser.add_argument("--agreement", type=Path,
                        default=HERE.parent / "experiments/exp_145_host_evaluation/results/annotator_agreement.json")
    parser.add_argument("--submissions", type=Path, default=HERE / "data" / "submissions.csv")
    parser.add_argument("--exp141", type=Path, default=HERE / "data" / "exp141_soup_tta_report.json",
                        help="float32 single-view, soup and fusion validation results")
    parser.add_argument("--out", type=Path, default=HERE / "figures")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    host = json.loads(args.host_eval.read_text())
    emitted = json.loads(args.emitted.read_text())
    agreement = json.loads(args.agreement.read_text())
    exp141 = json.loads(args.exp141.read_text())
    singles, fusion = exp141["single_view"], exp141["conclusion"]["best_fusion"]["best"]
    single = exp141["conclusion"]["best_single"]
    records, _ = records_for(args.annotations)

    human_val = agreement["validation_photographs"]["pq"]
    qualitative = figure_qualitative(records, emitted, args.images, args.out)
    failures = figure_failures(records, emitted, args.images, args.out)
    figure_host_metrics(host, args.out)
    curve = figure_curve(host, human_val, args.out)
    size = figure_size(host, args.out)
    progress = figure_progress(args.submissions, args.out)
    spread = uncertainty(records, emitted, host["host_pq"])
    coco = json.loads(args.annotations.read_text())
    all_photos = {image["file_name"] for image in coco["images"]}

    gt, pred = ({int(k): v for k, v in host[key].items()} for key in ("gt_degree", "pred_degree"))
    total_gt, total_pred = sum(gt.values()), sum(pred.values())
    timing = host["timing"]
    values = dict(
        HostPQ=f"{host['host_pq']:.4f}", OurPQ=f"{host['our_pq']:.4f}",
        NumTP=host["tp"], NumFP=host["fp"], NumFN=host["fn"],
        ValSQ=f"{host['sq']:.3f}", ValRQ=f"{host['rq']:.3f}",
        MeanIoUOverlap=f"{host['mean_iou_overlapping']:.3f}", MeanDiceOverlap=f"{host['mean_dice_overlapping']:.3f}",
        ValPhotos=host["split"]["photographs"], ValRecords=host["split"]["records"],
        ValGT=host["split"]["gt_filaments"], ValPred=host["split"]["predicted_filaments"],
        GTMissedPct=f"{100 * gt.get(0, 0) / total_gt:.1f}", GTOneOnePct=f"{100 * gt.get(1, 0) / total_gt:.1f}",
        GTSplitPct=f"{100 * sum(v for k, v in gt.items() if k >= 2) / total_gt:.1f}",
        PredUnmatchedPct=f"{100 * pred.get(0, 0) / total_pred:.1f}", PredOneOnePct=f"{100 * pred.get(1, 0) / total_pred:.1f}",
        PredMergePct=f"{100 * sum(v for k, v in pred.items() if k >= 2) / total_pred:.1f}",
        HumanPQ=f"{human_val:.3f}", HumanPQAll=f"{agreement['all_photographs']['pq']:.3f}",
        HumanPairs=agreement["all_photographs"]["pairs"], HumanValPairs=agreement["validation_photographs"]["pairs"],
        HumanSQ=f"{agreement['validation_photographs']['sq']:.3f}", HumanRQ=f"{agreement['validation_photographs']['rq']:.3f}",
        SecondsPerPhoto=f"{timing['seconds_per_photograph_median']:.2f}", InferenceGPU=timing["device"].replace("Tesla ", ""),
        TestMinutes=f"{180 * timing['seconds_per_photograph_mean'] / 60:.0f}",
        PeakGiB=f"{timing['peak_gpu_gib']:.1f}", LoadSeconds=f"{timing['model_load_seconds']:.1f}",
        EpochFortyPQ=f"{curve['epoch40']['pq']:.4f}", EpochFiftyPQ=f"{curve['epoch50']['pq']:.4f}",
        BestEpoch=curve["best_epoch"],
        RecallSmall=f"{size['recall_smallest']:.2f}", RecallLarge=f"{size['recall_largest']:.2f}",
        PrecisionSmall=f"{size['precision_smallest']:.2f}", PrecisionLarge=f"{size['precision_largest']:.2f}",
        SmallArea=f"{size['smallest_edge']:.0f}", LargeArea=f"{size['largest_edge']:.0f}",
        QualPQ=f"{qualitative['pq']:.3f}", QualMedian=f"{qualitative['median']:.3f}",
        QualCandidates=qualitative["candidates"],
        Submissions=progress["submissions"], FirstPublic=f"{progress['first']:.2f}", BestPublic=f"{progress['best']:.2f}",
        FloatFortyPQ=f"{singles['e040']['pq']:.4f}", FloatFiftyPQ=f"{singles['e050']['pq']:.4f}",
        SoupPQ=f"{singles['soup']['pq']:.4f}", FusionPQ=f"{fusion['pq']:.4f}",
        FusionExtraTP=fusion["tp"] - single["tp"], FusionExtraFP=fusion["fp"] - single["fp"],
        FailMissArea=failures["chosen"]["(a) missed filament"]["area"],
        FailFPArea=failures["chosen"]["(b) false positive"]["area"],
        FailSplitArea=failures["chosen"]["(c) split filament"]["area"],
        PQLow=f"{spread['model']['pq']['low']:.3f}", PQHigh=f"{spread['model']['pq']['high']:.3f}",
        SQLow=f"{spread['model']['sq']['low']:.3f}", SQHigh=f"{spread['model']['sq']['high']:.3f}",
        RQLow=f"{spread['model']['rq']['low']:.3f}", RQHigh=f"{spread['model']['rq']['high']:.3f}",
        HumanPQLow=f"{spread['human']['pq']['low']:.3f}", HumanPQHigh=f"{spread['human']['pq']['high']:.3f}",
        HumanPhotos=spread["human_photographs"],
        RecallSmallLow=f"{size['recall_smallest_ci'][0]:.2f}", RecallSmallHigh=f"{size['recall_smallest_ci'][1]:.2f}",
        RecallLargeLow=f"{size['recall_largest_ci'][0]:.2f}", RecallLargeHigh=f"{size['recall_largest_ci'][1]:.2f}",
        PerBin=size["per_bin"],
        AllPhotos=len(all_photos), AllRecords=len(coco["images"]), AllFilaments=len(coco["annotations"]),
        TrainPhotos=len(all_photos) - host["split"]["photographs"],
        TrainRecords=len(coco["images"]) - host["split"]["records"],
        TrainGT=len(coco["annotations"]) - host["split"]["gt_filaments"],
    )
    generated = HERE / "generated"
    generated.mkdir(exist_ok=True)
    (generated / "numbers.tex").write_text(macros(values), encoding="utf-8")
    print(json.dumps(dict(qualitative=qualitative, failures=failures, curve=curve, size=size, progress=progress,
                          uncertainty=spread), indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
