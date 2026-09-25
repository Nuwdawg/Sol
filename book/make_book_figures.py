"""Draw the onboarding book's data figures and write its numbers as LaTeX macros.

Everything here runs on data already in the repository or on this machine:
the competition annotations and photographs under data/, exp145's host evaluation
(paper/data/host_eval.json), the emitted validation masks, the annotator-agreement
report and the submission history. Numbers quoted from past experiments (resolution
sweeps, erosion sweeps, oracle ceilings) are transcribed from the experiment log, and
each such array names the experiment it comes from.

Style and helpers are shared with paper/make_figures.py, so the book and the report
look alike.

    PYTHONPATH=. python book/make_book_figures.py \
        --annotations data/MAGFiLO_1.0_Annotations_kaggle2026_train.json \
        --images data/competition/unpacked/MAGFiLO_1.0_Kaggle_2026
"""
from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import re
import shutil
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np
from pycocotools import mask as mu

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
spec = importlib.util.spec_from_file_location("paperfig", ROOT / "paper" / "make_figures.py")
paperfig = importlib.util.module_from_spec(spec)
spec.loader.exec_module(paperfig)            # sets the shared rcParams
plt = paperfig.plt
from matplotlib.lines import Line2D           # noqa: E402
from matplotlib.patches import Patch          # noqa: E402

from experiments.exp_033_inference.src.sweep import records_for   # noqa: E402

BLUE, ORANGE, AQUA = paperfig.BLUE, paperfig.ORANGE, paperfig.AQUA
INK, INK2, MUTED, AXIS = paperfig.INK, paperfig.INK2, paperfig.MUTED, paperfig.AXIS
BLUE_LIGHT = paperfig.BLUE_LIGHT
WIDTH = 5.2                  # book text width, inches
STATIONS = {"Bh": "Big Bear", "Ch": "Cerro Tololo", "Lh": "Learmonth",
            "Mh": "Mauna Loa", "Th": "El Teide", "Uh": "Udaipur"}


def save(figure, name: str) -> None:
    figure.savefig(HERE / "figures" / name)
    plt.close(figure)


def polygon_rle(segmentation, height=2048, width=2048) -> dict:
    return mu.merge(mu.frPyObjects(segmentation, height, width))


# --------------------------------------------------------------------------- data
def fig_stations(images: Path, coco: dict) -> dict:
    """One photograph from each GONG station, display-stretched."""
    by_station = defaultdict(list)
    for image in coco["images"]:
        by_station[image["file_name"][-7:-5]].append(image["file_name"])
    figure, axes = plt.subplots(2, 3, figsize=(WIDTH, WIDTH * 2 / 3 + 0.2))
    figure.subplots_adjust(wspace=0.04, hspace=0.18, left=0, right=1, top=0.95, bottom=0)
    chosen = {}
    for axis, code in zip(axes.ravel(), sorted(STATIONS)):
        name = sorted(set(by_station[code]))[len(set(by_station[code])) // 2]
        gray = cv2.imread(str(images / "train" / "train_images" / name), cv2.IMREAD_GRAYSCALE)
        axis.imshow(cv2.resize(paperfig.stretch(gray), (600, 600), interpolation=cv2.INTER_AREA),
                    cmap="gray", vmin=0, vmax=1)
        axis.set_title(f"{STATIONS[code]} ({code})  {name[:4]}-{name[4:6]}-{name[6:8]}", fontsize=6.8)
        axis.set_axis_off()
        chosen[code] = name
    save(figure, "fig_stations.pdf")
    return chosen


def fig_annotators(images: Path, coco: dict) -> dict:
    """A photograph drawn by three annotators, full disk plus one zoom."""
    by_photo = defaultdict(list)
    for image in coco["images"]:
        by_photo[image["file_name"]].append(image["id"])
    annotations = defaultdict(list)
    for a in coco["annotations"]:
        annotations[a["image_id"]].append(a)
    triples = sorted(name for name, ids in by_photo.items() if len(ids) == 3)
    # Deterministic pick: the three-annotator photograph with the most filaments in total.
    name = max(triples, key=lambda n: sum(len(annotations[i]) for i in by_photo[n]))
    ids = sorted(by_photo[name])
    gray = cv2.imread(str(images / "train" / "train_images" / name), cv2.IMREAD_GRAYSCALE)
    shown = paperfig.stretch(gray)
    colours = [BLUE, ORANGE, AQUA]
    masks = {i: [mu.decode(polygon_rle(a["segmentation"])) for a in annotations[i]] for i in ids}
    # Zoom on the region with the most annotated pixels of all three annotators.
    heat = sum(m.astype(np.float32) for i in ids for m in masks[i])
    heat = cv2.blur(heat, (301, 301))
    cy, cx = np.unravel_index(int(np.argmax(heat)), heat.shape)
    size = 620
    x0, y0 = int(np.clip(cx - size // 2, 0, 2048 - size)), int(np.clip(cy - size // 2, 0, 2048 - size))
    figure, axes = plt.subplots(1, 2, figsize=(WIDTH, WIDTH / 2 + 0.35))
    figure.subplots_adjust(wspace=0.04, left=0, right=1, top=1, bottom=0.12)
    scale = 600 / 2048
    axes[0].imshow(cv2.resize(shown, (600, 600), interpolation=cv2.INTER_AREA), cmap="gray", vmin=0, vmax=1)
    axes[1].imshow(shown[y0:y0 + size, x0:x0 + size], cmap="gray", vmin=0, vmax=1)
    for colour, image_id in zip(colours, ids):
        for m in masks[image_id]:
            paperfig.draw_outline(axes[0], paperfig.contours(m, scale), colour, 0.6, halo=1.0)
            crop = m[y0:y0 + size, x0:x0 + size]
            if crop.any():
                paperfig.draw_outline(axes[1], paperfig.contours(crop), colour, 1.0, halo=1.4)
    axes[0].add_patch(plt.Rectangle((x0 * scale, y0 * scale), size * scale, size * scale,
                                    fill=False, edgecolor="white", linewidth=0.9))
    for axis in axes:
        axis.set_axis_off()
    axes[1].set_xlim(0, size); axes[1].set_ylim(size, 0)
    counts = [len(masks[i]) for i in ids]
    figure.legend(handles=[Line2D([], [], color=c, linewidth=1.6, label=f"annotator {k + 1}: {n} filaments")
                           for k, (c, n) in enumerate(zip(colours, counts))],
                  loc="lower center", ncol=3, bbox_to_anchor=(0.5, 0.0), fontsize=6.5)
    save(figure, "fig_annotators.pdf")
    return dict(name=name, counts=counts)


def fig_filament_stats(coco: dict) -> dict:
    """Filament sizes and crowding across all training annotations."""
    long_side = np.array([max(a["bbox"][2], a["bbox"][3]) for a in coco["annotations"]])
    area = np.array([a["area"] for a in coco["annotations"]])
    per_record = Counter(a["image_id"] for a in coco["annotations"])
    counts = np.array([per_record.get(i["id"], 0) for i in coco["images"]])
    figure, axes = plt.subplots(1, 3, figsize=(WIDTH, 1.75))
    figure.subplots_adjust(wspace=0.45)
    bins = np.logspace(np.log10(max(4, long_side.min())), np.log10(long_side.max()), 40)
    axes[0].hist(long_side, bins=bins, color=BLUE, linewidth=0)
    axes[0].set_xscale("log")
    for q, label, height in ((50, "p50", 0.96), (90, "p90", 0.74), (99, "p99", 0.52)):
        v = np.percentile(long_side, q)
        axes[0].axvline(v, color=INK, linewidth=0.7, linestyle=(0, (3, 2)))
        axes[0].text(v * 1.08, axes[0].get_ylim()[1] * height, f"{label} {v:.0f}", fontsize=5.8, va="top",
                     bbox=dict(facecolor="white", edgecolor="none", pad=0.4))
    axes[0].set_xlabel("long side of box (px)")
    axes[0].set_ylabel("filaments")
    axes[0].set_title("(a) length", loc="left")
    abins = np.logspace(np.log10(max(10, area.min())), np.log10(area.max()), 40)
    axes[1].hist(area, bins=abins, color=AQUA, linewidth=0)
    axes[1].set_xscale("log")
    axes[1].axvline(np.median(area), color=INK, linewidth=0.7, linestyle=(0, (3, 2)))
    axes[1].text(np.median(area) * 1.08, axes[1].get_ylim()[1] * 0.92, f"median\n{np.median(area):.0f}",
                 fontsize=5.8, va="top")
    axes[1].set_xlabel("area (px)")
    axes[1].set_title("(b) area", loc="left")
    values, freq = np.unique(counts, return_counts=True)
    axes[2].bar(values, freq, width=0.8, color=ORANGE, linewidth=0)
    axes[2].set_xlabel("filaments per record")
    axes[2].set_title("(c) crowding", loc="left")
    save(figure, "fig_filament_stats.pdf")
    return dict(p50=float(np.percentile(long_side, 50)), p90=float(np.percentile(long_side, 90)),
                p99=float(np.percentile(long_side, 99)), max_long=float(long_side.max()),
                area_median=float(np.median(area)), area_p10=float(np.percentile(area, 10)),
                max_per_record=int(counts.max()), mean_per_record=float(counts.mean()),
                empty_records=int((counts == 0).sum()))


def fig_years(images: Path, coco: dict) -> dict:
    train = sorted({i["file_name"] for i in coco["images"]})
    test = sorted(p.name for p in (images / "test" / "test_images").glob("*.jpeg"))
    years = list(range(2011, 2023))
    def share(names, key):
        c = Counter(key(n) for n in names)
        return np.array([c.get(k, 0) / len(names) for k in (years if key is year_of else sorted(STATIONS))])
    year_of = lambda n: int(n[:4])
    station_of = lambda n: n[-7:-5]
    figure, axes = plt.subplots(1, 2, figsize=(WIDTH, 1.7), gridspec_kw=dict(width_ratios=[2, 1]))
    figure.subplots_adjust(wspace=0.25)
    x = np.arange(len(years))
    axes[0].bar(x - 0.2, share(train, year_of) * 100, 0.38, color=BLUE, label=f"train ({len(train)})")
    axes[0].bar(x + 0.2, share(test, year_of) * 100, 0.38, color=ORANGE, label=f"test ({len(test)})")
    axes[0].set_xticks(x); axes[0].set_xticklabels([str(y)[2:] for y in years])
    axes[0].set_xlabel("year (20xx)"); axes[0].set_ylabel("share of photographs (%)")
    axes[0].legend(loc="upper right", fontsize=6)
    xs = np.arange(len(STATIONS))
    axes[1].bar(xs - 0.2, share(train, station_of) * 100, 0.38, color=BLUE)
    axes[1].bar(xs + 0.2, share(test, station_of) * 100, 0.38, color=ORANGE)
    axes[1].set_xticks(xs); axes[1].set_xticklabels(sorted(STATIONS))
    axes[1].set_xlabel("GONG station")
    save(figure, "fig_years.pdf")
    return dict(train=len(train), test=len(test))


def fig_raster(coco: dict) -> dict:
    """Which pixels two rasterisers call 'inside' the same polygon."""
    # A small, thin filament: rasterisation differences are a large share of its area.
    candidates = [a for a in coco["annotations"] if 250 < a["area"] < 400 and len(a["segmentation"]) == 1]
    ann = sorted(candidates, key=lambda a: a["id"])[0]
    poly = np.array(ann["segmentation"][0], dtype=np.float64).reshape(-1, 2)
    x0, y0 = np.floor(poly.min(axis=0)).astype(int) - 3
    x1, y1 = np.ceil(poly.max(axis=0)).astype(int) + 3
    h, w = y1 - y0, x1 - x0
    local = poly - [x0, y0]
    coco_mask = mu.decode(mu.merge(mu.frPyObjects([local.ravel().tolist()], h, w)))
    cv_mask = np.zeros((h, w), np.uint8)
    cv2.fillPoly(cv_mask, [np.round(local).astype(np.int32)], 1)
    both = (coco_mask & cv_mask).astype(bool)
    only_cv = (cv_mask & ~coco_mask.astype(bool)).astype(bool)
    only_coco = (coco_mask.astype(bool) & ~cv_mask.astype(bool))
    canvas = np.ones((h, w, 3))
    canvas[both] = np.array([0x9E, 0xC5, 0xF4]) / 255
    canvas[only_cv] = np.array([0xEB, 0x68, 0x34]) / 255
    canvas[only_coco] = np.array([0x1B, 0xAF, 0x7A]) / 255
    figure, axis = plt.subplots(figsize=(WIDTH, WIDTH * h / w * 0.62 + 0.3))
    axis.imshow(canvas, interpolation="nearest", extent=(-0.5, w - 0.5, h - 0.5, -0.5))
    closed = np.vstack([local, local[:1]])
    axis.plot(closed[:, 0] - 0.5, closed[:, 1] - 0.5, color=INK, linewidth=0.8)
    axis.set_xticks(np.arange(-0.5, w, 1), minor=True); axis.set_yticks(np.arange(-0.5, h, 1), minor=True)
    axis.grid(which="minor", color="white", linewidth=0.4)
    axis.grid(which="major", visible=False)
    axis.tick_params(which="both", length=0, labelbottom=False, labelleft=False)
    for s in axis.spines.values():
        s.set_visible(False)
    axis.legend(handles=[Patch(color=BLUE_LIGHT, label=f"both ({int(both.sum())} px)"),
                         Patch(color=ORANGE, label=f"cv2.fillPoly only ({int(only_cv.sum())} px)"),
                         Patch(color=AQUA, label=f"pycocotools only ({int(only_coco.sum())} px)"),
                         Line2D([], [], color=INK, linewidth=0.8, label="annotated polygon")],
                loc="upper center", bbox_to_anchor=(0.5, -0.02), ncol=4, fontsize=6)
    save(figure, "fig_raster.pdf")
    return dict(coco=int(coco_mask.sum()), cv=int(cv_mask.sum()),
                ratio=float(cv_mask.sum() / max(1, coco_mask.sum())), iou=float(both.sum() / (both | only_cv | only_coco).sum()))


# --------------------------------------------------------------------------- metric
def fig_leverage(host: dict) -> dict:
    """What each kind of change is worth, from the deployed model's counts."""
    tp, fp, fn = host["tp"], host["fp"], host["fn"]
    s = host["sq"] * tp
    d = tp + 0.5 * fp + 0.5 * fn
    base = s / d
    n = np.arange(0, 121)
    # A recovered miss turns one FN into one TP: +0.65 in the numerator, and the
    # denominator gains 1 (TP) and loses 0.5 (FN).
    recover = (s + 0.65 * n) / (tp + n + 0.5 * fp + 0.5 * (fn - n))
    remove_fp = s / (tp + 0.5 * (fp - np.minimum(n, fp)) + 0.5 * fn)
    polish = (s + 0.05 * n) / d
    figure, axis = plt.subplots(figsize=(WIDTH * 0.62, 1.9))
    for values, colour, label in ((recover, BLUE, "recover a missed filament (IoU 0.65)"),
                                  (remove_fp, ORANGE, "remove a false positive"),
                                  (polish, AQUA, "raise a match's IoU by 0.05")):
        axis.plot(n, values, color=colour, linewidth=1.4, label=label)
    axis.axhline(base, color=MUTED, linewidth=0.7, linestyle=(0, (3, 2)))
    axis.set_xlabel("number of such changes")
    axis.set_ylabel("validation PQ")
    axis.legend(loc="upper left", fontsize=6)
    save(figure, "fig_leverage.pdf")
    return dict(base=base, recover50=float(recover[50]), remove50=float(remove_fp[50]), polish50=float(polish[50]))


def fig_bootstrap(records: dict, emitted: dict, host_pq: float) -> dict:
    model, human = [], []
    for name in sorted(records):
        predicted = [{"size": [2048, 2048], "counts": c} for c in emitted[name]]
        model.append(sum(paperfig.record_sums(rles, predicted) for _, rles in records[name]))
        annotators = [rles for _, rles in records[name]]
        if len(annotators) >= 2:
            human.append(sum(paperfig.record_sums(t, list(o)) for a, o in enumerate(annotators)
                             for b, t in enumerate(annotators) if a != b))
    rng = np.random.default_rng(2026)
    def draws(rows):
        rows = np.array(rows)
        picks = rng.integers(0, len(rows), size=(10000, len(rows)))
        return paperfig.pooled(rows[picks].sum(axis=1))[0]
    m, h = draws(model), draws(human)
    figure, axis = plt.subplots(figsize=(WIDTH * 0.62, 1.8))
    bins = np.linspace(0.28, 0.54, 70)
    axis.hist(h, bins=bins, color=ORANGE, alpha=0.85, linewidth=0, label="annotator vs annotator")
    axis.hist(m, bins=bins, color=BLUE, alpha=0.85, linewidth=0, label="our model vs annotators")
    for values, colour in ((m, BLUE), (h, ORANGE)):
        lo, hi = np.percentile(values, [2.5, 97.5])
        axis.axvline(lo, color=colour, linewidth=0.8, linestyle=(0, (3, 2)))
        axis.axvline(hi, color=colour, linewidth=0.8, linestyle=(0, (3, 2)))
    axis.set_xlabel("pooled PQ in a bootstrap resample of photographs")
    axis.set_ylabel("resamples")
    axis.legend(loc="upper left", fontsize=6)
    save(figure, "fig_bootstrap.pdf")
    return dict(model_lo=float(np.percentile(m, 2.5)), model_hi=float(np.percentile(m, 97.5)),
                human_lo=float(np.percentile(h, 2.5)), human_hi=float(np.percentile(h, 97.5)))


# --------------------------------------------------------------------------- validation
def fig_temporal() -> None:
    """exp log, 'The validation split leaks through time' (leaderboard-analysis.md)."""
    days = ["same day", "1 day", "2 days", "3 days", "7 days"]
    share = [10.4, 43.4, 64.2, 79.2, 94.3]
    figure, axis = plt.subplots(figsize=(WIDTH * 0.6, 1.6))
    axis.bar(range(5), share, color=BLUE, width=0.7, linewidth=0)
    for x, v in enumerate(share):
        axis.text(x, v + 2, f"{v:.1f}%", ha="center", fontsize=6.3, color=INK2)
    axis.set_xticks(range(5)); axis.set_xticklabels(days)
    axis.set_ylim(0, 108)
    axis.set_ylabel("validation photographs (%)")
    axis.set_xlabel("a training photograph lies within")
    save(figure, "fig_temporal.pdf")


def fig_val_public(submissions: Path) -> list:
    rows = list(csv.DictReader(submissions.open(encoding="utf-8")))
    points = []
    for r in rows:
        match = re.search(r"(?:validation|local|val) PQ\s*(0?\.\d+)", r["description"])
        if match:
            points.append((float(match.group(1)), float(r["public_pq"]), r["family"]))
    figure, axis = plt.subplots(figsize=(WIDTH * 0.62, 2.1))
    colours = {"YOLO-seg": BLUE, "Mask2Former": ORANGE, "Semantic U-Net": AQUA, "Other": MUTED}
    for family, colour in colours.items():
        pts = [(v, p) for v, p, f in points if f == family]
        if pts:
            axis.scatter(*zip(*pts), s=16, color=colour, edgecolor="white", linewidth=0.5, zorder=3, label=family)
    xs = np.linspace(0.28, 0.49, 10)
    axis.fill_between(xs, xs - 0.078, xs - 0.067, color=MUTED, alpha=0.18, linewidth=0,
                      label="val - 0.078 to val - 0.067")
    axis.set_xlabel("validation PQ")
    axis.set_ylabel("public PQ (2 decimals)")
    axis.legend(loc="upper left", fontsize=6)
    save(figure, "fig_val_public.pdf")
    return points


# --------------------------------------------------------------------------- YOLO line
def fig_resolution() -> None:
    """exp002 (trained 1280) and exp010 (trained 2048) inference-resolution sweeps
    (leaderboard-analysis.md: 'Inference resolution', 'Resolution is maxed out', exp024)."""
    exp002 = {1280: 0.3736, 1600: 0.3965, 2048: 0.4064, 2560: 0.3993, 3072: 0.3624}
    exp010 = {1792: 0.4180, 2048: 0.4274, 2304: 0.4264, 2560: 0.4088, 3072: 0.3583}
    figure, axis = plt.subplots(figsize=(WIDTH * 0.62, 1.8))
    for data, colour, label in ((exp002, BLUE, "exp002, trained at 1280"), (exp010, ORANGE, "exp010, trained at 2048")):
        axis.plot(list(data), list(data.values()), color=colour, marker="o", markersize=3.4,
                  markeredgecolor="white", markeredgewidth=0.6, linewidth=1.3, label=label)
    axis.axvline(2048, color=AXIS, linewidth=0.8)
    axis.text(2070, 0.365, "native\n2048", fontsize=6.2, color=INK2)
    axis.set_xlabel("inference image size (px)")
    axis.set_ylabel("validation PQ")
    axis.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2, fontsize=6, frameon=False)
    save(figure, "fig_resolution.pdf")


def fig_erosion() -> None:
    """exp005 (exp002, conf 0.35) and exp104 (two-architecture ensemble) growth sweeps."""
    exp005 = {-4: 0.1156, -2: 0.3702, -1: 0.4404, 0: 0.4169, 3: 0.2016}
    exp104 = {-2: 0.4311, -1: 0.4681, 0: 0.4412, 1: 0.3828, 2: 0.3055}
    figure, axis = plt.subplots(figsize=(WIDTH * 0.62, 1.8))
    for data, colour, label in ((exp005, BLUE, "exp002 masks (exp005)"), (exp104, ORANGE, "two-model ensemble (exp104)")):
        axis.plot(list(data), list(data.values()), color=colour, marker="o", markersize=3.4,
                  markeredgecolor="white", markeredgewidth=0.6, linewidth=1.3, label=label)
    axis.axvline(0, color=AXIS, linewidth=0.8)
    axis.set_xlabel("pixels grown (negative = eroded)")
    axis.set_ylabel("validation PQ")
    axis.legend(loc="lower center", fontsize=6)
    save(figure, "fig_erosion.pdf")


def fig_d4(images: Path, coco: dict) -> None:
    """The eight dihedral views of one real filament crop."""
    ann = max((a for a in coco["annotations"] if 1500 < a["area"] < 4000),
              key=lambda a: max(a["bbox"][2], a["bbox"][3]) / max(1, min(a["bbox"][2], a["bbox"][3])))
    image = next(i for i in coco["images"] if i["id"] == ann["image_id"])
    gray = cv2.imread(str(images / "train" / "train_images" / image["file_name"]), cv2.IMREAD_GRAYSCALE)
    x, y, w, h = ann["bbox"]
    cx, cy, half = int(x + w / 2), int(y + h / 2), int(max(w, h) * 0.75)
    crop = paperfig.stretch(gray)[max(0, cy - half):cy + half, max(0, cx - half):cx + half]
    mask = mu.decode(polygon_rle(ann["segmentation"]))[max(0, cy - half):cy + half, max(0, cx - half):cx + half]
    figure, axes = plt.subplots(2, 4, figsize=(WIDTH, WIDTH / 2 + 0.25))
    figure.subplots_adjust(wspace=0.05, hspace=0.25, left=0, right=1, top=0.93, bottom=0)
    names = ["identity", "rotate 90", "rotate 180", "rotate 270",
             "flip", "flip + rotate 90", "flip + rotate 180", "flip + rotate 270"]
    for k, (axis, name) in enumerate(zip(axes.ravel(), names)):
        mirror, turns = k >= 4, k % 4
        view = np.fliplr(crop) if mirror else crop
        m = np.fliplr(mask) if mirror else mask
        view, m = np.rot90(view, turns), np.rot90(m, turns)
        axis.imshow(view, cmap="gray", vmin=0, vmax=1)
        paperfig.draw_outline(axis, paperfig.contours(np.ascontiguousarray(m)), ORANGE, 0.8, halo=1.2)
        axis.set_title(name, fontsize=6.5)
        axis.set_axis_off()
    save(figure, "fig_d4.pdf")


def fig_ensembles() -> None:
    """Every fusion, ensemble and weight-averaging variant against its own control.

    Validation PQ changes transcribed from the experiment log: exp037/exp033 (D4 fusion),
    exp060 (mask NMS), exp102, exp140, exp021, exp101, exp030, exp144, exp059, exp143, exp061.
    """
    rows = [
        ("eight-view D4 fusion of one YOLO (exp037)", 0.0167),
        ("mask NMS instead of box NMS, one view (exp060)", 0.0038),
        ("two YOLO architectures, fused pools (exp102)", 0.00134),
        ("YOLO + Mask2Former, calibrated (exp140)", 0.0010),
        ("second YOLO as a veto (exp021)", 0.0007),
        ("three checkpoints of one run, 24 views (exp101)", 0.00005),
        ("pixel averaging of two YOLO masks (exp030)", -0.0005),
        ("eight-view fusion of Mask2Former (exp144)", -0.0005),
        ("soft probability vote instead of hard (exp059)", -0.0016),
        ("weight soup, Mask2Former epochs 40 + 50 (exp143)", -0.0017),
        ("weight interpolation of two YOLO runs (exp061)", -0.0065),
    ]
    labels = [r[0] for r in rows]
    gains = np.array([r[1] for r in rows])
    figure, axis = plt.subplots(figsize=(WIDTH, 2.55))
    y = np.arange(len(rows))
    axis.barh(y, gains, 0.62, color=[BLUE if g > 0 else ORANGE for g in gains])
    for yy, g in zip(y, gains):
        axis.text(g + (0.0003 if g >= 0 else -0.0003), yy, f"{g:+.5f}" if abs(g) < 1e-4 else f"{g:+.4f}", va="center",
                  ha="left" if g >= 0 else "right", fontsize=6, color=INK2)
    axis.axvline(0, color=AXIS, linewidth=0.8)
    axis.set_yticks(y); axis.set_yticklabels(labels, fontsize=6.3)
    axis.invert_yaxis()
    axis.set_xlim(-0.0105, 0.0205)
    axis.set_xlabel("change in validation PQ against the variant's own control")
    axis.grid(axis="x", color=paperfig.GRID, linewidth=0.5)
    axis.grid(axis="y", visible=False)
    save(figure, "fig_ensembles.pdf")


# --------------------------------------------------------------------------- post-processing and recall
def fig_rim() -> None:
    """exp log 'Where the mask error actually lives' (124 matched pairs)."""
    bands = ["0-1", "1-2", "2-3", "3-5", "5-8", "8+"]
    share = [41.4, 22.7, 13.3, 8.6, 4.5, 9.5]
    figure, axis = plt.subplots(figsize=(WIDTH * 0.6, 1.6))
    axis.bar(range(6), share, color=[BLUE, BLUE, BLUE_LIGHT, BLUE_LIGHT, BLUE_LIGHT, BLUE_LIGHT], width=0.7)
    for x, v in enumerate(share):
        axis.text(x, v + 1.2, f"{v:.1f}%", ha="center", fontsize=6.3, color=INK2)
    axis.set_xticks(range(6)); axis.set_xticklabels(bands)
    axis.set_xlabel("distance from the annotated boundary (px)")
    axis.set_ylabel("share of disagreement")
    save(figure, "fig_rim.pdf")


def fig_ceilings() -> None:
    """exp113-exp122 oracle ceilings against the best real operator under an oracle gate."""
    families = ["split merges", "lateral growth", "axial growth", "union fragments", "admit discarded"]
    ceiling = [0.022342, 0.018865, 0.018103, 0.011579, 0.020822]
    real = [0.002846, 0.006350, 0.001204, None, 0.000480]     # no union operator was built
    figure, axis = plt.subplots(figsize=(WIDTH, 1.8))
    y = np.arange(len(families))
    axis.barh(y + 0.18, ceiling, 0.34, color=BLUE_LIGHT, label="oracle ceiling (reads ground truth)")
    axis.barh(y - 0.18, [r or 0.0 for r in real], 0.34, color=BLUE, label="best implementable operator")
    for yy, c, r in zip(y, ceiling, real):
        axis.text(c + 0.0003, yy + 0.18, f"+{c:.4f}", va="center", fontsize=6)
        axis.text((r or 0.0) + 0.0003, yy - 0.18, "no operator built" if r is None else f"+{r:.4f}",
                  va="center", fontsize=6, color=INK2 if r is None else INK)
    axis.set_yticks(y); axis.set_yticklabels(families)
    axis.invert_yaxis()
    axis.set_xlabel("gain in validation PQ")
    axis.grid(axis="x", color=paperfig.GRID, linewidth=0.5)
    axis.grid(axis="y", visible=False)
    axis.set_xlim(0, 0.0255)
    axis.legend(loc="upper center", bbox_to_anchor=(0.5, -0.3), ncol=2, fontsize=6)
    save(figure, "fig_ceilings.pdf")


def fig_profile(images: Path, coco: dict) -> None:
    """Intensity across a real filament against the step edge a synthetic test assumed."""
    ann = sorted((a for a in coco["annotations"] if 3000 < a["area"] < 6000), key=lambda a: a["id"])[0]
    image = next(i for i in coco["images"] if i["id"] == ann["image_id"])
    gray = cv2.imread(str(images / "train" / "train_images" / image["file_name"]), cv2.IMREAD_GRAYSCALE).astype(float)
    mask = mu.decode(polygon_rle(ann["segmentation"]))
    ys, xs = np.nonzero(mask)
    pts = np.stack([xs, ys], 1).astype(float)
    centre = pts.mean(0)
    _, _, vt = np.linalg.svd(pts - centre, full_matrices=False)
    normal = vt[1]
    offsets = np.arange(-25, 26)
    samples, inside = [], []
    for t in np.linspace(-0.3, 0.3, 9):
        base = centre + vt[0] * t * np.ptp((pts - centre) @ vt[0])
        line = base[None] + offsets[:, None] * normal[None]
        xi, yi = np.clip(np.round(line).astype(int), 0, 2047).T
        samples.append(gray[yi, xi]); inside.append(mask[yi, xi])
    profile = np.mean(samples, 0)
    inside = np.mean(inside, 0) > 0.5
    figure, axis = plt.subplots(figsize=(WIDTH * 0.62, 1.8))
    axis.axvspan(offsets[inside].min() - 0.5, offsets[inside].max() + 0.5, color=AQUA, alpha=0.15,
                 linewidth=0, label="annotated width")
    axis.plot(offsets, profile, color=BLUE, linewidth=1.4, label="real filament (mean of 9 cuts)")
    lo, hi = profile.min(), np.percentile(profile, 90)
    step = np.where(inside, lo, hi)
    axis.step(offsets, step, where="mid", color=ORANGE, linewidth=1.2, label="synthetic step edge")
    axis.set_xlabel("distance across the filament (px)")
    axis.set_ylabel("intensity (0-255)")
    axis.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=3, fontsize=6, frameon=False)
    save(figure, "fig_profile.pdf")


def fig_stages() -> None:
    """exp132: every validation truth by the best eroded candidate the D4 pool holds."""
    labels = ["matched by the rule", "lost to the confidence floor", "boundary failure (0.3-0.5)",
              "weak proposal (0.1-0.3)", "never proposed (<0.1)"]
    counts = [880, 172, 165, 45, 63]
    colours = [BLUE, ORANGE, AQUA, "#EDA100", MUTED]
    figure, axis = plt.subplots(figsize=(WIDTH, 1.05))
    left = 0
    for count, colour, label in zip(counts, colours, labels):
        axis.barh(0, count, left=left, color=colour, height=0.6, edgecolor="white", linewidth=1.2)
        if count > 60:
            axis.text(left + count / 2, 0, f"{count}", ha="center", va="center", color="white", fontsize=7, fontweight="bold")
        left += count
    axis.set_xlim(0, sum(counts)); axis.set_yticks([])
    axis.set_xlabel("validation ground-truth filaments (1,325)")
    axis.grid(visible=False)
    axis.spines["left"].set_visible(False)
    axis.legend(handles=[Patch(color=c, label=f"{l} ({n})") for c, l, n in zip(colours, labels, counts)],
                loc="upper center", bbox_to_anchor=(0.5, -0.75), ncol=3, fontsize=6)
    save(figure, "fig_stages.pdf")


def fig_breakeven() -> None:
    q = np.linspace(0.5, 0.95, 50)
    figure, axis = plt.subplots(figsize=(WIDTH * 0.6, 1.75))
    for pq, colour, label in ((0.4559, BLUE, "baseline PQ 0.456 (YOLO D4)"), (0.4705, ORANGE, "PQ 0.4705 (Mask2Former)")):
        axis.plot(q, pq / (2 * q), color=colour, linewidth=1.4, label=label)
    axis.scatter([0.651], [0.4559 / 1.302], s=18, color=BLUE, zorder=3, edgecolor="white")
    axis.annotate("exp118: 0.35 at q = 0.651", (0.651, 0.35), xytext=(0.70, 0.40), fontsize=6.2,
                  arrowprops=dict(arrowstyle="-", color=MUTED, linewidth=0.6))
    axis.axhline(0.251, color=MUTED, linewidth=0.7, linestyle=(0, (3, 2)))
    axis.text(0.52, 0.256, "blind admission precision 0.25", fontsize=6.2, color=INK2)
    axis.set_xlabel("mean IoU q of a correctly admitted candidate")
    axis.set_ylabel("break-even precision")
    axis.legend(loc="upper right", fontsize=6)
    save(figure, "fig_breakeven.pdf")


def fig_classical() -> None:
    """exp103: share of the disk a threshold keeps when it keeps 80% of filament pixels."""
    names = ["darkness", "wide\ndepression", "oriented\ntop-hat", "Hessian\nridge"]
    kept = [0.374, 0.300, 0.378, 0.262]
    figure, axis = plt.subplots(figsize=(WIDTH * 0.6, 1.6))
    axis.bar(range(4), kept, color=BLUE, width=0.65)
    axis.axhline(0.05, color=ORANGE, linewidth=1.0)
    axis.set_xlim(-0.5, 4.75)
    axis.text(3.45, 0.064, "usable\n< 0.05", fontsize=6.3, color=INK2, ha="left", va="bottom")
    for x, v in enumerate(kept):
        axis.text(x, v + 0.01, f"{v:.3f}", ha="center", fontsize=6.3, color=INK2)
    axis.set_xticks(range(4)); axis.set_xticklabels(names, fontsize=6.3)
    axis.set_ylabel("share of disk kept")
    axis.set_ylim(0, 0.45)
    save(figure, "fig_classical.pdf")


# --------------------------------------------------------------------------- Mask2Former
def fig_memory() -> None:
    """exp139 addendum: CPU-measured memory of the full Swin-S Mask2Former (bf16, fwd+loss+bwd)."""
    px = np.array([512, 768])
    plain = np.array([1.629, 3.168])
    ckpt = np.array([1.079, 2.035])
    pixels = px ** 2 / 1e6
    figure, axis = plt.subplots(figsize=(WIDTH * 0.6, 1.8))
    grid = np.linspace(0, 4.3, 50)
    for values, colour, label in ((plain, BLUE, "no checkpointing"), (ckpt, AQUA, "backbone checkpointing")):
        slope, intercept = np.polyfit(pixels, values, 1)
        axis.plot(grid, slope * grid + intercept, color=colour, linewidth=1.0, linestyle=(0, (3, 2)))
        axis.scatter(pixels, values, color=colour, s=16, zorder=3, edgecolor="white", label=label)
    axis.scatter([1.048576], [6.64], color=ORANGE, marker="s", s=16, zorder=3, label="T4 measured, 1024 crop (HF attention)")
    axis.scatter([1.048576], [5.16], color=ORANGE, marker="^", s=18, zorder=3, label="T4 measured, 1024 crop (CUDA kernel)")
    axis.set_xlabel("pixels per image (millions)")
    axis.set_ylabel("activation memory (GiB)")
    axis.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2, fontsize=5.8, frameon=False)
    save(figure, "fig_memory.pdf")


def fig_kink() -> None:
    """Bilinear sampling: the value is continuous, its derivative jumps at pixel centres."""
    values = np.array([0.2, 0.9, 0.4, 0.6, 0.1])
    x = np.linspace(0, 4, 801)
    interp = np.interp(x, np.arange(5), values)
    deriv = np.gradient(interp, x)
    figure, axes = plt.subplots(2, 1, figsize=(WIDTH * 0.6, 2.2), sharex=True)
    figure.subplots_adjust(hspace=0.15)
    axes[0].plot(x, interp, color=BLUE, linewidth=1.4)
    axes[0].scatter(np.arange(5), values, color=INK, s=10, zorder=3)
    axes[0].set_ylabel("sampled value")
    axes[1].plot(x, deriv, color=ORANGE, linewidth=1.4)
    for c in range(5):
        for axis in axes:
            axis.axvline(c, color=AXIS, linewidth=0.6, linestyle=(0, (2, 2)))
    axes[1].set_ylabel("d value / d location")
    axes[1].set_xlabel("sampling location (pixel centres at integers)")
    save(figure, "fig_kink.pdf")


def copy_paper_figures() -> None:
    for name in ("fig_curve.pdf", "fig_host_metrics.pdf", "fig_qualitative.pdf", "fig_failures.pdf",
                 "fig_size.pdf", "fig_progress.pdf", "thumb_input.png", "thumb_output.png"):
        shutil.copy(ROOT / "paper" / "figures" / name, HERE / "figures" / name)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--images", type=Path, required=True, help="unpacked MAGFiLO_1.0_Kaggle_2026 folder")
    args = parser.parse_args()
    (HERE / "figures").mkdir(exist_ok=True)
    (HERE / "generated").mkdir(exist_ok=True)
    coco = json.loads(args.annotations.read_text())
    host = json.loads((ROOT / "paper" / "data" / "host_eval.json").read_text())
    emitted = json.loads((ROOT / "paper" / "data" / "validation_emitted.json").read_text())
    records, _ = records_for(args.annotations)

    stations = fig_stations(args.images, coco)
    annot = fig_annotators(args.images, coco)
    stats = fig_filament_stats(coco)
    counts = fig_years(args.images, coco)
    raster = fig_raster(coco)
    leverage = fig_leverage(host)
    boot = fig_bootstrap(records, emitted, host["host_pq"])
    fig_temporal()
    points = fig_val_public(ROOT / "paper" / "data" / "submissions.csv")
    fig_resolution(); fig_erosion(); fig_d4(args.images, coco); fig_ensembles()
    fig_rim(); fig_ceilings(); fig_profile(args.images, coco)
    fig_stages(); fig_breakeven(); fig_classical(); fig_memory(); fig_kink()
    copy_paper_figures()
    shutil.copy(ROOT / "paper" / "generated" / "numbers.tex", HERE / "generated" / "paper_numbers.tex")

    photographs = Counter(i["file_name"] for i in coco["images"])
    multiplicity = Counter(photographs.values())
    per_station = Counter(n[-7:-5] for n in photographs)
    values = dict(
        LongPfifty=f"{stats['p50']:.0f}", LongPninety=f"{stats['p90']:.0f}", LongPninetynine=f"{stats['p99']:.0f}",
        LongMax=f"{stats['max_long']:.0f}", AreaMedian=f"{stats['area_median']:.0f}", AreaPten=f"{stats['area_p10']:.0f}",
        MaxPerRecord=stats["max_per_record"], MeanPerRecord=f"{stats['mean_per_record']:.1f}",
        EmptyRecords=stats["empty_records"],
        OnceAnnotated=multiplicity.get(1, 0), TwiceAnnotated=multiplicity.get(2, 0), ThriceAnnotated=multiplicity.get(3, 0),
        TestPhotos=counts["test"],
        StationBh=per_station.get("Bh", 0), StationCh=per_station.get("Ch", 0), StationLh=per_station.get("Lh", 0),
        StationMh=per_station.get("Mh", 0), StationTh=per_station.get("Th", 0), StationUh=per_station.get("Uh", 0),
        RasterCoco=raster["coco"], RasterCv=raster["cv"], RasterRatio=f"{raster['ratio']:.2f}",
        RasterIoU=f"{raster['iou']:.3f}",
        AnnotatorOne=annot["counts"][0], AnnotatorTwo=annot["counts"][1], AnnotatorThree=annot["counts"][2],
        LeverageRecover=f"{leverage['recover50']:.4f}", LeverageRemove=f"{leverage['remove50']:.4f}",
        LeveragePolish=f"{leverage['polish50']:.4f}",
        ValPublicPoints=len(points),
    )
    text = paperfig.macros(values).replace("paper/make_figures.py", "book/make_book_figures.py")
    (HERE / "generated" / "book_numbers.tex").write_text(text, encoding="utf-8")
    print(json.dumps(dict(stations=stations, annotators=annot, stats=stats, raster=raster, leverage=leverage,
                          bootstrap=boot, points=len(points)), indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
