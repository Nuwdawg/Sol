"""exp149: evidence for each exp139 candidate, and the exact emission and PQ path to score rules.

Everything here reads the exp147 cache; nothing runs a network. A candidate of the `full`
view (the deployed view) is described by:

- the model's own terms: score (class probability x mean mask probability), p, m, area,
  and the class probability of its query at every decoder layer;
- agreement from views the model saw differently: the seven non-identity dihedral views,
  nine 1024 tiles at native scale, and the frame resampled to 2560. For each source the
  best IoU with any candidate of that source and the score of that candidate;
- physics of the photograph: position on the disk (r/R), and how dark the masked pixels
  are against a ring around them, relative to the disk's median intensity. A filament is
  absorption against the chromosphere, so a real one is darker than its surroundings.

The emission is exp033's operation for operation (paint in descending score, keep the
unclaimed pixels, drop pieces under the area floor) and PQ is the official pooled rule, so
a rule scored here deploys unchanged.
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
from pycocotools import mask as mu

from experiments.exp_033_inference.src.sweep import pq_rle, records_for
from shared.solar_disk import find_disk
from shared.utils import aggregate_pq

SIZE = 2048
KERNEL = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))


def as_rle(rle: dict) -> dict:
    counts = rle["counts"]
    return dict(size=list(rle["size"]), counts=counts.encode("ascii") if isinstance(counts, str) else counts)


def load_cache(path: Path) -> dict:
    return json.loads(Path(path).read_text())["photographs"]


class Crop:
    """A binary mask kept as its bounding-box crop, for fast painting."""

    __slots__ = ("x0", "y0", "mask")

    def __init__(self, rle: dict, grow: int = 0):
        full = mu.decode(as_rle(rle))
        if grow < 0:
            full = cv2.erode(full, KERNEL, iterations=-grow)
        elif grow > 0:
            full = cv2.dilate(full, KERNEL, iterations=grow)
        ys, xs = np.nonzero(full)
        if len(ys) == 0:
            self.x0 = self.y0 = 0
            self.mask = np.zeros((0, 0), np.uint8)
            return
        self.y0, self.x0 = int(ys.min()), int(xs.min())
        self.mask = np.ascontiguousarray(full[self.y0:ys.max() + 1, self.x0:xs.max() + 1])


def paint(entries: list[tuple[float, Crop]], min_area: int) -> list[tuple[float, dict]]:
    """exp033's paint_panoptic on crops: descending score, unclaimed pixels, area floor."""
    claimed = np.zeros((SIZE, SIZE), np.uint8)
    kept = []
    for score, crop in sorted(entries, key=lambda item: -item[0]):
        if crop.mask.size == 0:
            continue
        h, w = crop.mask.shape
        window = claimed[crop.y0:crop.y0 + h, crop.x0:crop.x0 + w]
        remaining = crop.mask & (1 - window)
        area = int(remaining.sum())
        if area < min_area:
            continue
        window |= remaining
        full = np.zeros((SIZE, SIZE), np.uint8)
        full[crop.y0:crop.y0 + h, crop.x0:crop.x0 + w] = remaining
        rle = mu.encode(np.asfortranarray(full))
        kept.append((score, rle))
    return kept


def score_rule(pool: dict, records: dict, thresholds, min_area: int = 100) -> tuple[list[dict], dict]:
    """PQ at every threshold for `pool` = {photo: [(score, Crop)]}; also per-record rows at each threshold."""
    thresholds = sorted(set(round(float(t), 6) for t in thresholds))
    totals = {t: [] for t in thresholds}
    per_record = {t: {} for t in thresholds}
    for name, passes in records.items():
        entries = [(s, c) for s, c in pool.get(name, []) if s >= thresholds[0]]
        painted = paint(entries, min_area)
        for t in thresholds:
            predicted = [rle for s, rle in painted if s >= t]
            for image_id, truth in passes:
                row = pq_rle(predicted, truth)
                totals[t].append(row)
                per_record[t][image_id] = dict(row, photograph=name)
    rows = [dict(aggregate_pq(totals[t]), conf=t, min_area=min_area) for t in thresholds]
    return rows, per_record


def truth_ious(rles: list[dict], passes: list) -> np.ndarray:
    """[candidates, records] best IoU of each raw candidate mask with any truth of that record."""
    out = np.zeros((len(rles), len(passes)), np.float64)
    if not rles:
        return out
    for j, (_, truth) in enumerate(passes):
        if truth:
            ious = np.asarray(mu.iou(rles, truth, [0] * len(truth))).reshape(len(rles), len(truth))
            out[:, j] = ious.max(axis=1)
    return out


def best_overlap(rles: list[dict], others: list[dict], scores: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """For each mask in `rles`: best IoU with `others`, and the score of that best partner."""
    if not rles or not others:
        return np.zeros(len(rles)), np.zeros(len(rles))
    ious = np.asarray(mu.iou(rles, others, [0] * len(others))).reshape(len(rles), len(others))
    best = ious.argmax(axis=1)
    return ious.max(axis=1), scores[best]


def ring_contrast(gray: np.ndarray, crop: Crop, disk_median: float, width: int = 6) -> tuple[float, float]:
    """(inside mean - ring mean) / disk median, and inside mean / disk median."""
    h, w = crop.mask.shape
    pad = width + 2
    y0, x0 = max(0, crop.y0 - pad), max(0, crop.x0 - pad)
    y1, x1 = min(SIZE, crop.y0 + h + pad), min(SIZE, crop.x0 + w + pad)
    local = np.zeros((y1 - y0, x1 - x0), np.uint8)
    local[crop.y0 - y0:crop.y0 - y0 + h, crop.x0 - x0:crop.x0 - x0 + w] = crop.mask
    ring = cv2.dilate(local, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * width + 1, 2 * width + 1))) & (1 - local)
    patch = gray[y0:y1, x0:x1].astype(np.float64)
    inside = patch[local > 0]
    around = patch[(ring > 0) & (patch > 20)]
    if inside.size == 0 or around.size == 0 or disk_median <= 0:
        return 0.0, 1.0
    return float((inside.mean() - around.mean()) / disk_median), float(inside.mean() / disk_median)


def shape_terms(crop: Crop) -> tuple[float, float]:
    """Elongation (major/minor axis of the pixel cloud) and mean thickness (area / skeleton-free proxy)."""
    ys, xs = np.nonzero(crop.mask)
    if len(ys) < 3:
        return 1.0, 0.0
    cov = np.cov(np.stack([xs, ys]).astype(np.float64))
    eig = np.sort(np.linalg.eigvalsh(cov))
    elongation = float(np.sqrt(max(eig[1], 1e-9) / max(eig[0], 1e-9)))
    length = 4.0 * np.sqrt(max(eig[1], 1e-9))                   # ~ extent along the major axis
    return elongation, float(len(ys) / max(length, 1.0))


FEATURES = ("score", "p", "m", "log_area", "layer_min3", "layer_std4", "layer_first",
            "d4_votes", "d4_mean_score", "d4_mean_iou", "tile_iou", "tile_score",
            "scale_iou", "scale_score", "radius", "contrast", "relative_dark", "elongation", "thickness")


def photo_evidence(name: str, entry: dict, passes: list, gray: np.ndarray | None) -> dict:
    """Per full-view candidate: features, per-record truth IoU, and its crop."""
    full = entry["full"]
    rles = [as_rle(c["rle"]) for c in full]
    scores = np.array([c["score"] for c in full])
    rows = np.zeros((len(full), len(FEATURES)), np.float64)
    for i, c in enumerate(full):
        layers = np.array(c.get("layers", [c["p"]]), np.float64)
        above = np.nonzero(layers >= 0.5)[0]
        rows[i, :7] = (c["score"], c["p"], c["m"], np.log(max(c["area"], 1)), layers[-3:].min(),
                       layers[-4:].std(), (above[0] / max(len(layers) - 1, 1)) if len(above) else 1.5)
    d4 = entry.get("d4", [])
    votes = np.zeros(len(full)); mean_score = np.zeros(len(full)); mean_iou = np.zeros(len(full))
    for view in range(1, 8):
        members = [c for c in d4 if c["view"] == view]
        iou, partner = best_overlap(rles, [as_rle(c["rle"]) for c in members],
                                    np.array([c["score"] for c in members]))
        votes += iou > 0.5
        mean_score += np.where(iou > 0.5, partner, 0.0)
        mean_iou += iou
    rows[:, 7], rows[:, 8], rows[:, 9] = votes / 7.0, mean_score / 7.0, mean_iou / 7.0
    for column, source in ((10, "tiles"), (12, "scale")):
        members = entry.get(source, [])
        iou, partner = best_overlap(rles, [as_rle(c["rle"]) for c in members], np.array([c["score"] for c in members]))
        rows[:, column], rows[:, column + 1] = iou, np.where(iou > 0.3, partner, 0.0)
    crops = [Crop(c["rle"]) for c in full]
    if gray is not None:
        x, y, r = find_disk(gray)
        yy, xx = np.ogrid[:SIZE, :SIZE]
        disk = ((xx - x) ** 2 + (yy - y) ** 2) <= (0.95 * r) ** 2
        median = float(np.median(gray[disk]))
        for i, crop in enumerate(crops):
            if crop.mask.size == 0:
                continue
            ys, xs = np.nonzero(crop.mask)
            cx, cy = crop.x0 + xs.mean(), crop.y0 + ys.mean()
            rows[i, 14] = float(np.hypot(cx - x, cy - y) / r)
            rows[i, 15], rows[i, 16] = ring_contrast(gray, crop, median)
            rows[i, 17], rows[i, 18] = shape_terms(crop)
    return dict(name=name, features=rows, truth=truth_ious(rles, passes), crops=crops, scores=scores,
                records=[image_id for image_id, _ in passes])


def extra_candidates(entry: dict, source: str, full_rles: list[dict], link: float = 0.3) -> list[dict]:
    """Candidates of another view that overlap no full-view candidate at IoU >= link."""
    members = entry.get(source, [])
    if not members:
        return []
    rles = [as_rle(c["rle"]) for c in members]
    if full_rles:
        ious = np.asarray(mu.iou(rles, full_rles, [0] * len(full_rles))).reshape(len(rles), len(full_rles))
        novel = ious.max(axis=1) < link
    else:
        novel = np.ones(len(rles), bool)
    return [c for c, keep in zip(members, novel) if keep]


def load_all(cache_path: Path, annotations: Path, images: Path | None) -> tuple[dict, dict, list[dict]]:
    cache = load_cache(cache_path)
    records, _ = records_for(annotations)
    evidence = []
    for name in sorted(records):
        gray = cv2.imread(str(images / name), cv2.IMREAD_GRAYSCALE) if images is not None else None
        evidence.append(photo_evidence(name, cache[name], records[name], gray))
    return cache, records, evidence


def bootstrap_difference(base: dict, other: dict, records: dict, draws: int = 5000, seed: int = 2026) -> dict:
    """Paired photograph bootstrap of PQ(other) - PQ(base), from per-record rows at a fixed rule."""
    names = sorted(records)
    index = {name: i for i, name in enumerate(names)}
    sums = np.zeros((2, len(names), 4))                      # iou_sum, tp, fp, fn
    for k, rows in enumerate((base, other)):
        for image_id, row in rows.items():
            sums[k, index[row["photograph"]]] += (row["iou_sum"], row["tp"], row["fp"], row["fn"])
    rng = np.random.default_rng(seed)
    weights = rng.multinomial(len(names), np.full(len(names), 1.0 / len(names)), size=draws).astype(np.float64)

    def pq(total):
        iou, tp, fp, fn = total[..., 0], total[..., 1], total[..., 2], total[..., 3]
        return np.where(tp > 0, iou / np.maximum(tp, 1), 0) * tp / np.maximum(tp + 0.5 * fp + 0.5 * fn, 1e-9)

    a = pq(weights @ sums[0])
    b = pq(weights @ sums[1])
    diff = b - a
    return dict(mean=float(diff.mean()), low=float(np.percentile(diff, 2.5)), high=float(np.percentile(diff, 97.5)),
                positive=float((diff > 0).mean()))
