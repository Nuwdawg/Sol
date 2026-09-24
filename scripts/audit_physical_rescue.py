"""Validate image-supported rescue of low-confidence pretrained proposals.

The pretrained segmenter remains the proposal generator.  This audit asks
whether simple, physically motivated H-alpha evidence can select useful
low-confidence proposals without changing the accepted high-confidence set.
Ground truth is used only for the canonical validation score and diagnostics.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from pycocotools import mask as mu

from experiments.exp_033_inference.src.sweep import encode, pq_rle, records_for
from shared.preprocessing import SolarDisk, flatten_limb_darkening
from shared.utils import aggregate_pq, paint_panoptic


def candidate_features(
    mask: np.ndarray,
    flat: np.ndarray,
    disk_mask: np.ndarray,
    disk_scale: float,
) -> dict:
    """Measure darkness and filament-like shape inside a proposed mask."""
    x, y, width, height = cv2.boundingRect(mask)
    margin = 8
    x1, y1 = max(0, x - margin), max(0, y - margin)
    x2 = min(mask.shape[1], x + width + margin)
    y2 = min(mask.shape[0], y + height + margin)
    mask_crop = mask[y1:y2, x1:x2]
    disk_crop = disk_mask[y1:y2, x1:x2]
    flat_crop = flat[y1:y2, x1:x2]
    inside = mask_crop.astype(bool) & disk_crop
    area = int(inside.sum())
    if area == 0:
        return dict(contrast_z=-99.0, dark_fraction=0.0, eccentricity=0.0,
                    area=0, on_disk_fraction=0.0)

    ring = cv2.dilate(inside.astype(np.uint8), np.ones((5, 5), np.uint8)).astype(bool)
    ring &= ~inside & disk_crop
    inner_median = float(np.median(flat_crop[inside]))
    ring_median = float(np.median(flat_crop[ring])) if ring.any() else inner_median
    contrast_z = (ring_median - inner_median) / disk_scale
    dark_fraction = float(np.mean(flat_crop[inside] < ring_median))

    moments = cv2.moments(inside.astype(np.uint8), binaryImage=True)
    if moments["m00"] <= 0:
        eccentricity = 0.0
    else:
        cov = np.array([
            [moments["mu20"], moments["mu11"]],
            [moments["mu11"], moments["mu02"]],
        ], dtype=np.float64) / moments["m00"]
        eigenvalues = np.maximum(np.linalg.eigvalsh(cov), 0.0)
        eccentricity = float(np.sqrt(1.0 - eigenvalues[0] / max(eigenvalues[1], 1e-9)))
    return dict(
        contrast_z=float(contrast_z),
        dark_fraction=dark_fraction,
        eccentricity=eccentricity,
        area=area,
        on_disk_fraction=float(area / max(int(mask_crop.sum()), 1)),
    )


def detect_disk_fast(image: np.ndarray) -> SolarDisk:
    """Find the dominant bright disk without the expensive Hough transform."""
    blurred = cv2.GaussianBlur(image, (9, 9), 2.0)
    _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    count, _, stats, centroids = cv2.connectedComponentsWithStats(binary, 8)
    if count <= 1:
        height, width = image.shape
        return SolarDisk(width / 2.0, height / 2.0, .45 * min(height, width))
    largest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    cx, cy = centroids[largest]
    radius = np.sqrt(float(stats[largest, cv2.CC_STAT_AREA]) / np.pi)
    return SolarDisk(float(cx), float(cy), float(radius))


def emitted(entries: list[dict], selected: set[int], kernel: np.ndarray) -> list[dict]:
    candidates = []
    for index, entry in enumerate(entries):
        if entry["score"] < .35 and index not in selected:
            continue
        mask = cv2.erode(mu.decode(entry["rle"]), kernel)
        candidates.append((entry["score"], mask))
    return [encode(mask) for _, mask, _ in paint_panoptic(candidates, min_area=300)]


def emit_with_baseline(
    baseline: list[dict],
    entries: list[dict],
    selected: set[int],
    kernel: np.ndarray,
) -> list[dict]:
    """Append selected low-score masks to an already-painted baseline."""
    output = list(baseline)
    if baseline:
        claimed = mu.decode(mu.merge(baseline)).astype(bool)
    else:
        claimed = np.zeros((2048, 2048), dtype=bool)
    for index in sorted(selected, key=lambda i: entries[i]["score"], reverse=True):
        mask = cv2.erode(mu.decode(entries[index]["rle"]), kernel).astype(bool)
        mask[claimed] = False
        if int(mask.sum()) < 300:
            continue
        output.append(encode(mask))
        claimed |= mask
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--images", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    cv2.setNumThreads(1)

    cache = json.loads(args.candidates.read_text())
    records, split = records_for(args.annotations)
    assert set(cache) == set(records)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))

    stage_path = args.output / "feature_stage.json"
    if stage_path.exists():
        stage = json.loads(stage_path.read_text())
        features = stage["features"]
        oracle = stage["oracle"]
        baseline_metric = stage["baseline"]
        print("RESUMED FEATURES", flush=True)
    else:
        features: dict[str, list[dict]] = {}
        oracle = dict(missed=0, recoverable_low_confidence=0)
        baseline_rows = []
        for number, name in enumerate(sorted(records), 1):
            image = cv2.imread(str(args.images / name), cv2.IMREAD_GRAYSCALE)
            if image is None:
                raise FileNotFoundError(args.images / name)
            feature_image = cv2.resize(image, (512, 512), interpolation=cv2.INTER_AREA)
            disk = detect_disk_fast(feature_image)
            flat = flatten_limb_darkening(feature_image, disk)
            disk_mask = disk.mask(feature_image.shape, shrink=.98).astype(bool)
            disk_values = flat[disk_mask]
            disk_scale = float(np.median(np.abs(disk_values - np.median(disk_values)))) + 1e-6
            rows = []
            low_entries = [(index, entry) for index, entry in enumerate(cache[name])
                           if entry["score"] < .35]
            for index, entry in low_entries:
                full_mask = mu.decode(entry["rle"])
                feature_mask = cv2.resize(full_mask, (512, 512), interpolation=cv2.INTER_NEAREST)
                row = dict(index=index, model_score=float(entry["score"]),
                           record_count=0, hit_records=0, iou_sum=0.0)
                row.update(candidate_features(feature_mask, flat, disk_mask, disk_scale))
                row["area"] = int(mu.area(entry["rle"]))
                rows.append(row)
            features[name] = rows
            feature_by_index = {row["index"]: row for row in rows}

            base = emitted(cache[name], set(), kernel)
            lows = [entry["rle"] for _, entry in low_entries]
            for image_id, truth in records[name]:
                baseline_rows.append(dict(pq_rle(base, truth), image_id=image_id, photograph=name))
                base_iou = np.asarray(mu.iou(base, truth, [0] * len(truth))) if base and truth else np.zeros((len(base), len(truth)))
                matched_p, matched_g = set(), set()
                for i, j in sorted(np.argwhere(base_iou > .5), key=lambda pair: -base_iou[tuple(pair)]):
                    if int(i) not in matched_p and int(j) not in matched_g:
                        matched_p.add(int(i)); matched_g.add(int(j))
                low_iou = np.asarray(mu.iou(lows, truth, [0] * len(truth))) if lows and truth else np.zeros((len(lows), len(truth)))
                unmatched_truth = [j for j in range(len(truth)) if j not in matched_g]
                for low_position, (index, _) in enumerate(low_entries):
                    row = feature_by_index[index]
                    row["record_count"] += 1
                    best_iou = (float(low_iou[low_position, unmatched_truth].max())
                                if unmatched_truth else 0.0)
                    if best_iou > .5:
                        row["hit_records"] += 1
                        row["iou_sum"] += best_iou
                for j in range(len(truth)):
                    if j in matched_g:
                        continue
                    oracle["missed"] += 1
                    if len(lows) and float(low_iou[:, j].max()) > .5:
                        oracle["recoverable_low_confidence"] += 1
            if number % 10 == 0:
                print(f"FEATURES {number}/{len(records)}", flush=True)
        baseline_metric = aggregate_pq(baseline_rows)
        temporary = stage_path.with_suffix(".tmp")
        temporary.write_text(json.dumps(dict(features=features, oracle=oracle,
                                              baseline=baseline_metric)))
        temporary.replace(stage_path)
        print("SAVED FEATURES", flush=True)

    # Coarse global validation sweep.  Every rule is label-free at inference:
    # model confidence plus darkness, dark-pixel coverage and elongation.
    rules = []
    for confidence in (.14, .17, .20, .23, .26, .29, .32):
        for contrast in (0.0, .15, .30, .50, .75, 1.0):
            for dark_fraction in (.50, .60, .70):
                for eccentricity in (0.0, .70, .85, .93):
                    rules.append(dict(confidence=confidence, contrast_z=contrast,
                                      dark_fraction=dark_fraction, eccentricity=eccentricity))

    def selected_for(name: str, rule: dict) -> set[int]:
        return {
            row["index"] for row in features[name]
            if row["model_score"] >= rule["confidence"]
            and row["contrast_z"] >= rule["contrast_z"]
            and row["dark_fraction"] >= rule["dark_fraction"]
            and row["eccentricity"] >= rule["eccentricity"]
            and row["on_disk_fraction"] >= .95
            and row["area"] >= 300
        }

    approximate = []
    for rule in rules:
        selected_by_name = {name: selected_for(name, rule) for name in sorted(records)}
        selected_rows = [row for name in sorted(records) for row in features[name]
                         if row["index"] in selected_by_name[name]]
        utility = sum(row["iou_sum"] - .5 * (row["record_count"] - row["hit_records"])
                      for row in selected_rows)
        signature = tuple((name, *sorted(selected_by_name[name])) for name in sorted(records))
        approximate.append(dict(rule=rule, approximate_utility=float(utility),
                                additions=len(selected_rows), signature=signature))
    approximate.sort(key=lambda row: row["approximate_utility"], reverse=True)
    shortlist, seen = [], set()
    for row in approximate:
        if row["signature"] in seen:
            continue
        seen.add(row["signature"])
        shortlist.append(row)
        if len(shortlist) == 32:
            break

    progress_path = args.output / "exact_progress.json"
    scored = json.loads(progress_path.read_text()) if progress_path.exists() else []
    completed_rules = {json.dumps(row["rule"], sort_keys=True) for row in scored}

    baseline_cache_path = args.output / "baseline_cache.json"
    if baseline_cache_path.exists():
        baseline_cache = json.loads(baseline_cache_path.read_text())
        print("RESUMED BASELINE CACHE", flush=True)
    else:
        baseline_cache = {
            name: emitted(cache[name], set(), kernel) for name in sorted(records)
        }
        temporary = baseline_cache_path.with_suffix(".tmp")
        temporary.write_text(json.dumps(baseline_cache))
        temporary.replace(baseline_cache_path)
        print("SAVED BASELINE CACHE", flush=True)

    # Prove that the optimized path is exactly equivalent before scoring.
    parity_rule = shortlist[0]["rule"]
    for name in sorted(records)[:3]:
        selected = selected_for(name, parity_rule)
        assert emit_with_baseline(baseline_cache[name], cache[name], selected, kernel) == emitted(
            cache[name], selected, kernel
        )
    print("BASELINE CACHE PARITY OK", flush=True)

    for rule_number, approximate_row in enumerate(shortlist, 1):
        rule = approximate_row["rule"]
        if json.dumps(rule, sort_keys=True) in completed_rules:
            print(f"RESUMED SCORE {rule_number}/{len(shortlist)}", flush=True)
            continue
        score_rows, prediction_cache, additions = [], {}, 0
        for name in sorted(records):
            selected = selected_for(name, rule)
            additions += len(selected)
            prediction_cache[name] = emit_with_baseline(
                baseline_cache[name], cache[name], selected, kernel
            )
            for image_id, truth in records[name]:
                score_rows.append(dict(pq_rle(prediction_cache[name], truth),
                                       image_id=image_id, photograph=name))
        metric = aggregate_pq(score_rows)
        scored.append(dict(rule=rule, additions=additions,
                           approximate_utility=approximate_row["approximate_utility"], **metric))
        temporary = progress_path.with_suffix(".tmp")
        temporary.write_text(json.dumps(scored))
        temporary.replace(progress_path)
        print(f"SCORED {rule_number}/{len(shortlist)}", flush=True)

    scored.sort(key=lambda row: row["pq"], reverse=True)
    # Recreate the true best cache after sorting; the loop-local shortcut above
    # deliberately avoids retaining hundreds of full caches.
    best_rule = scored[0]["rule"]
    best_cache = {}
    for name in sorted(records):
        selected = selected_for(name, best_rule)
        best_cache[name] = emit_with_baseline(
            baseline_cache[name], cache[name], selected, kernel
        )

    report = dict(
        status="complete",
        split=split.summary(),
        baseline=baseline_metric,
        oracle_diagnostic=oracle,
        best=scored[0],
        top_rules=scored[:20],
        interpretation=(
            "Ground truth was used only to score global rules and measure proposal recall. "
            "The deployable rule uses pretrained confidence and image/shape evidence."
        ),
    )
    (args.output / "features.json").write_text(json.dumps(features))
    (args.output / "best_predictions.json").write_text(json.dumps(best_cache))
    (args.output / "report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
