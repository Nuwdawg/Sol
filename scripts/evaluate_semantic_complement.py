"""Canonical validation of a frozen dense model as a YOLO complement.

The dense checkpoint is never promoted alone.  It supplies independent
semantic support for weak pretrained instance proposals and proposes dark
regions outside the accepted YOLO panoptic mask.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from pycocotools import mask as mu
import torch

from experiments.exp_033_inference.src.sweep import encode, pq_rle, records_for
from shared.utils import aggregate_pq, paint_panoptic


def build_model():
    import segmentation_models_pytorch as smp
    return smp.Unet(encoder_name="resnet34", encoder_weights=None,
                    in_channels=3, classes=1)


def predict_probability(model, path: Path, device: torch.device) -> np.ndarray:
    raw = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if raw is None:
        raise FileNotFoundError(path)
    small = cv2.resize(raw, (1024, 1024), interpolation=cv2.INTER_AREA)
    array = np.repeat((small.astype(np.float32) / 255.0)[None], 3, axis=0)
    tensor = torch.from_numpy(array)[None].to(device)
    with torch.inference_mode():
        probability = torch.sigmoid(model(tensor))
        for dims in ((2,), (3,), (2, 3)):
            flipped = torch.flip(tensor, dims=dims)
            probability += torch.flip(torch.sigmoid(model(flipped)), dims=dims)
        probability /= 4.0
    return probability[0, 0].float().cpu().numpy()


def eroded_mask(entry: dict, kernel: np.ndarray) -> np.ndarray:
    return cv2.erode(mu.decode(entry["rle"]), kernel)


def emit_yolo(entries: list[dict], selected: set[int], kernel: np.ndarray) -> list[dict]:
    candidates = []
    for index, entry in enumerate(entries):
        if entry["score"] < .35 and index not in selected:
            continue
        candidates.append((entry["score"], eroded_mask(entry, kernel)))
    return [encode(mask) for _, mask, _ in paint_panoptic(candidates, min_area=300)]


def emit_with_baseline(
    baseline: list[dict], claimed_baseline: np.ndarray, entries: list[dict],
    selected: set[int], kernel: np.ndarray
) -> list[dict]:
    """Append selected low-score masks without rebuilding fixed high-score masks."""
    output = list(baseline)
    claimed = claimed_baseline.copy()
    for index in sorted(selected, key=lambda i: entries[i]["score"], reverse=True):
        mask = eroded_mask(entries[index], kernel).astype(bool)
        mask[claimed] = False
        if int(mask.sum()) < 300:
            continue
        output.append(encode(mask))
        claimed |= mask
    return output


def components(binary: np.ndarray, min_area: int, min_eccentricity: float) -> list[np.ndarray]:
    count, labels, stats, _ = cv2.connectedComponentsWithStats(binary.astype(np.uint8), 8)
    result = []
    for label in range(1, count):
        area = int(stats[label, cv2.CC_STAT_AREA])
        if area < min_area:
            continue
        mask = (labels == label).astype(np.uint8)
        moments = cv2.moments(mask, binaryImage=True)
        covariance = np.array([[moments["mu20"], moments["mu11"]],
                               [moments["mu11"], moments["mu02"]]], dtype=np.float64)
        covariance /= max(moments["m00"], 1.0)
        values = np.maximum(np.linalg.eigvalsh(covariance), 0.0)
        eccentricity = float(np.sqrt(1.0 - values[0] / max(values[1], 1e-9)))
        if eccentricity >= min_eccentricity:
            result.append(mask)
    return result


def component_entries(binary: np.ndarray, min_area: int = 150) -> list[dict]:
    """Encode connected components once with reusable shape statistics."""
    count, labels, stats, _ = cv2.connectedComponentsWithStats(binary.astype(np.uint8), 8)
    result = []
    for label in range(1, count):
        area = int(stats[label, cv2.CC_STAT_AREA])
        if area < min_area:
            continue
        mask = (labels == label).astype(np.uint8)
        moments = cv2.moments(mask, binaryImage=True)
        covariance = np.array([[moments["mu20"], moments["mu11"]],
                               [moments["mu11"], moments["mu02"]]], dtype=np.float64)
        covariance /= max(moments["m00"], 1.0)
        values = np.maximum(np.linalg.eigvalsh(covariance), 0.0)
        eccentricity = float(np.sqrt(1.0 - values[0] / max(values[1], 1e-9)))
        result.append(dict(rle=encode(mask), area=area, eccentricity=eccentricity))
    return result


def score_cache(cache: dict[str, list[dict]], records: dict) -> dict:
    rows = []
    for name, passes in records.items():
        for image_id, truth in passes:
            rows.append(dict(pq_rle(cache[name], truth), image_id=image_id, photograph=name))
    return aggregate_pq(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--images", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--support-only", action="store_true")
    parser.add_argument("--candidate-audit", type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    cv2.setNumThreads(1)

    if torch.cuda.is_available():
        device = torch.device("cuda")
        accelerator = torch.cuda.get_device_name(0)
    else:
        import torch_xla.core.xla_model as xm
        device = xm.xla_device()
        accelerator = str(device)
    print("ACCELERATOR", accelerator, flush=True)
    model = build_model()
    state = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    model = model.to(device).eval()

    raw = json.loads(args.candidates.read_text())
    records, split = records_for(args.annotations)
    assert set(raw) == set(records)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))

    baseline, support, novel_pools = {}, {}, {}
    probability_thresholds = (.30, .40, .50, .60, .70)
    for position, name in enumerate(sorted(records), 1):
        probability = predict_probability(model, args.images / name, device)
        baseline[name] = emit_yolo(raw[name], set(), kernel)
        rows = []
        for index, entry in enumerate(raw[name]):
            if entry["score"] >= .35:
                continue
            mask = mu.decode(entry["rle"])
            small_mask = cv2.resize(mask, (1024, 1024), interpolation=cv2.INTER_NEAREST).astype(bool)
            values = probability[small_mask]
            rows.append(dict(index=index, model_score=float(entry["score"]),
                             semantic_mean=float(values.mean()) if len(values) else 0.0,
                             semantic_p75=float(np.percentile(values, 75)) if len(values) else 0.0))
        support[name] = rows

        if not args.support_only:
            base_union = np.zeros((2048, 2048), np.uint8)
            for rle in baseline[name]:
                base_union |= mu.decode(rle)
            exclusion = cv2.dilate(base_union, np.ones((5, 5), np.uint8))
            pools = {}
            full = cv2.resize(probability, (2048, 2048), interpolation=cv2.INTER_LINEAR)
            for threshold in probability_thresholds:
                novel = ((full >= threshold) & (exclusion == 0)).astype(np.uint8)
                pools[str(threshold)] = component_entries(novel)
            novel_pools[name] = pools
        if position % 10 == 0:
            print(f"INFERENCE {position}/{len(records)}", flush=True)

    baseline_metric = score_cache(baseline, records)
    assert abs(baseline_metric["pq"] - .45586386872921264) < 1e-8, baseline_metric
    baseline_claimed = {
        name: (mu.decode(mu.merge(rows)).astype(bool) if rows
               else np.zeros((2048, 2048), dtype=bool))
        for name, rows in baseline.items()
    }

    results = [dict(family="baseline", rule={}, additions=0, **baseline_metric)]
    prediction_caches = {"baseline": baseline}

    # Weak YOLO masks are admitted only when the independently trained dense
    # model assigns high semantic probability to their pixels.
    audit_features = None
    if args.candidate_audit:
        audit_features = json.loads(args.candidate_audit.read_text())["features"]
    support_rules, seen_signatures = [], set()
    for statistic in ("semantic_mean", "semantic_p75"):
        for confidence in (.14, .17, .20, .23, .26, .29, .32):
            for semantic in (.30, .40, .50, .60, .70, .80):
                selected_by_name = {
                    name: {row["index"] for row in support[name]
                           if row["model_score"] >= confidence
                           and row[statistic] >= semantic}
                    for name in sorted(records)
                }
                signature = tuple((name, *sorted(selected))
                                  for name, selected in selected_by_name.items())
                if signature in seen_signatures:
                    continue
                seen_signatures.add(signature)
                key = f"support_{statistic}_c{confidence:.2f}_s{semantic:.2f}"
                approximate_utility = 0.0
                if audit_features is not None:
                    for name, selected in selected_by_name.items():
                        rows = {row["index"]: row for row in audit_features[name]}
                        approximate_utility += sum(
                            rows[index]["iou_sum"]
                            - .5 * (rows[index]["record_count"] - rows[index]["hit_records"])
                            for index in selected
                        )
                support_rules.append(dict(
                    key=key, statistic=statistic, confidence=confidence,
                    semantic=semantic, selected=selected_by_name,
                    additions=sum(map(len, selected_by_name.values())),
                    approximate_utility=approximate_utility,
                ))

    if audit_features is not None:
        support_rules.sort(key=lambda row: row["approximate_utility"], reverse=True)
        support_rules = support_rules[:16]
    for rule_row in support_rules:
                selected_by_name = rule_row["selected"]
                key = rule_row["key"]
                cache, additions = {}, 0
                for name in sorted(records):
                    selected = selected_by_name[name]
                    additions += len(selected)
                    cache[name] = emit_with_baseline(
                        baseline[name], baseline_claimed[name], raw[name], selected, kernel
                    )
                metric = score_cache(cache, records)
                prediction_caches[key] = cache
                results.append(dict(family="semantic_support", key=key,
                                    rule=dict(confidence=rule_row["confidence"],
                                              statistic=rule_row["statistic"],
                                              semantic=rule_row["semantic"]),
                                    additions=additions,
                                    approximate_utility=rule_row["approximate_utility"],
                                    **metric))
                print("SCORED", key, metric["pq"], flush=True)

    # Independent semantic regions are restricted to pixels outside accepted
    # YOLO instances, then filtered globally for area and elongation.
    for threshold in (() if args.support_only else probability_thresholds):
        for min_area in (300, 600, 1200, 2400):
            for eccentricity in (.0, .75, .90):
                cache, additions = {}, 0
                for name in sorted(records):
                    accepted = [mu.decode(row["rle"])
                                for row in novel_pools[name][str(threshold)]
                                if row["area"] >= min_area
                                and row["eccentricity"] >= eccentricity]
                    additions += len(accepted)
                    candidates = [(2.0, mu.decode(rle)) for rle in baseline[name]]
                    candidates.extend((1.0, mask) for mask in accepted)
                    cache[name] = [encode(mask) for _, mask, _ in
                                   paint_panoptic(candidates, min_area=300)]
                metric = score_cache(cache, records)
                key = f"novel_t{threshold:.2f}_a{min_area}_e{eccentricity:.2f}"
                prediction_caches[key] = cache
                results.append(dict(family="semantic_novel", key=key,
                                    rule=dict(threshold=threshold, min_area=min_area,
                                              eccentricity=eccentricity),
                                    additions=additions, **metric))
                print("SCORED", key, metric["pq"], flush=True)

    results.sort(key=lambda row: row["pq"], reverse=True)
    best_key = results[0].get("key", "baseline")
    report = dict(status="complete", split=split.summary(), baseline=baseline_metric,
                  best=results[0], top=results[:20], checkpoint=str(args.checkpoint),
                  interpretation=("Frozen dense checkpoint used only as independent semantic "
                                  "support or for non-overlapping proposals; no training."))
    (args.output / "report.json").write_text(json.dumps(report, indent=2))
    (args.output / "support.json").write_text(json.dumps(support))
    (args.output / "best_predictions.json").write_text(json.dumps(prediction_caches[best_key]))
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
