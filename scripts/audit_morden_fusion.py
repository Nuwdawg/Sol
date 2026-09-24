"""Use exp085 fields to rescore and reshape the retained pretrained D4 instances.

The canonical labels are used only for choosing a rule.  Test deployment must
reproduce the chosen image-only rule exactly.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from pycocotools import mask as mask_util

from experiments.exp_004_spine_tpu.src.decompose import _nearest_seed_labels
from experiments.exp_033_inference.src.sweep import encode, pq_rle, records_for
from shared.utils import aggregate_pq


BASELINE_PQ = 0.45586386872921264
NATIVE = 2048


def probabilities(directory: Path, name: str) -> tuple[np.ndarray, np.ndarray]:
    stem = Path(name).stem
    mask = cv2.imread(str(directory / f"{stem}_mask.png"), cv2.IMREAD_GRAYSCALE)
    spine = cv2.imread(str(directory / f"{stem}_spine.png"), cv2.IMREAD_GRAYSCALE)
    if mask is None or spine is None or mask.shape != (NATIVE, NATIVE):
        raise ValueError(f"missing probability maps for {name}")
    return mask.astype(np.float32) / 255.0, spine.astype(np.float32) / 255.0


def score_cache(cache: dict[str, list[dict]], records: dict) -> dict:
    rows = []
    for name, passes in records.items():
        predicted = [entry["rle"] if "rle" in entry else entry for entry in cache[name]]
        rows.extend(pq_rle(predicted, truth) for _, truth in passes)
    return aggregate_pq(rows)


def mask_features(mask: np.ndarray, semantic: np.ndarray, spine: np.ndarray) -> dict:
    values = semantic[mask > 0]
    spine_values = spine[mask > 0]
    return {
        "semantic_mean": float(values.mean()) if values.size else 0.0,
        "semantic_p75": float(np.percentile(values, 75)) if values.size else 0.0,
        "semantic_coverage_035": float((values >= 0.35).mean()) if values.size else 0.0,
        "spine_mean": float(spine_values.mean()) if spine_values.size else 0.0,
        "spine_p90": float(np.percentile(spine_values, 90)) if spine_values.size else 0.0,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--probabilities", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    cv2.setNumThreads(1)
    output = args.output
    output.mkdir(parents=True, exist_ok=True)
    baseline = json.loads(args.baseline.read_text())
    candidates = json.loads(args.candidates.read_text())
    records, split = records_for(args.annotations)
    if set(baseline) != set(records) or set(candidates) != set(records):
        raise ValueError("prediction caches do not match canonical validation photos")
    baseline_metric = score_cache(baseline, records)
    if abs(baseline_metric["pq"] - BASELINE_PQ) > 1e-12:
        raise ValueError(f"baseline parity failed: {baseline_metric}")

    probability_cache = {name: probabilities(args.probabilities, name) for name in records}
    accepted_features: dict[str, list[dict]] = {}
    for name, entries in baseline.items():
        semantic, spine = probability_cache[name]
        accepted_features[name] = [
            {**entry, **mask_features(mask_util.decode(entry["rle"]), semantic, spine)}
            for entry in entries
        ]

    trials = [{"family": "baseline", "rule": {}, **baseline_metric}]

    # Precision audit: retain only accepted D4 objects with strong independent
    # evidence from the new solar-specific network.
    for feature, thresholds in {
        "semantic_mean": (0.10, 0.20, 0.30, 0.40, 0.50),
        "semantic_p75": (0.15, 0.25, 0.35, 0.45, 0.55),
        "semantic_coverage_035": (0.10, 0.25, 0.40, 0.55, 0.70),
        "spine_p90": (0.10, 0.20, 0.30, 0.40, 0.50),
    }.items():
        for threshold in thresholds:
            cache = {
                name: [entry for entry in entries if entry[feature] >= threshold]
                for name, entries in accepted_features.items()
            }
            trials.append({
                "family": "prune", "rule": {"feature": feature, "threshold": threshold},
                "retained": sum(map(len, cache.values())), **score_cache(cache, records),
            })

    # Boundary audit: the D4 objects are seeds/identity; exp085 supplies only a
    # semantic field.  Semantic components without any D4 seed are never added.
    for semantic_threshold in (0.25, 0.35, 0.45, 0.55):
        per_mode = {mode: {} for mode in ("semantic", "intersection", "union")}
        for name, entries in baseline.items():
            semantic, _ = probability_cache[name]
            binary = (semantic >= semantic_threshold).astype(np.uint8)
            binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
            count, components = cv2.connectedComponents(binary, connectivity=8)
            seeds = np.zeros_like(components, np.int32)
            base_masks = []
            for label, entry in enumerate(entries, start=1):
                base_mask = mask_util.decode(entry["rle"]).astype(np.uint8)
                base_masks.append(base_mask)
                seeds[base_mask > 0] = label
            if entries:
                touched = np.unique(components[seeds > 0])
                touched = touched[touched > 0]
                allowed = np.isin(components, touched)
                assigned = _nearest_seed_labels(allowed.astype(np.uint8), seeds)
            else:
                assigned = np.zeros_like(components)
            for mode in per_mode:
                emitted = []
                for label, (entry, base_mask) in enumerate(zip(entries, base_masks), start=1):
                    grown = (assigned == label).astype(np.uint8)
                    if mode == "semantic":
                        changed = grown
                    elif mode == "intersection":
                        changed = base_mask & binary
                    else:
                        changed = base_mask | grown
                    if int(changed.sum()) < 300:
                        changed = base_mask
                    emitted.append({"score": entry["score"], "rle": encode(changed)})
                per_mode[mode][name] = emitted
        for mode, cache in per_mode.items():
            trials.append({
                "family": "boundary", "rule": {"mode": mode, "threshold": semantic_threshold},
                **score_cache(cache, records),
            })

    # Recall audit: weak D4 proposals may enter only when exp085 independently
    # supports their mask or supplied-spine morphology. Baseline masks always
    # paint first and retain their exact bytes.
    erode = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    weak: dict[str, list[dict]] = {}
    for name, entries in candidates.items():
        semantic, spine = probability_cache[name]
        accepted_rles = {entry["rle"]["counts"] if isinstance(entry["rle"]["counts"], str) else entry["rle"]["counts"].decode() for entry in baseline[name]}
        rows = []
        for entry in entries:
            if float(entry["score"]) >= 0.35:
                continue
            mask = cv2.erode(mask_util.decode(entry["rle"]), erode)
            if int(mask.sum()) < 300:
                continue
            rows.append({**entry, "mask": mask, **mask_features(mask, semantic, spine)})
        weak[name] = rows

    support_rules = []
    for confidence in (0.20, 0.23, 0.26, 0.29, 0.32):
        for semantic_mean in (0.30, 0.40, 0.50, 0.60, 0.70):
            support_rules.append({"confidence": confidence, "semantic_mean": semantic_mean})
        for spine_p90 in (0.25, 0.35, 0.45, 0.55, 0.65):
            support_rules.append({"confidence": confidence, "spine_p90": spine_p90})
    for rule in support_rules:
        cache = {}
        additions = 0
        for name in records:
            claimed = np.zeros((NATIVE, NATIVE), np.uint8)
            emitted = list(baseline[name])
            for entry in baseline[name]:
                claimed |= mask_util.decode(entry["rle"]).astype(np.uint8)
            selected = [row for row in weak[name] if row["score"] >= rule["confidence"]]
            criterion = "semantic_mean" if "semantic_mean" in rule else "spine_p90"
            selected = [row for row in selected if row[criterion] >= rule[criterion]]
            for row in sorted(selected, key=lambda item: -item["score"]):
                remaining = (row["mask"] & (1 - claimed)).astype(np.uint8)
                if int(remaining.sum()) < 300:
                    continue
                claimed |= remaining
                emitted.append({"score": row["score"], "rle": encode(remaining)})
                additions += 1
            cache[name] = emitted
        trials.append({"family": "support", "rule": rule, "additions": additions,
                       **score_cache(cache, records)})

    trials.sort(key=lambda row: row["pq"], reverse=True)
    report = {
        "status": "complete",
        "split": split.summary(),
        "baseline": baseline_metric,
        "best": trials[0],
        "top": trials[:25],
        "family_best": {
            family: next(row for row in trials if row["family"] == family)
            for family in sorted({row["family"] for row in trials})
        },
    }
    (output / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    (output / "accepted_features.json").write_text(json.dumps(accepted_features), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
