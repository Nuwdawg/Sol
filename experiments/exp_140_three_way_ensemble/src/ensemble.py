"""exp140: does Mask2Former (exp139) add to the exp119 YOLO ensemble? Validation first.

Sources, each as instance pools {photograph: [{score, rle}]}:
- exp002: YOLO D4 instances (cached validation pool, cached raw test views);
- exp097: YOLOv8l-2048 D4 instances (cached validation pool; test views predicted here);
- exp139: Mask2Former epoch 40, single view, float32 (validation PQ 0.47052).

Why scores are calibrated: the sources score on different scales. exp139's best cut is
0.85, the YOLO pair's 0.35. exp119's cross-model fusion averages member scores, so an
uncalibrated mix lets one scale dominate. Each source's score is mapped to the
probability the metric asks about (CLAUDE.md section 4): the share of the photograph's
annotator records that hold a truth mask with IoU > 0.5 to the candidate, after that
source's own validated erosion. The map is isotonic (monotone), fitted on validation.
For the validation numbers it is CROSS-FITTED over two folds of photographs, so no
candidate's score comes from a map fitted on its own photograph. For the test, one map
per source is fitted on all validation photographs.

Fusion is exp119's own `fuse_sources` (link 0.5, pixel vote 0.5, score x support), and
emission is exp033's `evaluate_thresholds` / exp119's `emit`, so every number is on the
same scoring path as exp119's 0.468146 and exp139's 0.47052. Both are replayed first.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
from pycocotools import mask as mu

from experiments.exp_033_inference.src.cached_tta import fuse
from experiments.exp_033_inference.src.sweep import encode, evaluate_thresholds, pq_rle, records_for
from experiments.exp_097_runpod_yolo.src.evaluate_ensemble import fuse_sources
from shared.utils import aggregate_pq, paint_panoptic

NATIVE = 2048
KERNEL = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
EROSION = {"exp002": 1, "exp097": 1, "exp139": 0}          # each source's validated best
CONFS = tuple(round(0.05 * i, 2) for i in range(1, 20))
EMISSION = [(grow, area) for grow in (0, -1) for area in (100, 300)]
EXP119_PQ, EXP119_RULE = 0.468146, dict(conf=0.35, grow=-1, min_area=100)
VARIANTS = {
    "raw_three": dict(sources=("exp002", "exp097", "exp139"), calibrated=False, pre_erode=False),
    "cal_three": dict(sources=("exp002", "exp097", "exp139"), calibrated=True, pre_erode=False),
    "cal_three_pre": dict(sources=("exp002", "exp097", "exp139"), calibrated=True, pre_erode=True),
    "cal_097_139_pre": dict(sources=("exp097", "exp139"), calibrated=True, pre_erode=True),
}


# ------------------------------------------------------------------ helpers ---
def as_text(rle: dict) -> dict:
    counts = rle["counts"]
    return dict(size=rle["size"], counts=counts.decode("ascii") if isinstance(counts, bytes) else counts)


def eroded(rle: dict, iterations: int) -> dict:
    if iterations <= 0:
        return as_text(rle)
    return encode(cv2.erode(mu.decode(as_text(rle)), KERNEL, iterations=iterations))


def instances_from_views(views) -> list[dict]:
    """One YOLO model's eight dihedral views -> fused D4 instances (exp119's call)."""
    return fuse([(float(score), as_text(rle)) for score, rle in views], 0.5, 0.5)


def emit(entries: list[dict], conf: float, grow: int, min_area: int) -> list[dict]:
    """exp033 sweep's emission, operation for operation."""
    candidates = []
    for entry in entries:
        if entry["score"] < conf:
            continue
        mask = mu.decode(as_text(entry["rle"]))
        if grow < 0:
            mask = cv2.erode(mask, KERNEL, iterations=-grow)
        elif grow != 0:
            raise ValueError(grow)
        candidates.append((entry["score"], mask))
    return [encode(mask) for _, mask, _ in paint_panoptic(candidates, min_area=min_area)]


# -------------------------------------------------------------- calibration ---
def isotonic_fit(scores: np.ndarray, labels: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Weighted pool-adjacent-violators, increasing.

    Equal scores are pooled first, so block centres are strictly increasing and
    `np.interp` between them is well defined. Returns (centres, means).
    """
    values, inverse, counts = np.unique(np.asarray(scores, float), return_inverse=True, return_counts=True)
    sums = np.bincount(inverse, weights=np.asarray(labels, float))
    blocks: list[list[float]] = []            # [sum_label, weight, sum_score_weighted]
    for x, s, c in zip(values, sums, counts):
        blocks.append([s, float(c), x * c])
        while len(blocks) > 1 and blocks[-2][0] / blocks[-2][1] > blocks[-1][0] / blocks[-1][1]:
            s2, c2, sx2 = blocks.pop()
            blocks[-1][0] += s2
            blocks[-1][1] += c2
            blocks[-1][2] += sx2
    centres = np.array([b[2] / b[1] for b in blocks])
    means = np.array([b[0] / b[1] for b in blocks])
    return centres, means


def isotonic_apply(model: tuple[np.ndarray, np.ndarray], scores: np.ndarray) -> np.ndarray:
    centres, means = model
    if len(centres) == 1:
        return np.full(len(scores), means[0])
    return np.interp(scores, centres, means)


def candidate_labels(pool: dict, records: dict, iterations: int) -> dict[str, np.ndarray]:
    """Share of each photograph's records with a truth mask at IoU > 0.5 to each candidate."""
    labels = {}
    for name, passes in records.items():
        entries = pool.get(name, [])
        if not entries:
            labels[name] = np.zeros(0)
            continue
        rles = [eroded(entry["rle"], iterations) for entry in entries]
        hits = np.zeros(len(rles))
        for _, truth in passes:
            if truth:
                ious = np.asarray(mu.iou(rles, truth, [0] * len(truth)))
                hits += (ious.max(axis=1) > 0.5)
        labels[name] = hits / max(1, len(passes))
    return labels


def fit_calibrator(pool: dict, labels: dict, names: list[str]):
    scores = np.array([entry["score"] for name in names for entry in pool.get(name, [])])
    values = np.concatenate([labels[name] for name in names if len(labels[name])]) if scores.size else np.zeros(0)
    if scores.size == 0:
        raise SystemExit("no candidates to calibrate on")
    return isotonic_fit(scores, values)


def recalibrated(pool: dict, model, names: list[str]) -> dict:
    out = {}
    for name in names:
        entries = pool.get(name, [])
        scores = isotonic_apply(model, np.array([entry["score"] for entry in entries])) if entries else []
        out[name] = [dict(score=float(s), rle=entry["rle"]) for s, entry in zip(scores, entries)]
    return out


# ------------------------------------------------------------------- fusion ---
def fused_cache(pools: dict, names: list[str], variant: dict, link: float = 0.5, vote: float = 0.5) -> dict:
    cache = {}
    for name in names:
        entries = []
        for source in variant["sources"]:
            iterations = EROSION[source] if variant["pre_erode"] else 0
            entries.extend((float(entry["score"]), eroded(entry["rle"], iterations))
                           for entry in pools[source].get(name, []))
        cache[name] = [dict(score=e["score"], rle=as_text(e["rle"]))
                       for e in fuse_sources(entries, link, vote, len(variant["sources"]))]
    return cache


def sweep(cache: dict, records: dict, emission=None) -> list[dict]:
    rows = []
    for grow, area in (emission or EMISSION):
        rows.extend(evaluate_thresholds(cache, records, CONFS, grow=grow, min_area=area))
    return sorted(rows, key=lambda row: -row["pq"])


def load_pool(path: Path) -> dict:
    return {Path(k).name: v for k, v in json.loads(Path(path).read_text()).items()}


# --------------------------------------------------------------------- main ---
def validate(args) -> dict:
    records, split = records_for(args.annotations)
    if args.limit:
        records = {name: records[name] for name in sorted(records)[:args.limit]}
    names = sorted(records)
    pools = dict(exp002=load_pool(args.exp002_val_pool), exp097=load_pool(args.exp097_val_pool),
                 exp139=load_pool(args.exp139_val_cache))
    for source, pool in pools.items():
        missing = [name for name in names if name not in pool]
        if missing:
            raise SystemExit(f"{source} validation pool misses {len(missing)} photographs: {missing[:3]}")
    report = dict(split=split.summary())

    # 1. Replays: the scoring path must reproduce both reference numbers.
    rows = []
    for name, passes in records.items():
        fused = fuse_sources([(float(e["score"]), as_text(e["rle"])) for s in ("exp002", "exp097")
                              for e in pools[s][name]], 0.5, 0.5, 2)
        predicted = emit(fused, **EXP119_RULE)
        rows.extend(pq_rle(predicted, truth) for _, truth in passes)
    replay_119 = aggregate_pq(rows)
    report["replay_exp119"] = replay_119
    print("REPLAY_EXP119 " + json.dumps(replay_119), flush=True)
    if abs(replay_119["pq"] - EXP119_PQ) > 5e-4:
        raise SystemExit(f"exp119 replay {replay_119['pq']} differs from {EXP119_PQ}")
    alone = sweep(pools["exp139"], records)[0]
    report["replay_exp139"] = alone
    print("REPLAY_EXP139 " + json.dumps(alone), flush=True)

    # 2. Labels and cross-fitted calibration (two folds of photographs).
    labels = {source: candidate_labels(pool, records, EROSION[source]) for source, pool in pools.items()}
    folds = [names[0::2], names[1::2]]
    calibrated = {source: {} for source in pools}
    for index, fold in enumerate(folds):
        other = folds[1 - index]
        for source, pool in pools.items():
            model = fit_calibrator(pool, labels[source], other)
            calibrated[source].update(recalibrated(pool, model, fold))
    report["calibration"] = {}
    for source, pool in pools.items():
        centres, means = fit_calibrator(pool, labels[source], names)
        report["calibration"][source] = dict(
            candidates=int(sum(len(pool[n]) for n in names)),
            mean_label=float(np.mean(np.concatenate([labels[source][n] for n in names if len(labels[source][n])]))),
            curve=[(round(float(c), 3), round(float(m), 3)) for c, m in zip(centres[::max(1, len(centres) // 12)],
                                                                            means[::max(1, len(means) // 12)])])

    # 3. Variants.
    results = {}
    for label, variant in VARIANTS.items():
        source_pools = calibrated if variant["calibrated"] else pools
        cache = fused_cache(source_pools, names, variant)
        best = sweep(cache, records)[0]
        results[label] = best
        print(f"VARIANT {label} " + json.dumps(best), flush=True)
    report["variants"] = results
    winner = max(results, key=lambda key: results[key]["pq"])
    references = dict(exp119=replay_119["pq"], exp139=alone["pq"])
    # exp139 alone scored public 0.41 (submission 56309562), above exp119's 0.39, and
    # transferred better than the YOLO fit predicted. The combination must therefore beat
    # exp139 itself on validation by a margin, not merely a fixed YOLO-derived line.
    passes_gate = results[winner]["pq"] >= alone["pq"] + args.margin
    report["decision"] = dict(variant=winner, rule=results[winner], margin=args.margin,
                              references=references, emit_test=bool(passes_gate))
    print("DECISION " + json.dumps(report["decision"]), flush=True)
    return report


def emit_test(args, report: dict) -> dict:
    """Test submission for the winning variant, with calibration fitted on all validation."""
    from experiments.exp_007_tta.src.dihedral import predict_views
    from ultralytics import YOLO
    records, _ = records_for(args.annotations)
    names = sorted(records)
    decision = report["decision"]
    variant = VARIANTS[decision["variant"]]
    rule = decision["rule"]
    val = dict(exp002=load_pool(args.exp002_val_pool), exp097=load_pool(args.exp097_val_pool),
               exp139=load_pool(args.exp139_val_cache))
    models = {}
    for source in variant["sources"]:
        labels = candidate_labels(val[source], records, EROSION[source])
        models[source] = fit_calibrator(val[source], labels, names)

    # Replay on validation with the all-photograph maps (optimistic next to the cross-fitted number).
    pools = {s: (recalibrated(val[s], models[s], names) if variant["calibrated"] else val[s]) for s in variant["sources"]}
    cache = fused_cache(pools, names, variant)
    rows = []
    for name, passes in records.items():
        predicted = emit(cache[name], rule["conf"], rule["grow"], rule["min_area"])
        rows.extend(pq_rle(predicted, truth) for _, truth in passes)
    in_sample = aggregate_pq(rows)
    print("IN_SAMPLE_VALIDATION " + json.dumps(in_sample), flush=True)

    test = sorted(args.test_images.glob("*.jpeg"))
    if len(test) != 180:
        raise SystemExit(f"expected 180 test photographs, found {len(test)}")
    test_pools = {s: {} for s in variant["sources"]}
    if "exp002" in variant["sources"]:
        views = {}
        for path in args.exp002_test_views:
            for name, entries in json.loads(Path(path).read_text()).items():
                views.setdefault(Path(name).name, []).extend(entries)
        test_pools["exp002"] = {p.name: instances_from_views(views[p.name]) for p in test}
    if "exp097" in variant["sources"]:
        model = YOLO(str(args.exp097_weights))
        for position, path in enumerate(test, start=1):
            test_pools["exp097"][path.name] = instances_from_views(predict_views(model, path, NATIVE, 0.25, 100))
            if position % 30 == 0:
                print(f"EXP097_TEST {position}/180", flush=True)
    test_pools["exp139"] = load_pool(args.exp139_test_cache)
    if variant["calibrated"]:
        test_pools = {s: recalibrated(test_pools[s], models[s], [p.name for p in test]) for s in variant["sources"]}
    fused = fused_cache(test_pools, [p.name for p in test], variant)

    destination = args.output / "submission.csv"
    written, contributing = 0, set()
    with destination.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["filament_id", "segmentation_rle"])
        for path in test:
            claimed = np.zeros((NATIVE, NATIVE), np.uint8)
            for index, rle in enumerate(emit(fused[path.name], rule["conf"], rule["grow"], rule["min_area"]), start=1):
                mask = mu.decode(rle)
                if not mask.sum():
                    raise SystemExit(f"empty instance emitted for {path.stem}")
                if (claimed & mask).any():
                    raise SystemExit(f"overlapping instances emitted for {path.stem}")
                claimed |= mask
                writer.writerow([f"{path.stem}_{index}", rle["counts"]])
                written += 1
                contributing.add(path.stem)
    return dict(rows=written, contributing_images=len(contributing), in_sample_validation=in_sample,
                sha256=hashlib.sha256(destination.read_bytes()).hexdigest().upper())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--exp002-val-pool", type=Path, required=True)
    parser.add_argument("--exp097-val-pool", type=Path, required=True)
    parser.add_argument("--exp139-val-cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--margin", type=float, default=0.005,
                        help="validation PQ the combination must add over exp139 alone")
    parser.add_argument("--test-images", type=Path, default=None)
    parser.add_argument("--exp002-test-views", type=Path, nargs="*", default=[])
    parser.add_argument("--exp097-weights", type=Path, default=None)
    parser.add_argument("--exp139-test-cache", type=Path, default=None)
    parser.add_argument("--skip-exp119-replay", action="store_true", help="local smoke tests only")
    parser.add_argument("--limit", type=int, default=0, help="first N validation photographs, smoke tests only")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    if args.skip_exp119_replay:
        global EXP119_PQ
        EXP119_PQ = float("nan")
    report = validate(args)
    (args.output / "ensemble_report.json").write_text(json.dumps(report, indent=2))
    if report["decision"]["emit_test"] and args.test_images is not None:
        report["test"] = emit_test(args, report)
        (args.output / "ensemble_report.json").write_text(json.dumps(report, indent=2))
        print("TEST " + json.dumps(report["test"]), flush=True)


if __name__ == "__main__":
    main()
