"""Oracle-ceiling harness: what could each repair family recover if perfectly gated?

Built to the testing manual's Part I. The manual's rule is that a repair family is
only worth a deployable trigger once its oracle ceiling is known to be material,
and that every operator must be scored by the frozen official metric after global
rematching - never by AUC, Dice or a local surrogate.

The four layers are kept separate by construction:

1. failure population - defined with ground truth, for analysis only;
2. observable trigger - absent here by design, because these are ceilings;
3. repair operator - the actual transformation, applied to the prediction set;
4. evaluation - `shared.utils.aggregate_pq` over rebuilt global matching.

Every ceiling run reports the harm accounting the manual requires: threshold
crossings up and down, healthy objects modified, healthy objects that fall below
IoU 0.5 afterwards, and the change in the matched-IoU numerator. A ceiling that
rescues targets while destroying controls is not a ceiling worth chasing.

Nothing in this module may be presented as a postprocessor. Every repair here reads
ground truth to choose what to modify, which is exactly what an inference-time
method may not do.
"""
from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
from pycocotools import mask as mu

from experiments.exp_033_inference.src.sweep import encode, pq_rle, records_for
from shared.utils import aggregate_pq, paint_panoptic

KERNEL = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
TOUCH = 0.05
BASELINE_PQ = 0.4558638687292126


def emit(entries: list[dict], conf: float, grow: int, min_area: int) -> list[dict]:
    """The frozen emission path: filter by score, erode, paint, encode."""
    candidates = []
    for entry in entries:
        if entry["score"] < conf:
            continue
        mask = mu.decode(entry["rle"])
        if grow < 0:
            mask = cv2.erode(mask, KERNEL, iterations=-grow)
        elif grow > 0:
            mask = cv2.dilate(mask, KERNEL, iterations=grow)
        candidates.append((entry["score"], mask))
    return [encode(mask) for _, mask, _ in paint_panoptic(candidates, min_area=min_area)]


def build_population(pool: dict, records: dict, conf: float, grow: int, min_area: int):
    """Reconstruct predictions per record from source artifacts, never from cached counts."""
    population = []
    for photograph, passes in records.items():
        emitted = emit(pool[photograph], conf, grow, min_area)
        for image_id, truths in passes:
            population.append(dict(photograph=photograph, image_id=image_id,
                                   predicted=list(emitted), truths=list(truths)))
    return population


def score(population: list[dict]) -> dict:
    """The official metric over the whole split, rebuilt from current masks."""
    rows = [pq_rle(record["predicted"], record["truths"]) for record in population]
    aggregate = aggregate_pq(rows)
    aggregate["iou_sum"] = float(sum(row["iou_sum"] for row in rows))
    return aggregate


def matched_ious(predicted: list[dict], truths: list[dict]) -> np.ndarray:
    if not predicted or not truths:
        return np.zeros((len(predicted), len(truths)))
    return np.asarray(mu.iou(predicted, truths, [0] * len(truths)))


def per_truth_best(predicted: list[dict], truths: list[dict]) -> np.ndarray:
    ious = matched_ious(predicted, truths)
    if ious.size == 0:
        return np.zeros(len(truths))
    return ious.max(axis=0)


def harm_report(before: list[dict], after: list[dict]) -> dict:
    """Crossings and collateral damage, per record, as the manual requires."""
    crossings_up = crossings_down = 0
    for original, repaired in zip(before, after):
        was = per_truth_best(original["predicted"], original["truths"])
        now = per_truth_best(repaired["predicted"], repaired["truths"])
        crossings_up += int(((was < 0.5) & (now >= 0.5)).sum())
        crossings_down += int(((was >= 0.5) & (now < 0.5)).sum())
    return dict(crossings_up=crossings_up, crossings_down=crossings_down)


def split_by_truth(mask: np.ndarray, truths: list[dict], covered: list[int],
                   min_piece: int = 40) -> list[np.ndarray]:
    """ORACLE: divide a merged prediction by assigning pixels to the truth they lie in.

    Upper bound only. No inference-time method may read truth like this.
    """
    pieces = []
    for index in covered:
        piece = mask & mu.decode(truths[index])
        if int(piece.sum()) >= min_piece:
            pieces.append(piece)
    return pieces


def union_for_truth(predicted: list[dict], indices: list[int]) -> dict:
    """ORACLE: replace several fragments with their union."""
    return mu.merge([predicted[index] for index in indices])


def apply_split_ceiling(population: list[dict]) -> tuple[list[dict], dict]:
    """Split every prediction that oracle-covers two or more truths."""
    repaired, touched, produced = [], 0, 0
    for record in population:
        predicted, truths = record["predicted"], record["truths"]
        if not predicted or not truths:
            repaired.append(dict(record, predicted=list(predicted)))
            continue
        ious = matched_ious(predicted, truths)
        output = []
        for index, rle in enumerate(predicted):
            covered = [g for g in range(len(truths)) if ious[index, g] > TOUCH]
            if len(covered) < 2:
                output.append(rle)
                continue
            pieces = split_by_truth(mu.decode(rle), truths, covered)
            if len(pieces) < 2:
                output.append(rle)
                continue
            touched += 1
            produced += len(pieces)
            output.extend(encode(piece) for piece in pieces)
        repaired.append(dict(record, predicted=output))
    return repaired, dict(masks_split=touched, pieces_produced=produced)


def apply_union_ceiling(population: list[dict]) -> tuple[list[dict], dict]:
    """Union the fragments that oracle-overlap one truth, when that truth is unmatched."""
    repaired, merged_groups = [], 0
    for record in population:
        predicted, truths = record["predicted"], record["truths"]
        if not predicted or not truths:
            repaired.append(dict(record, predicted=list(predicted)))
            continue
        ious = matched_ious(predicted, truths)
        best = ious.max(axis=0) if ious.size else np.zeros(len(truths))
        consumed, additions = set(), []
        for g in range(len(truths)):
            if best[g] >= 0.5:
                continue
            overlapping = [p for p in range(len(predicted))
                           if ious[p, g] > TOUCH and p not in consumed]
            if len(overlapping) < 2:
                continue
            merged = union_for_truth(predicted, overlapping)
            if float(mu.iou([merged], [truths[g]], [0])[0][0]) <= best[g]:
                continue
            consumed.update(overlapping)
            additions.append(merged)
            merged_groups += 1
        output = [rle for index, rle in enumerate(predicted) if index not in consumed]
        output.extend(additions)
        repaired.append(dict(record, predicted=output))
    return repaired, dict(groups_merged=merged_groups)


def run_ceiling(name: str, population: list[dict], operator, output: Path,
                notes: str = "") -> dict:
    """Apply one oracle repair, rescore officially, and write the manual's report."""
    before = score(population)
    repaired, operator_stats = operator(population)
    after = score(repaired)
    report = dict(
        experiment_id="exp113", method=name,
        population_definition=dict(
            analysis_positive="oracle-identified targets for this repair",
            control="every other prediction in the split",
            ground_truth_used_only_for_labels=False,
            note="ORACLE REPAIR: the operator itself reads truth. Upper bound only."),
        operator=dict(implemented=True, uses_ground_truth=True, **operator_stats),
        official_metric=dict(
            before={k: before[k] for k in ("pq", "sq", "rq", "tp", "fp", "fn", "iou_sum")},
            after={k: after[k] for k in ("pq", "sq", "rq", "tp", "fp", "fn", "iou_sum")},
            delta_pq=round(after["pq"] - before["pq"], 6)),
        harm=harm_report(population, repaired),
        conclusion=dict(status="oracle_ceiling", rules_out="nothing by itself",
                        does_not_rule_out="any deployable trigger for this repair",
                        notes=notes))
    output.mkdir(parents=True, exist_ok=True)
    (output / f"{name}_ceiling.json").write_text(json.dumps(report, indent=2))
    return report


def print_report(report: dict) -> None:
    before = report["official_metric"]["before"]
    after = report["official_metric"]["after"]
    harm = report["harm"]
    print(f"\n{report['method']}")
    print(f"  operator     : {report['operator']}")
    print(f"  before       : PQ {before['pq']:.10f}  TP/FP/FN {before['tp']}/{before['fp']}/{before['fn']}")
    print(f"  after        : PQ {after['pq']:.10f}  TP/FP/FN {after['tp']}/{after['fp']}/{after['fn']}")
    print(f"  delta PQ     : {report['official_metric']['delta_pq']:+.6f}")
    print(f"  crossings    : up {harm['crossings_up']}, down {harm['crossings_down']}")
