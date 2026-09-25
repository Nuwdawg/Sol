"""The submission emitter must paint exactly what the sweep scored.

exp098 chooses a rule by sweeping confidence, erosion and minimum area with
`sweep.evaluate`. `emit_exp097_submission.emit` then writes the test CSV under
that rule. If the two painting paths ever diverge, the deployed submission stops
being the thing that was validated, which is the failure CLAUDE.md section 5
records: a rule tuned one way and shipped another, caught only by counting rows.

This locks them together on real data: the stored exp002 D4 pool, the real
annotations, and the operating point behind the 0.39 submission.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from experiments.exp_033_inference.src.sweep import encode, evaluate, pq_rle, records_for
from scripts.emit_exp097_submission import emit
from shared.utils import aggregate_pq

ROOT = Path(__file__).resolve().parents[1]
POOL = ROOT / "data" / "exp033_cached_tta" / "candidates_0.5_0.5.json"
RULE = dict(conf=0.35, grow=-1, min_area=300)
BASELINE_PQ = 0.4558638687292126      # exp062: the 0.39 public submission's operating point


def test_emit_reproduces_the_swept_instances_and_pq():
    annotations = next(ROOT.rglob("MAGFiLO_1.0_Annotations_kaggle2026_train.json"), None)
    if annotations is None or not POOL.is_file():
        pytest.skip("competition annotations or the cached D4 pool are not present")
    pool = json.loads(POOL.read_text())
    records, _ = records_for(annotations)

    reference, _ = evaluate(pool, records, **RULE)
    rows = []
    for name, passes in records.items():
        predicted = emit(pool[name], **RULE)
        rows.extend(pq_rle(predicted, truth) for _, truth in passes)
    emitted = aggregate_pq(rows)

    assert (emitted["tp"], emitted["fp"], emitted["fn"]) == \
           (reference["tp"], reference["fp"], reference["fn"])
    assert emitted["pq"] == pytest.approx(reference["pq"], abs=1e-12)
    assert emitted["pq"] == pytest.approx(BASELINE_PQ, abs=1e-12)


def test_emit_rejects_dilation():
    """The guard lives inside the candidate loop, so it needs a candidate to fire."""
    mask = np.zeros((2048, 2048), np.uint8)
    mask[100:150, 100:150] = 1
    entry = dict(score=0.9, rle=encode(mask))
    with pytest.raises(ValueError):
        emit([entry], conf=0.35, grow=1, min_area=300)
    assert emit([entry], conf=0.95, grow=-1, min_area=300) == []   # below the confidence
