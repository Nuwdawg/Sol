"""The exp098 evaluator must reproduce the 0.39 pipeline's validation score.

`evaluate_checkpoints.py` refuses to score any new checkpoint unless the stored
exp002 D4 pool, emitted at confidence 0.35 with one pixel of erosion and a
300-pixel area floor, reproduces the exp062 baseline through the same code path.
This test runs that exact check locally on CPU, so a Kaggle GPU session is never
spent discovering that the comparison itself is broken.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
POOL = ROOT / "data" / "exp033_cached_tta" / "candidates_0.5_0.5.json"
ANNOTATIONS = ROOT / "data" / "MAGFiLO_1.0_Annotations_kaggle2026_train.json"

pytestmark = pytest.mark.skipif(not (POOL.exists() and ANNOTATIONS.exists()),
                                reason="stored D4 pool or annotations are not present")


def test_evaluator_reproduces_the_exp062_baseline():
    from experiments.exp_033_inference.src.sweep import evaluate_thresholds, records_for
    from experiments.exp_097_runpod_yolo.src.evaluate_checkpoints import BASELINE_PQ

    records, _ = records_for(ANNOTATIONS)
    pool = json.loads(POOL.read_text())
    row = evaluate_thresholds(pool, records, (0.35,), grow=-1, min_area=300)[0]
    assert row["pq"] == pytest.approx(BASELINE_PQ, abs=1e-9)
    assert (row["tp"], row["fp"], row["fn"]) == (860, 397, 465)
