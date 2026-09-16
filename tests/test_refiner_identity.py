"""The refiner's repaint path must be exactly identity-preserving.

exp092 changes boundaries only. If a refined mask happens to equal its seed, the
canonical PQ must come back byte-identical to the retained baseline. Without
this, a measured PQ change could come from the repaint bookkeeping - claim
order, the area floor, the fallback - rather than from the refinement under
test, and every promotion decision would be measuring the wrong thing.

`CLAUDE.md` section 5 records the deployment that diverged from its validation
and was caught only by counting rows. This is the cheap invariant for exp092.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from pycocotools import mask as mu

from experiments.exp_033_inference.src.sweep import pq_rle, records_for
from experiments.exp_092_refiner.src.evaluate import BASELINE_PQ, repaint
from shared.utils import aggregate_pq

ROOT = Path(__file__).resolve().parents[1]
EMISSION = ROOT / "data/exp083_d4_consensus/rich_candidates.json"
ANNOTATIONS = ROOT / "data/MAGFiLO_1.0_Annotations_kaggle2026_train.json"

pytestmark = pytest.mark.skipif(
    not (EMISSION.exists() and ANNOTATIONS.exists()),
    reason="retained emission or annotations are not present")


@pytest.fixture(scope="module")
def emission() -> dict:
    return json.loads(EMISSION.read_text())


def test_identity_refinement_reproduces_the_baseline(emission):
    records, _ = records_for(ANNOTATIONS)
    assert set(emission) == set(records)
    metrics = []
    for name, passes in records.items():
        probabilities = [mu.decode(row["rle"]).astype(np.float32) for row in emission[name]]
        predicted = repaint(emission[name], probabilities, threshold=.5, guard=.0,
                            min_area=150)
        assert len(predicted) == len(emission[name]), name
        metrics.extend(pq_rle(predicted, truth) for _, truth in passes)
    assert aggregate_pq(metrics)["pq"] == pytest.approx(BASELINE_PQ, abs=1e-12)


def test_guard_falls_back_to_the_seed_when_refinement_disagrees(emission):
    """A refinement that shares no pixels with its seed must be rejected."""
    name = sorted(emission)[0]
    entries = emission[name]
    empty = [np.zeros((2048, 2048), np.float32) for _ in entries]
    predicted = repaint(entries, empty, threshold=.5, guard=.5, min_area=150)
    assert [row["counts"] for row in predicted] == [row["rle"]["counts"] for row in entries]
