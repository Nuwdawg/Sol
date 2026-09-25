"""Unit tests for the exp097 preflight decision (pure logic, no GPU).

The preflight picks the fastest configuration that fits. Utilization is a
diagnostic: it must never block training.
"""
from __future__ import annotations

import pytest

from experiments.exp_097_runpod_yolo.src.preflight import project, select


def candidate(batch, ips, util, peak, total=46068.0, wait=0.05, oom=False):
    return dict(batch=batch, images_per_second=ips, gpu_utilization_mean=util,
                peak_reserved_mib=peak, total_mib=total, data_wait_fraction=wait, oom=oom)


def test_picks_the_fastest_batch():
    results = [candidate(2, 5.0, 80, 12000), candidate(4, 6.00, 90, 20000),
               candidate(6, 5.70, 92, 30000), candidate(8, 5.50, 93, 38000)]
    decision = select(results)
    assert decision["ok"] and decision["choice"]["batch"] == 4


def test_larger_batch_wins_only_a_near_tie():
    results = [candidate(4, 6.00, 90, 20000), candidate(6, 5.95, 92, 30000)]
    assert select(results)["choice"]["batch"] == 6          # within 1%: noise-level tie
    results = [candidate(4, 6.00, 90, 20000), candidate(6, 5.80, 92, 30000)]
    assert select(results)["choice"]["batch"] == 4          # 3.3% slower: speed wins


def test_excludes_oom_and_over_headroom_candidates():
    results = [candidate(4, 6.0, 90, 20000), candidate(8, 9.0, 95, 45000),
               candidate(12, 0.0, 0, 46000, oom=True)]
    assert select(results, headroom=0.85)["choice"]["batch"] == 4


def test_low_utilization_never_blocks_training_but_is_explained():
    input_bound = select([candidate(8, 4.0, 45, 30000, wait=0.4)])
    assert input_bound["ok"] and input_bound["choice"]["batch"] == 8
    assert "input pipeline limits speed" in input_bound["reason"]
    gpu_bound = select([candidate(8, 4.0, 55, 30000, wait=0.05)])
    assert gpu_bound["ok"] and "GPU itself limits speed" in gpu_bound["reason"]


def test_only_a_total_memory_failure_is_not_ok():
    decision = select([candidate(2, 0.0, 0, 46000, oom=True)])
    assert decision == dict(ok=False, choice=None,
                            reason="no candidate batch size completed within the memory headroom")


def test_projection_arithmetic():
    projection = project(dict(images_per_second=5.0), train_images=1000, price_per_hour=0.79,
                         validation_overhead=0.25)
    assert projection["epoch_train_seconds"] == pytest.approx(200.0)
    assert projection["epoch_seconds_estimate"] == pytest.approx(250.0)
    assert projection["usd_per_epoch_estimate"] == pytest.approx(250 / 3600 * 0.79, abs=1e-4)


def test_real_l40_measurement_is_accepted_by_the_default_headroom():
    """Regression: pod y2cc247f9sq3b5 measured batch 2 at 86.7% of memory and a 0.85
    headroom rejected the only batch that fits a 48 GB card."""
    measured = dict(batch=2, images_per_second=9.910729080506407, gpu_utilization_mean=None,
                    peak_reserved_mib=39418.0, total_mib=45468.5,
                    data_wait_fraction=0.0017102134512629096, oom=False)
    decision = select([measured])
    assert decision["ok"] and decision["choice"]["batch"] == 2
    assert not select([measured], headroom=0.85)["ok"]


def test_every_candidate_measures_enough_images():
    from experiments.exp_097_runpod_yolo.src.preflight import measure_batches_for
    assert measure_batches_for(1, 20, 240) == 240
    assert measure_batches_for(2, 20, 240) == 120
    assert measure_batches_for(24, 20, 240) == 20
