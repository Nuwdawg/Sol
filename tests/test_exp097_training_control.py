"""Unit tests for the exp097 convergence controller, telemetry and GPU monitor.

These are sanity checks of the control logic, not evidence about training. The
decisions they pin (when a decay fires, when training stops, how waiting is
attributed) are exactly the ones that would silently waste paid GPU time if they
were wrong.
"""
from __future__ import annotations

import pytest

from experiments.exp_097_runpod_yolo.src.convergence import PlateauConfig, PlateauController
from experiments.exp_097_runpod_yolo.src.gpu_monitor import GPUMonitor, parse_line, percentile
from experiments.exp_097_runpod_yolo.src.telemetry import EpochTelemetry


def run(controller, values, start=1):
    return [controller.update(epoch, value) for epoch, value in enumerate(values, start=start)]


def test_steady_improvement_never_decays_or_stops():
    controller = PlateauController(PlateauConfig(patience=3, smoothing=1, warmup_epochs=0))
    decisions = run(controller, [0.1 + 0.01 * i for i in range(40)])
    assert {d.action for d in decisions} == {"continue"}
    assert controller.lr_scale == 1.0


def test_decay_fires_after_exactly_patience_flat_epochs():
    config = PlateauConfig(patience=4, min_delta=0.001, smoothing=1, warmup_epochs=0,
                           max_decays=2)
    controller = PlateauController(config)
    decisions = run(controller, [0.5] + [0.5] * 4)
    assert [d.action for d in decisions] == ["continue"] * 4 + ["decay"]
    assert controller.lr_scale == pytest.approx(0.2)
    assert controller.epochs_since_best == 0


def test_stop_only_after_max_decays_and_a_final_plateau():
    config = PlateauConfig(patience=2, smoothing=1, warmup_epochs=0, max_decays=2,
                           decay_factor=0.5)
    controller = PlateauController(config)
    actions = [d.action for d in run(controller, [0.3] * 7)]
    assert actions == ["continue", "continue", "decay", "continue", "decay",
                       "continue", "stop"]
    assert controller.lr_scale == pytest.approx(0.25)


def test_improvement_after_decay_resets_patience():
    config = PlateauConfig(patience=2, smoothing=1, warmup_epochs=0, max_decays=1)
    controller = PlateauController(config)
    actions = [d.action for d in run(controller, [0.3, 0.3, 0.3, 0.4, 0.4, 0.4])]
    assert actions == ["continue", "continue", "decay", "continue", "continue", "stop"]


def test_warmup_and_smoothing_window_make_no_decisions():
    config = PlateauConfig(patience=1, smoothing=3, warmup_epochs=2)
    controller = PlateauController(config)
    decisions = run(controller, [0.9, 0.1, 0.1])
    assert [d.reason for d in decisions] == ["warmup", "warmup", "improved"]
    assert decisions[-1].smoothed == pytest.approx((0.9 + 0.1 + 0.1) / 3)


def test_invalid_inputs_fail_loudly():
    controller = PlateauController()
    with pytest.raises(ValueError):
        controller.update(1, float("nan"))
    controller.update(1, 0.2)
    with pytest.raises(ValueError):
        controller.update(1, 0.3)
    with pytest.raises(ValueError):
        PlateauConfig(decay_factor=1.0)


def test_telemetry_separates_data_wait_from_step_time():
    now = [0.0]
    syncs = []
    telemetry = EpochTelemetry(clock=lambda: now[0], synchronize=lambda: syncs.append(now[0]),
                               sync_every=2)
    telemetry.epoch_start()
    for _ in range(4):
        now[0] += 1.0          # waiting for the batch (first one is start-up)
        telemetry.batch_start(8)
        now[0] += 3.0          # GPU step
        telemetry.batch_end()
    summary = telemetry.summary()
    assert summary.batches == 4 and summary.images == 32
    assert summary.first_batch_wait_seconds == pytest.approx(1.0)
    assert summary.data_wait_seconds == pytest.approx(3.0)
    assert summary.step_seconds == pytest.approx(12.0)
    assert summary.data_wait_fraction == pytest.approx(3.0 / 15.0)
    assert len(syncs) == 2


def test_gpu_monitor_parsing_and_window_summary():
    assert parse_line("97, 40000, 46068, 301.5, 350.0, 71") == dict(
        utilization=97.0, memory_used_mib=40000.0, memory_total_mib=46068.0,
        power_w=301.5, power_limit_w=350.0, temperature_c=71.0)
    assert parse_line("[N/A], 1, 2, 3, 4, 5") is None
    partial = parse_line("50, 1, 2, [N/A], 4, 5")
    assert partial["power_w"] is None and partial["utilization"] == 50.0

    monitor = GPUMonitor(csv_path=None)
    for t, util in enumerate([10, 80, 90, 100]):
        monitor.record(float(t), dict(utilization=float(util), memory_used_mib=1.0,
                                      memory_total_mib=2.0, power_w=200.0,
                                      power_limit_w=400.0, temperature_c=60.0))
    summary = monitor.summarize(1.0, 3.0)
    assert summary["samples"] == 3
    assert summary["utilization_mean"] == pytest.approx(90.0)
    assert summary["power_fraction"] == pytest.approx(0.5)
    assert percentile([1.0, 2.0, 3.0, 4.0, 5.0], 0.5) == 3.0
