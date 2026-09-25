# exp128: The pre-registered measurement harness: proposal recall first, PQ last

## What was tried and the decisive result

The ledger has no row for this experiment. See the log sections below.

## How to run

Run from the repository root, so that `experiments.*` and `shared.*` import.
- `python -m experiments.exp_128_detector_retrain.src.evaluate_recall --help`
- `python -m experiments.exp_128_detector_retrain.src.prepare_dataset --help`
- `python scripts/package_exp128_code.py`
- Tests: `python -m pytest tests/test_pq_matches_official.py tests/test_prepare_yolo.py tests/test_utils.py`

## Copied files

These files are copies. The original lives on the branch named.

- `experiments/exp_002_yolo_seg/src/prepare_yolo.py` from `exp/002-yolo-seg`
- `experiments/exp_033_inference/__init__.py` from `exp/033-inference`
- `experiments/exp_033_inference/src/__init__.py` from `exp/033-inference`
- `experiments/exp_033_inference/src/sweep.py` from `exp/033-inference`
- `shared/__init__.py` from shared module (original in `main`)
- `shared/data_split.py` from shared module (original in `main`)
- `shared/utils.py` from shared module (original in `main`)
- `experiments/exp_097_runpod_yolo/__init__.py` from `exp/097-runpod-yolo`
- `experiments/exp_097_runpod_yolo/src/__init__.py` from `exp/097-runpod-yolo`
- `experiments/exp_097_runpod_yolo/src/convergence.py` from `exp/097-runpod-yolo`
- `experiments/exp_097_runpod_yolo/src/events.py` from `exp/097-runpod-yolo`
- `experiments/exp_097_runpod_yolo/src/fetch_data.py` from `exp/097-runpod-yolo`
- `experiments/exp_097_runpod_yolo/src/gpu_monitor.py` from `exp/097-runpod-yolo`
- `experiments/exp_097_runpod_yolo/src/kaggle_sync.py` from `exp/097-runpod-yolo`
- `experiments/exp_097_runpod_yolo/src/preflight.py` from `exp/097-runpod-yolo`
- `experiments/exp_097_runpod_yolo/src/telemetry.py` from `exp/097-runpod-yolo`
- `experiments/exp_097_runpod_yolo/src/train.py` from `exp/097-runpod-yolo`

## Experiment log

- No section heading names this experiment. Search `docs/ACTIVE_EXPERIMENT.md` and `docs/leaderboard-analysis.md` for `exp128`.

## Dependencies not in requirements.txt

The root `requirements.txt` does not list these imported packages: `pytest`, `shapely`, `ultralytics`. No versions are pinned here.

## Licence

Copyright (c) 2026 Ismail Kasozi. This branch uses Ultralytics, which is licensed under
AGPL-3.0, so the branch is licensed under the GNU Affero General Public License v3.0 (`LICENSE`).
