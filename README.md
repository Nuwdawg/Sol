# exp102: two YOLO architectures fused

## What was tried and the decisive result

From `book/chapters/appB_ledger.tex`:

| ledger row | what was tried | decisive result |
|---|---|---|
| exp102, 119 | two YOLO architectures fused | 0.4681; public 0.39 |

## How to run

Run from the repository root, so that `experiments.*` and `shared.*` import.
- `python scripts/emit_ensemble_submission.py`
- `python scripts/package_exp102_pool_ensemble.py`
- Kaggle kernel: `cd kaggle/exp102_pool_ensemble_cpu && kaggle kernels push -p .`
- Tests: `python -m pytest tests/test_inference_sweep.py tests/test_pq_matches_official.py tests/test_utils.py`

## Copied files

These files are copies. The original lives on the branch named.

- `experiments/exp_007_tta/src/dihedral.py` from `exp/007-tta`
- `experiments/exp_033_inference/__init__.py` from `exp/033-inference`
- `experiments/exp_033_inference/src/__init__.py` from `exp/033-inference`
- `experiments/exp_033_inference/src/cached_tta.py` from `exp/033-inference`
- `experiments/exp_033_inference/src/sweep.py` from `exp/033-inference`
- `experiments/exp_097_runpod_yolo/__init__.py` from `exp/097-runpod-yolo`
- `experiments/exp_097_runpod_yolo/src/__init__.py` from `exp/097-runpod-yolo`
- `experiments/exp_097_runpod_yolo/src/evaluate_ensemble.py` from `exp/097-runpod-yolo`
- `experiments/exp_097_runpod_yolo/src/events.py` from `exp/097-runpod-yolo`
- `shared/__init__.py` from shared module (original in `main`)
- `shared/data_split.py` from shared module (original in `main`)
- `shared/utils.py` from shared module (original in `main`)
- `experiments/exp_097_runpod_yolo/src/ensemble_pools.py` from `exp/097-runpod-yolo`

## Experiment log

- `docs/ACTIVE_EXPERIMENT.md` line 1675: exp102: cross-architecture pooling is the first ensemble that works (2026-09-16 10:12 UTC)

## Dependencies not in requirements.txt

The root `requirements.txt` does not list these imported packages: `pytest`, `ultralytics`. No versions are pinned here.

## Licence

Copyright (c) 2026 Ismail Kasozi. This branch uses Ultralytics, which is licensed under
AGPL-3.0, so the branch is licensed under the GNU Affero General Public License v3.0 (`LICENSE`).
