# exp101: three checkpoints of one run, 24 views

## What was tried and the decisive result

From `book/chapters/appB_ledger.tex`:

| ledger row | what was tried | decisive result |
|---|---|---|
| exp101 | three checkpoints of one run, 24 views | +0.00005 |

## How to run

Run from the repository root, so that `experiments.*` and `shared.*` import.
- `python scripts/package_exp101_ensemble.py`
- Kaggle kernel: `cd kaggle/exp101_exp097_ensemble_t4 && kaggle kernels push -p .`

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

## Experiment log

- `docs/ACTIVE_EXPERIMENT.md` line 1644: exp101: pooling checkpoints from one run buys nothing (2026-09-16 09:38 UTC)

## Dependencies not in requirements.txt

The root `requirements.txt` does not list these imported packages: `ultralytics`. No versions are pinned here.

## Licence

Copyright (c) 2026 Ismail Kasozi. This branch uses Ultralytics, which is licensed under
AGPL-3.0, so the branch is licensed under the GNU Affero General Public License v3.0 (`LICENSE`).
