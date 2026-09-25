# exp104: morphology on the ensemble

## What was tried and the decisive result

From `book/chapters/appB_ledger.tex`:

| ledger row | what was tried | decisive result |
|---|---|---|
| exp104 | morphology on the ensemble | one-pixel erosion best |

## How to run

Run from the repository root, so that `experiments.*` and `shared.*` import.
- `python scripts/package_exp104_emission_grid.py`
- Kaggle kernel: `cd kaggle/exp104_emission_grid_cpu && kaggle kernels push -p .`

## Copied files

These files are copies. The original lives on the branch named.

- `experiments/exp_033_inference/__init__.py` from `exp/033-inference`
- `experiments/exp_033_inference/src/__init__.py` from `exp/033-inference`
- `experiments/exp_033_inference/src/sweep.py` from `exp/033-inference`
- `experiments/exp_097_runpod_yolo/__init__.py` from `exp/097-runpod-yolo`
- `experiments/exp_097_runpod_yolo/src/__init__.py` from `exp/097-runpod-yolo`
- `experiments/exp_097_runpod_yolo/src/events.py` from `exp/097-runpod-yolo`
- `experiments/exp_097_runpod_yolo/src/sweep_emission.py` from `exp/097-runpod-yolo`
- `shared/__init__.py` from shared module (original in `main`)
- `shared/data_split.py` from shared module (original in `main`)
- `shared/utils.py` from shared module (original in `main`)

## Experiment log

- `docs/ACTIVE_EXPERIMENT.md` line 1846: exp104: dilation is dead on the 2048 ensemble too (2026-09-16 10:45 UTC)

## Dependencies not in requirements.txt

The root `requirements.txt` does not list these imported packages: `ultralytics`. No versions are pinned here.

## Licence

Copyright (c) 2026 Ismail Kasozi. This branch uses Ultralytics, which is licensed under
AGPL-3.0, so the branch is licensed under the GNU Affero General Public License v3.0 (`LICENSE`).
