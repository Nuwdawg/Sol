# exp098: Compare trained checkpoints on canonical per-annotator PQ, then D4 the best

## What was tried and the decisive result

From `book/chapters/appB_ledger.tex`:

| ledger row | what was tried | decisive result |
|---|---|---|
| exp098, 100 | scoring exp097's checkpoints | D4 0.4668; public 0.39 |

## How to run

Run from the repository root, so that `experiments.*` and `shared.*` import.
- `python scripts/evaluate_yolo_checkpoints.py`
- `python scripts/package_exp098_evaluation.py`
- Kaggle kernel: `cd kaggle/exp098_exp097_evaluation_t4 && kaggle kernels push -p .`
- Tests: `python -m pytest tests/test_inference_sweep.py tests/test_pq_matches_official.py tests/test_utils.py`

## Copied files

These files are copies. The original lives on the branch named.

- `experiments/exp_007_tta/src/dihedral.py` from `exp/007-tta`
- `experiments/exp_033_inference/__init__.py` from `exp/033-inference`
- `experiments/exp_033_inference/src/__init__.py` from `exp/033-inference`
- `experiments/exp_033_inference/src/cached_tta.py` from `exp/033-inference`
- `experiments/exp_033_inference/src/sweep.py` from `exp/033-inference`
- `scripts/existing_weight_trials.py` from `exp/059-soft`
- `shared/__init__.py` from shared module (original in `main`)
- `shared/data_split.py` from shared module (original in `main`)
- `shared/utils.py` from shared module (original in `main`)
- `experiments/exp_097_runpod_yolo/__init__.py` from `exp/097-runpod-yolo`
- `experiments/exp_097_runpod_yolo/src/__init__.py` from `exp/097-runpod-yolo`
- `experiments/exp_097_runpod_yolo/src/evaluate_checkpoints.py` from `exp/097-runpod-yolo`
- `experiments/exp_097_runpod_yolo/src/events.py` from `exp/097-runpod-yolo`

## Experiment log

- `docs/ACTIVE_EXPERIMENT.md` line 1329: exp098 scored stage1_best.pt: 0.0021 below the baseline, so no submission (2026-09-16 06:06 UTC)
- `docs/ACTIVE_EXPERIMENT.md` line 1472: exp098 v2: stage2 clears the baseline by 0.0109 (2026-09-16 07:25 UTC)
- `docs/ACTIVE_EXPERIMENT.md` line 1526: exp098 v3: training after epoch 88 produced nothing (2026-09-16 08:43 UTC)

## Dependencies not in requirements.txt

The root `requirements.txt` does not list these imported packages: `pytest`, `ultralytics`. No versions are pinned here.

## Licence

Copyright (c) 2026 Ismail Kasozi. This branch uses Ultralytics, which is licensed under
AGPL-3.0, so the branch is licensed under the GNU Affero General Public License v3.0 (`LICENSE`).
