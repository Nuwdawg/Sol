# exp140: calibrated YOLOv8l and Mask2Former ensemble

## What was tried and the decisive result

From `book/chapters/appB_ledger.tex`:

| ledger row | what was tried | decisive result |
|---|---|---|
| exp140 | calibrated YOLOv8l and Mask2Former ensemble | 0.4715; public 0.40 |

## How to run

Run from the repository root, so that `experiments.*` and `shared.*` import.
- `python -m experiments.exp_140_three_way_ensemble.src.ensemble --help`
- `python -m experiments.exp_140_three_way_ensemble.src.infer_exp139 --help`
- `python scripts/package_exp140_ensemble.py`
- Kaggle kernel: `cd kaggle/exp140_ensemble_t4 && kaggle kernels push -p .`
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
- `experiments/exp_139_mask2former/__init__.py` from `main`
- `experiments/exp_139_mask2former/src/__init__.py` from `main`
- `experiments/exp_139_mask2former/src/data.py` from `main`
- `experiments/exp_139_mask2former/src/evaluate.py` from `main`
- `experiments/exp_139_mask2former/src/fast_ops.py` from `main`
- `experiments/exp_139_mask2former/src/model.py` from `main`
- `experiments/exp_139_mask2former/src/submit.py` from `main`
- `shared/__init__.py` from shared module (original in `main`)
- `shared/data_split.py` from shared module (original in `main`)
- `shared/utils.py` from shared module (original in `main`)
- `experiments/exp_139_mask2former/tests/__init__.py` from `main`
- `experiments/exp_139_mask2former/tests/reference_data.py` from `main`
- `experiments/exp_139_mask2former/tests/test_equivalence.py` from `main`
- `scripts/package_exp139_gpu_tests.py` from `main`

## Experiment log

- No section heading names this experiment. Search `docs/ACTIVE_EXPERIMENT.md` and `docs/leaderboard-analysis.md` for `exp140`.

## Dependencies not in requirements.txt

The root `requirements.txt` does not list these imported packages: `kernels`, `pytest`, `scipy`, `transformers`, `ultralytics`. No versions are pinned here.

## Licence

Copyright (c) 2026 Ismail Kasozi. This branch uses Ultralytics, which is licensed under
AGPL-3.0, so the branch is licensed under the GNU Affero General Public License v3.0 (`LICENSE`).
