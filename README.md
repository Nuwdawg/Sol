# exp097: YOLOv8l-seg at 2048 on an L40, convergence control, T4 rehearsal

## What was tried and the decisive result

From `book/chapters/appB_ledger.tex`:

| ledger row | what was tried | decisive result |
|---|---|---|
| exp097, 099 | YOLOv8l-seg at 2048 on an L40, convergence control, T4 rehearsal | converged at epoch 124; USD 2.74 |

## How to run

Run from the repository root, so that `experiments.*` and `shared.*` import.
- `python experiments/exp_097_runpod_yolo/smoke/smoke.py`
- `python -m experiments.exp_097_runpod_yolo.src.ensemble_pools --help`
- `python -m experiments.exp_097_runpod_yolo.src.evaluate_checkpoints --help`
- `python -m experiments.exp_097_runpod_yolo.src.evaluate_ensemble --help`
- `python -m experiments.exp_097_runpod_yolo.src.fetch_data --help`
- `python -m experiments.exp_097_runpod_yolo.src.kaggle_sync --help`
- `python -m experiments.exp_097_runpod_yolo.src.preflight --help`
- `python -m experiments.exp_097_runpod_yolo.src.sweep_emission --help`
- `python -m experiments.exp_097_runpod_yolo.src.train --help`
- `python runpod/inference_setup.py`
- `python runpod/train.py`
- `python scripts/emit_exp097_submission.py`
- `python scripts/evaluate_yolo_checkpoints.py`
- `python scripts/fetch_exp097_outputs.py`
- `python scripts/package_exp097_code.py`
- `python scripts/package_exp097_pod_request.py`
- `python scripts/package_exp097_submission.py`
- `python scripts/package_exp099_smoke.py`
- Kaggle dataset folder: `kaggle/exp097_code_dataset`
- Tests: `python -m pytest tests/test_exp097_data_integration.py tests/test_exp097_emission_parity.py tests/test_exp097_ensemble_fusion.py tests/test_exp097_evaluator_parity.py tests/test_exp097_kaggle_sync.py tests/test_exp097_packaging.py tests/test_exp097_preflight_select.py tests/test_exp097_training_control.py tests/test_inference_sweep.py tests/test_pq_matches_official.py tests/test_prepare_yolo.py tests/test_utils.py`

## Copied files

These files are copies. The original lives on the branch named.

- `experiments/exp_002_yolo_seg/src/prepare_yolo.py` from `exp/002-yolo-seg`
- `experiments/exp_007_tta/src/dihedral.py` from `exp/007-tta`
- `experiments/exp_033_inference/__init__.py` from `exp/033-inference`
- `experiments/exp_033_inference/src/__init__.py` from `exp/033-inference`
- `experiments/exp_033_inference/src/cached_tta.py` from `exp/033-inference`
- `experiments/exp_033_inference/src/sweep.py` from `exp/033-inference`
- `experiments/exp_128_detector_retrain/__init__.py` from `exp/128-detector-retrain`
- `experiments/exp_128_detector_retrain/src/__init__.py` from `exp/128-detector-retrain`
- `experiments/exp_128_detector_retrain/src/prepare_dataset.py` from `exp/128-detector-retrain`
- `experiments/exp_128_detector_retrain/src/preprocess.py` from `exp/128-detector-retrain`
- `scripts/existing_weight_trials.py` from `exp/059-soft`
- `shared/__init__.py` from shared module (original in `main`)
- `shared/data_split.py` from shared module (original in `main`)
- `shared/utils.py` from shared module (original in `main`)
- `experiments/exp_002_yolo_seg/src/predict.py` from `exp/002-yolo-seg`
- `experiments/exp_005_postproc/src/nearmiss.py` from `exp/005-center-offset`
- `shared/solar_disk.py` from shared module (original in `main`)

## Experiment log

- `docs/ACTIVE_EXPERIMENT.md` line 1028: 2026-09-15 23:30 Nairobi, exp097 prepared (RunPod, not launched)
- `docs/ACTIVE_EXPERIMENT.md` line 1055: 2026-09-15 23:45 Nairobi, exp097 code uploaded; pod on hold
- `docs/ACTIVE_EXPERIMENT.md` line 1078: 2026-09-16 00:03 Nairobi, exp097 LAUNCHED on RunPod
- `docs/ACTIVE_EXPERIMENT.md` line 1127: 2026-09-16 00:18 Nairobi, exp097 relaunched
- `docs/ACTIVE_EXPERIMENT.md` line 1137: exp097 pod 2 crashed at training start; four Ultralytics 8.4.153 defects fixed (2026-09-16)
- `docs/ACTIVE_EXPERIMENT.md` line 1210: exp097 relaunch and its monitoring limits (2026-09-16)
- `docs/ACTIVE_EXPERIMENT.md` line 1263: exp097 first stage ended at epoch 75 (2026-09-16 05:42 UTC)
- `docs/ACTIVE_EXPERIMENT.md` line 1411: exp097 converged at epoch 124 and the pod stopped itself (2026-09-16 07:11 UTC)

## Dependencies not in requirements.txt

The root `requirements.txt` does not list these imported packages: `pytest`, `requests`, `shapely`, `ultralytics`. No versions are pinned here.

## Licence

Copyright (c) 2026 Ismail Kasozi. This branch uses Ultralytics, which is licensed under
AGPL-3.0, so the branch is licensed under the GNU Affero General Public License v3.0 (`LICENSE`).
