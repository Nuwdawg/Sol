# exp068: YOLO11m at 2048, chunked full-resolution loss

## What was tried and the decisive result

From `book/chapters/appB_ledger.tex`:

| ledger row | what was tried | decisive result |
|---|---|---|
| exp067, 068 | YOLO11m at 2048, chunked full-resolution loss | D4 0.4470 |

## How to run

Run from the repository root, so that `experiments.*` and `shared.*` import.
- Kaggle kernel: `cd kaggle/exp068_chunked_train && kaggle kernels push -p .`

## Copied files

These files are copies. The original lives on the branch named.

- `experiments/exp_002_yolo_seg/src/prepare_yolo.py` from `exp/002-yolo-seg`
- `experiments/exp_007_tta/src/dihedral.py` from `exp/007-tta`
- `experiments/exp_032_chunked/__init__.py` from `exp/032-chunked`
- `experiments/exp_032_chunked/src/__init__.py` from `exp/032-chunked`
- `experiments/exp_032_chunked/src/train.py` from `exp/032-chunked`
- `experiments/exp_033_inference/__init__.py` from `exp/033-inference`
- `experiments/exp_033_inference/src/__init__.py` from `exp/033-inference`
- `experiments/exp_033_inference/src/cached_tta.py` from `exp/033-inference`
- `experiments/exp_033_inference/src/sweep.py` from `exp/033-inference`
- `scripts/evaluate_yolo_checkpoints.py` from `exp/098-exp097-evaluation`
- `scripts/existing_weight_trials.py` from `exp/059-soft`
- `shared/__init__.py` from shared module (original in `main`)
- `shared/chunked_mask_loss.py` from shared module (original in `main`)
- `shared/data_split.py` from shared module (original in `main`)
- `shared/roi_mask_loss.py` from shared module (original in `main`)
- `shared/utils.py` from shared module (original in `main`)

## Experiment log

- No section heading names this experiment. Search `docs/ACTIVE_EXPERIMENT.md` and `docs/leaderboard-analysis.md` for `exp068`.

## Dependencies not in requirements.txt

The root `requirements.txt` does not list these imported packages: `shapely`, `ultralytics`. No versions are pinned here.

## Licence

Copyright (c) 2026 Ismail Kasozi. This branch uses Ultralytics, which is licensed under
AGPL-3.0, so the branch is licensed under the GNU Affero General Public License v3.0 (`LICENSE`).
