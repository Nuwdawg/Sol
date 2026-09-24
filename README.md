# exp002: YOLO11m-seg at 1280, 149 epochs, stopped by a time budget

## What was tried and the decisive result

From `book/chapters/appB_ledger.tex`:

| ledger row | what was tried | decisive result |
|---|---|---|
| exp002 | YOLO11m-seg at 1280, 149 epochs, stopped by a time budget | 0.3736 tuned; public 0.32 |
| exp002 | the same weights, inference at the native 2048 | 0.4064; public 0.33 |

## How to run

Run from the repository root, so that `experiments.*` and `shared.*` import.
- `python experiments/exp_002_yolo_seg/src/predict.py`
- `python experiments/exp_002_yolo_seg/src/prepare_yolo.py`
- `python experiments/exp_002_yolo_seg/src/tune.py`
- `python kaggle/exp002/exp002_kaggle.py`
- `python kaggle/exp002_imgsz/imgsz_kernel.py`
- `python kaggle/exp002_tta/tta_kernel.py`
- `python kaggle/exp002_tune/tune_kernel.py`
- Kaggle kernel: `cd kaggle/exp002 && kaggle kernels push -p .`
- Kaggle kernel: `cd kaggle/exp002_imgsz && kaggle kernels push -p .`
- Kaggle kernel: `cd kaggle/exp002_tta && kaggle kernels push -p .`
- Kaggle kernel: `cd kaggle/exp002_tune && kaggle kernels push -p .`
- Kaggle dataset folder: `kaggle/exp002_pools_dataset`
- Tests: `python -m pytest tests/test_pq_matches_official.py tests/test_prepare_yolo.py tests/test_utils.py`

## Copied files

These files are copies. The original lives on the branch named.

- `shared/__init__.py` from shared module (original in `main`)
- `shared/data_split.py` from shared module (original in `main`)
- `shared/utils.py` from shared module (original in `main`)

## Experiment log

- `docs/leaderboard-analysis.md` line 113: exp_002 — YOLO instance segmentation, yolo11m-seg @ 1280

## Dependencies not in requirements.txt

The root `requirements.txt` does not list these imported packages: `pytest`, `shapely`, `ultralytics`. No versions are pinned here.

## Licence

Copyright (c) 2026 Ismail Kasozi. This branch uses Ultralytics, which is licensed under
AGPL-3.0, so the branch is licensed under the GNU Affero General Public License v3.0 (`LICENSE`).
