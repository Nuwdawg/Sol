# exp073: ROI mask loss; continuation

## What was tried and the decisive result

From `book/chapters/appB_ledger.tex`:

| ledger row | what was tried | decisive result |
|---|---|---|
| exp070, 073 | ROI mask loss; continuation | 1.58 against 2.86 s per step; D4 0.4555 |

## How to run

Run from the repository root, so that `experiments.*` and `shared.*` import.
- Kaggle kernel: `cd kaggle/exp073_roi_continuation && kaggle kernels push -p .`

## Copied files

These files are copies. The original lives on the branch named.

- `experiments/exp_002_yolo_seg/src/prepare_yolo.py` from `exp/002-yolo-seg`
- `experiments/exp_032_chunked/__init__.py` from `exp/032-chunked`
- `experiments/exp_032_chunked/src/__init__.py` from `exp/032-chunked`
- `experiments/exp_032_chunked/src/train.py` from `exp/032-chunked`
- `shared/__init__.py` from shared module (original in `main`)
- `shared/chunked_mask_loss.py` from shared module (original in `main`)
- `shared/data_split.py` from shared module (original in `main`)
- `shared/roi_mask_loss.py` from shared module (original in `main`)

## Experiment log

- No section heading names this experiment. Search `docs/ACTIVE_EXPERIMENT.md` and `docs/leaderboard-analysis.md` for `exp073`.

## Dependencies not in requirements.txt

The root `requirements.txt` does not list these imported packages: `shapely`, `ultralytics`. No versions are pinned here.

## Licence

Copyright (c) 2026 Ismail Kasozi. This branch uses Ultralytics, which is licensed under
AGPL-3.0, so the branch is licensed under the GNU Affero General Public License v3.0 (`LICENSE`).
