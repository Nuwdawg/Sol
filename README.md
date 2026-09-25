# exp067: YOLO11m at 2048, chunked full-resolution loss

## What was tried and the decisive result

From `book/chapters/appB_ledger.tex`:

| ledger row | what was tried | decisive result |
|---|---|---|
| exp067, 068 | YOLO11m at 2048, chunked full-resolution loss | D4 0.4470 |

## How to run

Run from the repository root, so that `experiments.*` and `shared.*` import.
- `python scripts/probe_training.py`
- Kaggle kernel: `cd kaggle/exp067_chunked_probe && kaggle kernels push -p .`

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

- No section heading names this experiment. Search `docs/ACTIVE_EXPERIMENT.md` and `docs/leaderboard-analysis.md` for `exp067`.

## Dependencies not in requirements.txt

The root `requirements.txt` does not list these imported packages: `shapely`, `ultralytics`. No versions are pinned here.

## Licence

Copyright (c) 2026 Ismail Kasozi. This branch uses Ultralytics, which is licensed under
AGPL-3.0, so the branch is licensed under the GNU Affero General Public License v3.0 (`LICENSE`).
