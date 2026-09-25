# exp031: YOLO at 1536, full-resolution loss, 50 annealed epochs

## What was tried and the decisive result

From `book/chapters/appB_ledger.tex`:

| ledger row | what was tried | decisive result |
|---|---|---|
| exp031 | YOLO at 1536, full-resolution loss, 50 annealed epochs | 0.4343; public 0.36 |

## How to run

Run from the repository root, so that `experiments.*` and `shared.*` import.
- `python -m experiments.exp_031_frozen.src.train --help`
- `python kaggle/exp031/frozen_kernel.py`
- Kaggle kernel: `cd kaggle/exp031 && kaggle kernels push -p .`
- Tests: `python -m pytest tests/test_chunked_mask_loss.py tests/test_roi_mask_loss.py`

## Copied files

These files are copies. The original lives on the branch named.

- `experiments/exp_032_chunked/__init__.py` from `exp/032-chunked`
- `experiments/exp_032_chunked/src/__init__.py` from `exp/032-chunked`
- `experiments/exp_032_chunked/src/train.py` from `exp/032-chunked`
- `shared/__init__.py` from shared module (original in `main`)
- `shared/chunked_mask_loss.py` from shared module (original in `main`)
- `shared/roi_mask_loss.py` from shared module (original in `main`)
- `experiments/exp_002_yolo_seg/src/prepare_yolo.py` from `exp/002-yolo-seg`
- `shared/data_split.py` from shared module (original in `main`)

## Experiment log

- `docs/leaderboard-analysis.md` line 2003: exp_031 — the first completed schedule, and what it settles

## Dependencies not in requirements.txt

The root `requirements.txt` does not list these imported packages: `pytest`, `shapely`, `ultralytics`. No versions are pinned here.

## Licence

Copyright (c) 2026 Ismail Kasozi. This branch uses Ultralytics, which is licensed under
AGPL-3.0, so the branch is licensed under the GNU Affero General Public License v3.0 (`LICENSE`).
