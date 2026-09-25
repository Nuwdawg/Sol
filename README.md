# exp010: YOLO11m at 2048 on a T4; fell to `mask_ratio` 2; 62 of 400 epochs

## What was tried and the decisive result

From `book/chapters/appB_ledger.tex`:

| ledger row | what was tried | decisive result |
|---|---|---|
| exp010 | YOLO11m at 2048 on a T4; fell to `mask_ratio` 2; 62 of 400 epochs | 0.4274; public 0.36 |

## How to run

Run from the repository root, so that `experiments.*` and `shared.*` import.
- `python kaggle/exp010/polygon_kernel.py`
- Kaggle kernel: `cd kaggle/exp010 && kaggle kernels push -p .`

## Copied files

These files are copies. The original lives on the branch named.

- `experiments/exp_002_yolo_seg/src/prepare_yolo.py` from `exp/002-yolo-seg`
- `shared/__init__.py` from shared module (original in `main`)
- `shared/data_split.py` from shared module (original in `main`)

## Experiment log

- `docs/leaderboard-analysis.md` line 1636: exp_010 session 1 — the run did not test the hypothesis it was built to test
- `docs/leaderboard-analysis.md` line 1684: exp_010 on the leaderboard — worse on validation, equal in public

## Dependencies not in requirements.txt

The root `requirements.txt` does not list these imported packages: `shapely`, `ultralytics`. No versions are pinned here.

## Licence

Copyright (c) 2026 Ismail Kasozi. This branch uses Ultralytics, which is licensed under
AGPL-3.0, so the branch is licensed under the GNU Affero General Public License v3.0 (`LICENSE`).
