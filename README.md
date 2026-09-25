# exp003: Kaggle kernel driver for experiment 003: large YOLO at full resolution

## What was tried and the decisive result

The ledger has no row for this experiment. See the log sections below.

## How to run

Run from the repository root, so that `experiments.*` and `shared.*` import.
- `python kaggle/exp003/exp003_kaggle.py`
- Kaggle kernel: `cd kaggle/exp003 && kaggle kernels push -p .`

## Copied files

These files are copies. The original lives on the branch named.

- `experiments/exp_002_yolo_seg/src/predict.py` from `exp/002-yolo-seg`
- `experiments/exp_002_yolo_seg/src/prepare_yolo.py` from `exp/002-yolo-seg`
- `shared/__init__.py` from shared module (original in `main`)
- `shared/data_split.py` from shared module (original in `main`)
- `shared/utils.py` from shared module (original in `main`)

## Experiment log

- `docs/leaderboard-analysis.md` line 387: What this implies for exp_003

## Dependencies not in requirements.txt

The root `requirements.txt` does not list these imported packages: `shapely`, `ultralytics`. No versions are pinned here.

## Licence

Copyright (c) 2026 Ismail Kasozi. This branch uses Ultralytics, which is licensed under
AGPL-3.0, so the branch is licensed under the GNU Affero General Public License v3.0 (`LICENSE`).
