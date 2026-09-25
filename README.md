# exp018: refiner trained on 3,429 real detector errors

## What was tried and the decisive result

From `book/chapters/appB_ledger.tex`:

| ledger row | what was tried | decisive result |
|---|---|---|
| exp011, 012, 018 | refiner trained on 3,429 real detector errors | learned the identity; -0.031 |

## How to run

Run from the repository root, so that `experiments.*` and `shared.*` import.
- `python kaggle/exp018/refine_kernel.py`
- Kaggle kernel: `cd kaggle/exp018 && kaggle kernels push -p .`

## Copied files

These files are copies. The original lives on the branch named.

- `experiments/exp_002_yolo_seg/src/predict.py` from `exp/002-yolo-seg`
- `experiments/exp_008_refiner/src/apply.py` from `exp/008-refiner`
- `experiments/exp_008_refiner/src/evaluate.py` from `exp/008-refiner`
- `experiments/exp_008_refiner/src/model.py` from `exp/008-refiner`
- `shared/__init__.py` from shared module (original in `main`)
- `shared/data_split.py` from shared module (original in `main`)
- `shared/utils.py` from shared module (original in `main`)

## Experiment log

- `docs/leaderboard-analysis.md` line 1490: exp_018 — and scored against the detector, it is worse than doing nothing

## Dependencies not in requirements.txt

The root `requirements.txt` does not list these imported packages: `ultralytics`. No versions are pinned here.

## Licence

Copyright (c) 2026 Ismail Kasozi. This branch uses Ultralytics, which is licensed under
AGPL-3.0, so the branch is licensed under the GNU Affero General Public License v3.0 (`LICENSE`).
