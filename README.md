# exp008: crop refiner trained on synthetic damage

## What was tried and the decisive result

From `book/chapters/appB_ledger.tex`:

| ledger row | what was tried | decisive result |
|---|---|---|
| exp008, 009 | crop refiner trained on synthetic damage | 0.4322 (-0.008); public 0.35 |

## How to run

Run from the repository root, so that `experiments.*` and `shared.*` import.
- `python experiments/exp_008_refiner/src/apply.py`
- `python experiments/exp_008_refiner/src/crops.py`
- `python experiments/exp_008_refiner/src/evaluate.py`
- `python experiments/exp_008_refiner/src/harvest.py`
- `python experiments/exp_008_refiner/src/train_refiner.py`
- `python kaggle/exp008/refiner_kernel.py`
- Kaggle kernel: `cd kaggle/exp008 && kaggle kernels push -p .`
- Tests: `python -m pytest tests/test_pq_matches_official.py tests/test_utils.py`

## Copied files

These files are copies. The original lives on the branch named.

- `experiments/exp_002_yolo_seg/src/predict.py` from `exp/002-yolo-seg`
- `shared/__init__.py` from shared module (original in `main`)
- `shared/data_split.py` from shared module (original in `main`)
- `shared/utils.py` from shared module (original in `main`)

## Experiment log

- `docs/leaderboard-analysis.md` line 994: exp_008/009 — the refiner loses to a one-pixel erosion (negative result)

## Dependencies not in requirements.txt

The root `requirements.txt` does not list these imported packages: `pytest`, `torch_xla`, `ultralytics`. No versions are pinned here.

## Licence

Copyright (c) 2026 Ismail Kasozi. This branch uses Ultralytics, which is licensed under
AGPL-3.0, so the branch is licensed under the GNU Affero General Public License v3.0 (`LICENSE`).
