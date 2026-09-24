# exp004: U-Net (ResNet-34) at 1024, 300 epochs on TPU, flip averaging

## What was tried and the decisive result

From `book/chapters/appB_ledger.tex`:

| ledger row | what was tried | decisive result |
|---|---|---|
| exp004 | U-Net (ResNet-34) at 1024, 300 epochs on TPU, flip averaging | public 0.28 |
| exp004 | spine-seeded instances, ablated on degraded annotations | +0.0005; dropped |

## How to run

Run from the repository root, so that `experiments.*` and `shared.*` import.
- `python experiments/exp_004_spine_tpu/src/predict.py`
- `python experiments/exp_004_spine_tpu/src/prepare.py`
- `python experiments/exp_004_spine_tpu/src/train_xla.py`
- `python kaggle/exp004/exp004_kaggle.py`
- Kaggle kernel: `cd kaggle/exp004 && kaggle kernels push -p .`
- Tests: `python -m pytest tests/test_pq_matches_official.py tests/test_utils.py`

## Copied files

These files are copies. The original lives on the branch named.

- `shared/__init__.py` from shared module (original in `main`)
- `shared/data_split.py` from shared module (original in `main`)
- `shared/utils.py` from shared module (original in `main`)

## Experiment log

- `docs/leaderboard-analysis.md` line 226: exp_004 — dense segmentation at 1024, trained to convergence on TPU

## Dependencies not in requirements.txt

The root `requirements.txt` does not list these imported packages: `pytest`, `torch_xla`. No versions are pinned here.

## Licence

Copyright (c) 2026 Ismail Kasozi. This branch uses Ultralytics, which is licensed under
AGPL-3.0, so the branch is licensed under the GNU Affero General Public License v3.0 (`LICENSE`).
