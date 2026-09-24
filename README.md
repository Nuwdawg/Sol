# exp046: first Mask2Former, TPU, 1024 px, about 4 epochs

## What was tried and the decisive result

From `book/chapters/appB_ledger.tex`:

| ledger row | what was tried | decisive result |
|---|---|---|
| exp046–058 | first Mask2Former, TPU, 1024 px, about 4 epochs | 0.300, then 0.247; public 0.20 |

## How to run

Run from the repository root, so that `experiments.*` and `shared.*` import.
- `python experiments/exp_046_mask2/src/evaluate.py`
- `python kaggle/exp046_mask2_train/train.py`
- `python scripts/check_mask2_target_resolution.py`
- Kaggle kernel: `cd kaggle/exp046_mask2_train && kaggle kernels push -p .`
- Tests: `python -m pytest tests/test_mask2_checkpointing.py tests/test_pq_matches_official.py tests/test_query_mask_loss.py tests/test_utils.py`

## Copied files

These files are copies. The original lives on the branch named.

- `experiments/exp_033_inference/__init__.py` from `exp/033-inference`
- `experiments/exp_033_inference/src/__init__.py` from `exp/033-inference`
- `experiments/exp_033_inference/src/sweep.py` from `exp/033-inference`
- `shared/__init__.py` from shared module (original in `main`)
- `shared/data_split.py` from shared module (original in `main`)
- `shared/mask2_checkpointing.py` from shared module (original in `main`)
- `shared/query_mask_loss.py` from shared module (original in `main`)
- `shared/utils.py` from shared module (original in `main`)

## Experiment log

- No section heading names this experiment. Search `docs/ACTIVE_EXPERIMENT.md` and `docs/leaderboard-analysis.md` for `exp046`.

## Dependencies not in requirements.txt

The root `requirements.txt` does not list these imported packages: `pytest`, `scipy`, `torch_xla`, `transformers`, `ultralytics`. No versions are pinned here.

## Licence

Copyright (c) 2026 Ismail Kasozi. This branch uses Ultralytics, which is licensed under
AGPL-3.0, so the branch is licensed under the GNU Affero General Public License v3.0 (`LICENSE`).
