# exp017: sub-pixel trim by a higher cut on the logit field

## What was tried and the decisive result

From `book/chapters/appB_ledger.tex`:

| ledger row | what was tried | decisive result |
|---|---|---|
| exp017 | sub-pixel trim by a higher cut on the logit field | no better than erosion |

## How to run

Run from the repository root, so that `experiments.*` and `shared.*` import.
- `python -m experiments.exp_017_softmask.src.logit_sweep --help`
- `python kaggle/exp017/logit_kernel.py`
- Kaggle kernel: `cd kaggle/exp017 && kaggle kernels push -p .`
- Tests: `python -m pytest tests/test_pq_matches_official.py tests/test_utils.py`

## Copied files

These files are copies. The original lives on the branch named.

- `shared/__init__.py` from shared module (original in `main`)
- `shared/data_split.py` from shared module (original in `main`)
- `shared/utils.py` from shared module (original in `main`)

## Experiment log

- `docs/leaderboard-analysis.md` line 1543: exp_017 — sub-pixel trimming is possible after all, and buys nothing (negative result)

## Dependencies not in requirements.txt

The root `requirements.txt` does not list these imported packages: `pytest`, `ultralytics`. No versions are pinned here.

## Licence

Copyright (c) 2026 Ismail Kasozi. This branch uses Ultralytics, which is licensed under
AGPL-3.0, so the branch is licensed under the GNU Affero General Public License v3.0 (`LICENSE`).
