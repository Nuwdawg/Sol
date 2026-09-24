# exp019: re-ranker on the logit-field profile

## What was tried and the decisive result

From `book/chapters/appB_ledger.tex`:

| ledger row | what was tried | decisive result |
|---|---|---|
| exp019 | re-ranker on the logit-field profile | promotes nothing |

## How to run

Run from the repository root, so that `experiments.*` and `shared.*` import.
- `python -m experiments.exp_019_fieldrank.src.rank --help`
- `python kaggle/exp019/rank_kernel.py`
- Kaggle kernel: `cd kaggle/exp019 && kaggle kernels push -p .`
- Tests: `python -m pytest tests/test_pq_matches_official.py tests/test_utils.py`

## Copied files

These files are copies. The original lives on the branch named.

- `experiments/exp_017_softmask/__init__.py` from `exp/017-softmask`
- `experiments/exp_017_softmask/src/__init__.py` from `exp/017-softmask`
- `experiments/exp_017_softmask/src/logit_sweep.py` from `exp/017-softmask`
- `shared/__init__.py` from shared module (original in `main`)
- `shared/data_split.py` from shared module (original in `main`)
- `shared/utils.py` from shared module (original in `main`)

## Experiment log

- `docs/leaderboard-analysis.md` line 1584: exp_019 — the field profile is a better ranker and still cannot promote (negative result)

## Dependencies not in requirements.txt

The root `requirements.txt` does not list these imported packages: `pytest`, `sklearn`, `ultralytics`. No versions are pinned here.

## Licence

Copyright (c) 2026 Ismail Kasozi. This branch uses Ultralytics, which is licensed under
AGPL-3.0, so the branch is licensed under the GNU Affero General Public License v3.0 (`LICENSE`).
