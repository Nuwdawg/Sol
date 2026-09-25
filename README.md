# exp028: GPU kernel: high-recall test submission, to probe what the test labels contain

## What was tried and the decisive result

From `book/chapters/appB_ledger.tex`:

| ledger row | what was tried | decisive result |
|---|---|---|
| exp028 | high-recall probe submission | 0.3865; public 0.32 |

## How to run

Run from the repository root, so that `experiments.*` and `shared.*` import.
- `python kaggle/exp028/probe_kernel.py`
- Kaggle kernel: `cd kaggle/exp028 && kaggle kernels push -p .`

## Copied files

These files are copies. The original lives on the branch named.

- `experiments/exp_025_verifier/__init__.py` from `exp/025-verifier`
- `experiments/exp_025_verifier/src/__init__.py` from `exp/025-verifier`
- `experiments/exp_025_verifier/src/harvest.py` from `exp/025-verifier`
- `experiments/exp_025_verifier/src/model.py` from `exp/025-verifier`
- `experiments/exp_025_verifier/src/predict_verified.py` from `exp/025-verifier`
- `shared/__init__.py` from shared module (original in `main`)
- `shared/data_split.py` from shared module (original in `main`)
- `shared/utils.py` from shared module (original in `main`)

## Experiment log

- `docs/leaderboard-analysis.md` line 1887: exp_028 — the test labels are not more inclusive (negative result, and a useful one)

## Dependencies not in requirements.txt

The root `requirements.txt` does not list these imported packages: `ultralytics`. No versions are pinned here.

## Licence

Copyright (c) 2026 Ismail Kasozi. This branch uses Ultralytics, which is licensed under
AGPL-3.0, so the branch is licensed under the GNU Affero General Public License v3.0 (`LICENSE`).
