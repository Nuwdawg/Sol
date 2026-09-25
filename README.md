# exp027: crop verifier labelled by "any annotator"

## What was tried and the decisive result

From `book/chapters/appB_ledger.tex`:

| ledger row | what was tried | decisive result |
|---|---|---|
| exp025–027 | crop verifier labelled by "any annotator" | 72% versus 17% precision |

## How to run

Run from the repository root, so that `experiments.*` and `shared.*` import.
- `python kaggle/exp027/apply_kernel.py`
- Kaggle kernel: `cd kaggle/exp027 && kaggle kernels push -p .`
- Tests: `python -m pytest tests/test_pq_matches_official.py tests/test_utils.py`

## Copied files

These files are copies. The original lives on the branch named.

- `experiments/exp_025_verifier/__init__.py` from `exp/025-verifier`
- `experiments/exp_025_verifier/src/__init__.py` from `exp/025-verifier`
- `experiments/exp_025_verifier/src/harvest.py` from `exp/025-verifier`
- `experiments/exp_025_verifier/src/model.py` from `exp/025-verifier`
- `experiments/exp_033_inference/__init__.py` from `exp/033-inference`
- `experiments/exp_033_inference/src/__init__.py` from `exp/033-inference`
- `experiments/exp_033_inference/src/sweep.py` from `exp/033-inference`
- `shared/__init__.py` from shared module (original in `main`)
- `shared/data_split.py` from shared module (original in `main`)
- `shared/utils.py` from shared module (original in `main`)
- `experiments/exp_025_verifier/src/apply.py` from `exp/025-verifier`

## Experiment log

- `docs/leaderboard-analysis.md` line 1819: exp_025-027 — the verifier is good at the wrong question (negative result)

## Dependencies not in requirements.txt

The root `requirements.txt` does not list these imported packages: `pytest`, `ultralytics`. No versions are pinned here.

## Licence

Copyright (c) 2026 Ismail Kasozi. This branch uses Ultralytics, which is licensed under
AGPL-3.0, so the branch is licensed under the GNU Affero General Public License v3.0 (`LICENSE`).
