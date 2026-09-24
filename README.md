# exp124: exp124: does a second architecture confirm a discarded candidate?

## What was tried and the decisive result

From `book/chapters/appB_ledger.tex`:

| ledger row | what was tried | decisive result |
|---|---|---|
| exp120–126 | feature verifiers | at most +0.0017 |
| exp124 | cross-architecture confirmation | AUC 0.688; no gain |

## How to run

Run from the repository root, so that `experiments.*` and `shared.*` import.
- `python scripts/audit_cross_architecture_verifier.py`
- `python scripts/package_cross_architecture_verifier.py`
- `python scripts/run_confirmed_verifier.py`
- Kaggle kernel: `cd kaggle/exp124_cross_architecture_cpu && kaggle kernels push -p .`
- Tests: `python -m pytest tests/test_pq_matches_official.py tests/test_utils.py`

## Copied files

These files are copies. The original lives on the branch named.

- `experiments/exp_033_inference/__init__.py` from `exp/033-inference`
- `experiments/exp_033_inference/src/__init__.py` from `exp/033-inference`
- `experiments/exp_033_inference/src/sweep.py` from `exp/033-inference`
- `shared/__init__.py` from shared module (original in `main`)
- `shared/data_split.py` from shared module (original in `main`)
- `shared/utils.py` from shared module (original in `main`)

## Experiment log

- `docs/ACTIVE_EXPERIMENT.md` line 2872: exp124: cross-architecture confirmation is the best feature found, and my grid missed it (2026-09-16 20:11 UTC)

## Dependencies not in requirements.txt

The root `requirements.txt` does not list these imported packages: `pytest`, `sklearn`, `ultralytics`. No versions are pinned here.

## Licence

Copyright (c) 2026 Ismail Kasozi. This branch uses Ultralytics, which is licensed under
AGPL-3.0, so the branch is licensed under the GNU Affero General Public License v3.0 (`LICENSE`).
