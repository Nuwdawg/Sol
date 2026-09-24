# exp114: real repair operators under a perfect gate

## What was tried and the decisive result

From `book/chapters/appB_ledger.tex`:

| ledger row | what was tried | decisive result |
|---|---|---|
| exp114–117 | real repair operators under a perfect gate | at most 34% of a ceiling |

## How to run

Run from the repository root, so that `experiments.*` and `shared.*` import.
- `python scripts/sweep_split_operators.py`
- Tests: `python -m pytest tests/test_pq_matches_official.py tests/test_utils.py`

## Copied files

These files are copies. The original lives on the branch named.

- `experiments/exp_033_inference/__init__.py` from `exp/033-inference`
- `experiments/exp_033_inference/src/__init__.py` from `exp/033-inference`
- `experiments/exp_033_inference/src/sweep.py` from `exp/033-inference`
- `experiments/exp_113_ceilings/src/harness.py` from `exp/113-ceilings`
- `experiments/exp_113_ceilings/src/interface_cut.py` from `exp/113-ceilings`
- `experiments/exp_113_ceilings/src/splitters.py` from `exp/113-ceilings`
- `shared/__init__.py` from shared module (original in `main`)
- `shared/data_split.py` from shared module (original in `main`)
- `shared/utils.py` from shared module (original in `main`)

## Experiment log

- `docs/ACTIVE_EXPERIMENT.md` line 2284: exp114: real splitters under a perfect gate recover 13% of the ceiling (2026-09-16 14:00 UTC)

## Dependencies not in requirements.txt

The root `requirements.txt` does not list these imported packages: `pytest`, `ultralytics`. No versions are pinned here.

## Licence

Copyright (c) 2026 Ismail Kasozi. This branch uses Ultralytics, which is licensed under
AGPL-3.0, so the branch is licensed under the GNU Affero General Public License v3.0 (`LICENSE`).
