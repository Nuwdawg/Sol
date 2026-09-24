# exp113: oracle ceilings of four repair families

## What was tried and the decisive result

From `book/chapters/appB_ledger.tex`:

| ledger row | what was tried | decisive result |
|---|---|---|
| exp113 | oracle ceilings of four repair families | +0.012 to +0.033 |

## How to run

Run from the repository root, so that `experiments.*` and `shared.*` import.
- `python scripts/run_ceilings.py`
- `python scripts/run_growth_ceilings.py`
- `python scripts/run_split_operators.py`
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

- `docs/ACTIVE_EXPERIMENT.md` line 2204: exp113: oracle ceilings for the split and union repair families (2026-09-16 12:59 UTC)
- `docs/ACTIVE_EXPERIMENT.md` line 2244: exp113b: growth ceilings, and a correction to exp108's operational reading (2026-09-16 13:19 UTC)

## Dependencies not in requirements.txt

The root `requirements.txt` does not list these imported packages: `pytest`, `ultralytics`. No versions are pinned here.

## Licence

Copyright (c) 2026 Ismail Kasozi. This branch uses Ultralytics, which is licensed under
AGPL-3.0, so the branch is licensed under the GNU Affero General Public License v3.0 (`LICENSE`).
