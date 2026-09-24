# exp117: real repair operators under a perfect gate

## What was tried and the decisive result

From `book/chapters/appB_ledger.tex`:

| ledger row | what was tried | decisive result |
|---|---|---|
| exp114–117 | real repair operators under a perfect gate | at most 34% of a ceiling |

## How to run

Run from the repository root, so that `experiments.*` and `shared.*` import.
- `python scripts/audit_width_profile.py`
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

- `docs/ACTIVE_EXPERIMENT.md` line 2409: exp117: growth operators lose to the control that reads no image at all (2026-09-16 17:39 UTC)
- `docs/ACTIVE_EXPERIMENT.md` line 2499: exp117 addendum: the growth repair has no trigger, and that was already measured
- `docs/ACTIVE_EXPERIMENT.md` line 2545: exp117 addendum 2: where the leverage actually is, costed before any run

## Dependencies not in requirements.txt

The root `requirements.txt` does not list these imported packages: `pytest`, `ultralytics`. No versions are pinned here.

## Licence

Copyright (c) 2026 Ismail Kasozi. This branch uses Ultralytics, which is licensed under
AGPL-3.0, so the branch is licensed under the GNU Affero General Public License v3.0 (`LICENSE`).
