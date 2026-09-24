# exp126: exp126: is a discarded candidate more credible when it stands apart from what was emitted?

## What was tried and the decisive result

From `book/chapters/appB_ledger.tex`:

| ledger row | what was tried | decisive result |
|---|---|---|
| exp120–126 | feature verifiers | at most +0.0017 |

## How to run

Run from the repository root, so that `experiments.*` and `shared.*` import.
- `python scripts/run_isolation_verifier.py`
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

- `docs/ACTIVE_EXPERIMENT.md` line 2845: exp126: isolation features are noise, and I am not counting the gain (2026-09-16 20:06 UTC)

## Dependencies not in requirements.txt

The root `requirements.txt` does not list these imported packages: `pytest`, `sklearn`, `ultralytics`. No versions are pinned here.

## Licence

Copyright (c) 2026 Ismail Kasozi. This branch uses Ultralytics, which is licensed under
AGPL-3.0, so the branch is licensed under the GNU Affero General Public License v3.0 (`LICENSE`).
