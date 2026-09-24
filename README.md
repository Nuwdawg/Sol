# exp134: per-record CNN verifier on crops

## What was tried and the decisive result

From `book/chapters/appB_ledger.tex`:

| ledger row | what was tried | decisive result |
|---|---|---|
| exp134 | per-record CNN verifier on crops | at chance |

## How to run

Run from the repository root, so that `experiments.*` and `shared.*` import.
- `python -m experiments.exp_134_rescorer.src.apply_rescorer --help`
- `python -m experiments.exp_134_rescorer.src.harvest_crops --help`
- `python -m experiments.exp_134_rescorer.src.train_rescorer --help`
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

## Experiment log

- `docs/ACTIVE_EXPERIMENT.md` line 3044: exp134: a CNN on the raw pixels cannot rescore either (2026-09-16 22:05 UTC)

## Dependencies not in requirements.txt

The root `requirements.txt` does not list these imported packages: `pytest`, `ultralytics`. No versions are pinned here.

## Licence

Copyright (c) 2026 Ismail Kasozi. This branch uses Ultralytics, which is licensed under
AGPL-3.0, so the branch is licensed under the GNU Affero General Public License v3.0 (`LICENSE`).
