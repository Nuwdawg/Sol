# exp105: near-miss audits and repair triggers

## What was tried and the decisive result

From `book/chapters/appB_ledger.tex`:

| ledger row | what was tried | decisive result |
|---|---|---|
| exp105–112 | near-miss audits and repair triggers | AUC at most 0.81; precision wall |

## How to run

Run from the repository root, so that `experiments.*` and `shared.*` import.
- `python scripts/audit_instance_structure.py`
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

- `docs/ACTIVE_EXPERIMENT.md` line 1707: exp105: the near-threshold failures are shape, not topology (2026-09-16 10:26 UTC)
- `docs/ACTIVE_EXPERIMENT.md` line 1755: exp105b: near-threshold masks are too small, but nothing observable marks them (2026-09-16 10:31 UTC)
- `docs/ACTIVE_EXPERIMENT.md` line 1797: exp105c: geometry is dead, and the neck trigger fails on precision (2026-09-16 10:44 UTC)

## Dependencies not in requirements.txt

The root `requirements.txt` does not list these imported packages: `pytest`, `ultralytics`. No versions are pinned here.

## Licence

Copyright (c) 2026 Ismail Kasozi. This branch uses Ultralytics, which is licensed under
AGPL-3.0, so the branch is licensed under the GNU Affero General Public License v3.0 (`LICENSE`).
