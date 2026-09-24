# exp088: headroom audit against annotators

## What was tried and the decisive result

From `book/chapters/appB_ledger.tex`:

| ledger row | what was tried | decisive result |
|---|---|---|
| exp088 | headroom audit against annotators | majority-mask IoU 0.81 against 0.68 |

## How to run

Run from the repository root, so that `experiments.*` and `shared.*` import.
- `python scripts/audit_annotator_agreement.py`
- `python scripts/audit_boundary_bias.py`
- `python scripts/audit_consensus_ceiling.py`
- `python scripts/audit_headroom.py`
- `python scripts/audit_threshold_structure.py`
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

- `docs/ACTIVE_EXPERIMENT.md` line 741: 2026-09-15 17:20-18:10 Nairobi, exp088 headroom decomposition and exp086 outcome
- `docs/ACTIVE_EXPERIMENT.md` line 749: exp088: where the gap actually is
- `docs/ACTIVE_EXPERIMENT.md` line 797: exp088 follow-up: limb-aware pruning

## Dependencies not in requirements.txt

The root `requirements.txt` does not list these imported packages: `pytest`, `ultralytics`. No versions are pinned here.

## Licence

Copyright (c) 2026 Ismail Kasozi. This branch uses Ultralytics, which is licensed under
AGPL-3.0, so the branch is licensed under the GNU Affero General Public License v3.0 (`LICENSE`).
