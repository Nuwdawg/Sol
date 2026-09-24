# exp077: Canonical validation of a frozen dense model as a YOLO complement

## What was tried and the decisive result

The ledger has no row for this experiment. See the log sections below.

## How to run

Run from the repository root, so that `experiments.*` and `shared.*` import.
- `python scripts/evaluate_semantic_complement.py`
- Kaggle kernel: `cd kaggle/exp077_semantic_complement && kaggle kernels push -p .`
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

- No section heading names this experiment. Search `docs/ACTIVE_EXPERIMENT.md` and `docs/leaderboard-analysis.md` for `exp077`.

## Dependencies not in requirements.txt

The root `requirements.txt` does not list these imported packages: `pytest`, `torch_xla`, `ultralytics`. No versions are pinned here.

## Licence

Copyright (c) 2026 Ismail Kasozi. This branch uses Ultralytics, which is licensed under
AGPL-3.0, so the branch is licensed under the GNU Affero General Public License v3.0 (`LICENSE`).
