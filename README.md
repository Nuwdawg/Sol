# exp033: eight-view fusion with the threshold re-swept

## What was tried and the decisive result

From `book/chapters/appB_ledger.tex`:

| ledger row | what was tried | decisive result |
|---|---|---|
| exp033, 037 | eight-view fusion with the threshold re-swept | 0.4571 (+0.0166); public 0.38 |

## How to run

Run from the repository root, so that `experiments.*` and `shared.*` import.
- `python -m experiments.exp_033_inference.src.cached_tta --help`
- `python -m experiments.exp_033_inference.src.harvest_test --help`
- `python -m experiments.exp_033_inference.src.submit_tta --help`
- `python -m experiments.exp_033_inference.src.sweep --help`
- `python scripts/evaluate_public_fusion.py`
- Kaggle kernel: `cd kaggle/exp033_cpu && kaggle kernels push -p .`
- Tests: `python -m pytest tests/test_inference_sweep.py tests/test_pq_matches_official.py tests/test_utils.py`

## Copied files

These files are copies. The original lives on the branch named.

- `experiments/exp_007_tta/src/dihedral.py` from `exp/007-tta`
- `shared/__init__.py` from shared module (original in `main`)
- `shared/data_split.py` from shared module (original in `main`)
- `shared/utils.py` from shared module (original in `main`)

## Experiment log

- No section heading names this experiment. Search `docs/ACTIVE_EXPERIMENT.md` and `docs/leaderboard-analysis.md` for `exp033`.

## Dependencies not in requirements.txt

The root `requirements.txt` does not list these imported packages: `pytest`, `ultralytics`. No versions are pinned here.

## Licence

Copyright (c) 2026 Ismail Kasozi. This branch uses Ultralytics, which is licensed under
AGPL-3.0, so the branch is licensed under the GNU Affero General Public License v3.0 (`LICENSE`).
