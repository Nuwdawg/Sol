# exp063: public probes of the variants

## What was tried and the decisive result

From `book/chapters/appB_ledger.tex`:

| ledger row | what was tried | decisive result |
|---|---|---|
| exp063 | public probes of the variants | 0.37, 0.36, 0.38, 0.20 |

## How to run

Run from the repository root, so that `experiments.*` and `shared.*` import.
- `python scripts/existing_weight_test.py`
- `python scripts/submit_authorized_probes.py`
- Kaggle kernel: `cd kaggle/exp063_submission_probes && kaggle kernels push -p .`
- Tests: `python -m pytest tests/test_inference_sweep.py tests/test_pq_matches_official.py tests/test_utils.py`

## Copied files

These files are copies. The original lives on the branch named.

- `experiments/exp_007_tta/src/dihedral.py` from `exp/007-tta`
- `experiments/exp_033_inference/__init__.py` from `exp/033-inference`
- `experiments/exp_033_inference/src/__init__.py` from `exp/033-inference`
- `experiments/exp_033_inference/src/cached_tta.py` from `exp/033-inference`
- `experiments/exp_033_inference/src/sweep.py` from `exp/033-inference`
- `scripts/existing_weight_trials.py` from `exp/059-soft`
- `shared/__init__.py` from shared module (original in `main`)
- `shared/data_split.py` from shared module (original in `main`)
- `shared/utils.py` from shared module (original in `main`)
- `experiments/exp_046_mask2/src/evaluate.py` from `exp/046-mask2`
- `scripts/download_kernel.py` from `main`

## Experiment log

- No section heading names this experiment. Search `docs/ACTIVE_EXPERIMENT.md` and `docs/leaderboard-analysis.md` for `exp063`.

## Dependencies not in requirements.txt

The root `requirements.txt` does not list these imported packages: `kagglesdk`, `pytest`, `requests`, `transformers`, `ultralytics`. No versions are pinned here.

## Licence

Copyright (c) 2026 Ismail Kasozi. This branch uses Ultralytics, which is licensed under
AGPL-3.0, so the branch is licensed under the GNU Affero General Public License v3.0 (`LICENSE`).
