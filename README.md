# exp100: Package the Kaggle T4 kernel that emits a submission from an exp097 checkpoint

## What was tried and the decisive result

From `book/chapters/appB_ledger.tex`:

| ledger row | what was tried | decisive result |
|---|---|---|
| exp098, 100 | scoring exp097's checkpoints | D4 0.4668; public 0.39 |

## How to run

Run from the repository root, so that `experiments.*` and `shared.*` import.
- `python scripts/package_exp097_submission.py`
- Kaggle kernel: `cd kaggle/exp100_exp097_submission_t4 && kaggle kernels push -p .`

## Copied files

These files are copies. The original lives on the branch named.

- `experiments/exp_007_tta/src/dihedral.py` from `exp/007-tta`
- `experiments/exp_033_inference/__init__.py` from `exp/033-inference`
- `experiments/exp_033_inference/src/__init__.py` from `exp/033-inference`
- `experiments/exp_033_inference/src/cached_tta.py` from `exp/033-inference`
- `experiments/exp_033_inference/src/sweep.py` from `exp/033-inference`
- `scripts/emit_exp097_submission.py` from `exp/097-runpod-yolo`
- `shared/__init__.py` from shared module (original in `main`)
- `shared/data_split.py` from shared module (original in `main`)
- `shared/utils.py` from shared module (original in `main`)

## Experiment log

- `docs/ACTIVE_EXPERIMENT.md` line 1362: exp100: stage1 submitted at the user's direction (2026-09-16 06:12 UTC)
- `docs/ACTIVE_EXPERIMENT.md` line 1385: exp100 scored 0.38 public, and it calibrates the new model (2026-09-16 06:41 UTC)

## Dependencies not in requirements.txt

The root `requirements.txt` does not list these imported packages: `ultralytics`. No versions are pinned here.

## Licence

Copyright (c) 2026 Ismail Kasozi. This branch uses Ultralytics, which is licensed under
AGPL-3.0, so the branch is licensed under the GNU Affero General Public License v3.0 (`LICENSE`).
