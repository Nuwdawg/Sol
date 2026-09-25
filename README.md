# exp023: second YOLO as a veto on the first

## What was tried and the decisive result

From `book/chapters/appB_ledger.tex`:

| ledger row | what was tried | decisive result |
|---|---|---|
| exp021, 023 | second YOLO as a veto on the first | +0.0007; public 0.36; deployment bug caught |

## How to run

Run from the repository root, so that `experiments.*` and `shared.*` import.
- `python kaggle/exp023/fused_kernel.py`
- Kaggle kernel: `cd kaggle/exp023 && kaggle kernels push -p .`

## Copied files

These files are copies. The original lives on the branch named.

- `experiments/exp_021_ensemble/__init__.py` from `exp/021-ensemble`
- `experiments/exp_021_ensemble/src/__init__.py` from `exp/021-ensemble`
- `experiments/exp_021_ensemble/src/predict_fused.py` from `exp/021-ensemble`
- `shared/__init__.py` from shared module (original in `main`)
- `shared/utils.py` from shared module (original in `main`)

## Experiment log

- `docs/leaderboard-analysis.md` line 1784: exp_023 — the fusion scores 0.36 in public, as its margin predicted

## Dependencies not in requirements.txt

The root `requirements.txt` does not list these imported packages: `ultralytics`. No versions are pinned here.

## Licence

Copyright (c) 2026 Ismail Kasozi. This branch uses Ultralytics, which is licensed under
AGPL-3.0, so the branch is licensed under the GNU Affero General Public License v3.0 (`LICENSE`).
