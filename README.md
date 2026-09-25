# exp030: pixel averaging of two YOLO models' masks

## What was tried and the decisive result

From `book/chapters/appB_ledger.tex`:

| ledger row | what was tried | decisive result |
|---|---|---|
| exp030 | pixel averaging of two YOLO models' masks | -0.0005 |

## How to run

Run from the repository root, so that `experiments.*` and `shared.*` import.
- `python kaggle/exp030/blend_kernel.py`
- Kaggle kernel: `cd kaggle/exp030 && kaggle kernels push -p .`

## Copied files

These files are copies. The original lives on the branch named.

- `experiments/exp_021_ensemble/__init__.py` from `exp/021-ensemble`
- `experiments/exp_021_ensemble/src/__init__.py` from `exp/021-ensemble`
- `experiments/exp_021_ensemble/src/blend.py` from `exp/021-ensemble`
- `shared/__init__.py` from shared module (original in `main`)
- `shared/data_split.py` from shared module (original in `main`)
- `shared/utils.py` from shared module (original in `main`)

## Experiment log

- `docs/leaderboard-analysis.md` line 1972: exp_030 — averaging the two models' masks (negative result)

## Licence

MIT, see `LICENSE`.
