# exp016: recall ceiling of the pool at confidence 0.05

## What was tried and the decisive result

From `book/chapters/appB_ledger.tex`:

| ledger row | what was tried | decisive result |
|---|---|---|
| exp016 | recall ceiling of the pool at confidence 0.05 | 83.2% covered, 2.1% unseen |

## How to run

Run from the repository root, so that `experiments.*` and `shared.*` import.
- `python kaggle/exp016/recall_kernel.py`
- Kaggle kernel: `cd kaggle/exp016 && kaggle kernels push -p .`

## Copied files

These files are copies. The original lives on the branch named.

- `experiments/exp_013_errors/__init__.py` from `exp/013-errors`
- `experiments/exp_013_errors/src/__init__.py` from `exp/013-errors`
- `experiments/exp_013_errors/src/recall_ceiling.py` from `exp/013-errors`
- `shared/__init__.py` from shared module (original in `main`)
- `shared/data_split.py` from shared module (original in `main`)
- `shared/utils.py` from shared module (original in `main`)

## Experiment log

- `docs/leaderboard-analysis.md` line 1279: exp_016 — the misses are seen and disbelieved, not unseen

## Licence

MIT, see `LICENSE`.
