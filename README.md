# exp012: refiner trained on 3,429 real detector errors

## What was tried and the decisive result

From `book/chapters/appB_ledger.tex`:

| ledger row | what was tried | decisive result |
|---|---|---|
| exp011, 012, 018 | refiner trained on 3,429 real detector errors | learned the identity; -0.031 |

## How to run

Run from the repository root, so that `experiments.*` and `shared.*` import.
- `python kaggle/exp012/refiner_real_kernel.py`
- Kaggle kernel: `cd kaggle/exp012 && kaggle kernels push -p .`

## Copied files

These files are copies. The original lives on the branch named.

- `experiments/exp_008_refiner/src/model.py` from `exp/008-refiner`
- `experiments/exp_008_refiner/src/train_refiner.py` from `exp/008-refiner`

## Experiment log

- `docs/leaderboard-analysis.md` line 1414: exp_012 — the refiner on real pairs learns the identity function (negative result)

## Dependencies not in requirements.txt

The root `requirements.txt` does not list these imported packages: `torch_xla`. No versions are pinned here.

## Licence

MIT, see `LICENSE`.
