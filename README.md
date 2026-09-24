# exp013: error decomposition with oracles

## What was tried and the decisive result

From `book/chapters/appB_ledger.tex`:

| ledger row | what was tried | decisive result |
|---|---|---|
| exp013 | error decomposition with oracles | orphans: 49% of FP, 59% of FN |

## How to run

Run from the repository root, so that `experiments.*` and `shared.*` import.
- `python -m experiments.exp_013_errors.src.decompose --help`
- `python -m experiments.exp_013_errors.src.multiplicity --help`
- `python -m experiments.exp_013_errors.src.recall_ceiling --help`
- `python kaggle/exp013/decompose_kernel.py`
- Kaggle kernel: `cd kaggle/exp013 && kaggle kernels push -p .`
- Tests: `python -m pytest tests/test_pq_matches_official.py tests/test_utils.py`

## Copied files

These files are copies. The original lives on the branch named.

- `shared/__init__.py` from shared module (original in `main`)
- `shared/data_split.py` from shared module (original in `main`)
- `shared/utils.py` from shared module (original in `main`)

## Experiment log

- `docs/leaderboard-analysis.md` line 1133: exp_013 — what the 456 false positives are actually made of

## Dependencies not in requirements.txt

The root `requirements.txt` does not list these imported packages: `pytest`. No versions are pinned here.

## Licence

MIT, see `LICENSE`.
