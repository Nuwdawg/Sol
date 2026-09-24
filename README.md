# exp014: orphan rates of the model and of annotators

## What was tried and the decisive result

From `book/chapters/appB_ledger.tex`:

| ledger row | what was tried | decisive result |
|---|---|---|
| exp014 | orphan rates of the model and of annotators | 19.8% against 19.4% |

## How to run

Run from the repository root, so that `experiments.*` and `shared.*` import.
- `python -m experiments.exp_014_annotator.src.irreducible --help`
- `python kaggle/exp014/irreducible_kernel.py`
- Kaggle kernel: `cd kaggle/exp014 && kaggle kernels push -p .`
- Tests: `python -m pytest tests/test_pq_matches_official.py tests/test_utils.py`

## Copied files

These files are copies. The original lives on the branch named.

- `shared/__init__.py` from shared module (original in `main`)
- `shared/data_split.py` from shared module (original in `main`)
- `shared/utils.py` from shared module (original in `main`)

## Experiment log

- `docs/leaderboard-analysis.md` line 1245: exp_014 — the phantom filaments are the label noise floor

## Dependencies not in requirements.txt

The root `requirements.txt` does not list these imported packages: `pytest`. No versions are pinned here.

## Licence

MIT, see `LICENSE`.
