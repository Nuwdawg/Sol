# exp022: CPU kernel: widen exp_010's operating-point grid past the edge it stopped at

## What was tried and the decisive result

From `book/chapters/appB_ledger.tex`:

| ledger row | what was tried | decisive result |
|---|---|---|
| exp022 | widened operating-point sweep | the old point was optimal |

## How to run

Run from the repository root, so that `experiments.*` and `shared.*` import.
- `python -m experiments.exp_022_opoint.src.sweep --help`
- `python kaggle/exp022/sweep_kernel.py`
- Kaggle kernel: `cd kaggle/exp022 && kaggle kernels push -p .`
- Tests: `python -m pytest tests/test_pq_matches_official.py tests/test_utils.py`

## Copied files

These files are copies. The original lives on the branch named.

- `shared/__init__.py` from shared module (original in `main`)
- `shared/data_split.py` from shared module (original in `main`)
- `shared/utils.py` from shared module (original in `main`)

## Experiment log

- `docs/leaderboard-analysis.md` line 1799: exp_022 — the operating point was already right (negative result)

## Dependencies not in requirements.txt

The root `requirements.txt` does not list these imported packages: `pytest`. No versions are pinned here.

## Licence

MIT, see `LICENSE`.
