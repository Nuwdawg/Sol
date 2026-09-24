# exp029: snapping masks to the intensity edge

## What was tried and the decisive result

From `book/chapters/appB_ledger.tex`:

| ledger row | what was tried | decisive result |
|---|---|---|
| exp029 | snapping masks to the intensity edge | destroys masks |

## How to run

Run from the repository root, so that `experiments.*` and `shared.*` import.
- `python -m experiments.exp_029_snap.src.snap --help`
- `python kaggle/exp029/snap_kernel.py`
- Kaggle kernel: `cd kaggle/exp029 && kaggle kernels push -p .`
- Tests: `python -m pytest tests/test_pq_matches_official.py tests/test_utils.py`

## Copied files

These files are copies. The original lives on the branch named.

- `shared/__init__.py` from shared module (original in `main`)
- `shared/data_split.py` from shared module (original in `main`)
- `shared/utils.py` from shared module (original in `main`)

## Experiment log

- `docs/leaderboard-analysis.md` line 1921: exp_029 — the boundary is not in the pixels (negative result)

## Dependencies not in requirements.txt

The root `requirements.txt` does not list these imported packages: `pytest`. No versions are pinned here.

## Licence

MIT, see `LICENSE`.
