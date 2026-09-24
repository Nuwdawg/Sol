# exp021: second YOLO as a veto on the first

## What was tried and the decisive result

From `book/chapters/appB_ledger.tex`:

| ledger row | what was tried | decisive result |
|---|---|---|
| exp021, 023 | second YOLO as a veto on the first | +0.0007; public 0.36; deployment bug caught |

## How to run

Run from the repository root, so that `experiments.*` and `shared.*` import.
- `python -m experiments.exp_021_ensemble.src.blend --help`
- `python -m experiments.exp_021_ensemble.src.fuse --help`
- `python -m experiments.exp_021_ensemble.src.predict_fused --help`
- `python kaggle/exp021/fuse_kernel.py`
- Kaggle kernel: `cd kaggle/exp021 && kaggle kernels push -p .`
- Tests: `python -m pytest tests/test_pq_matches_official.py tests/test_utils.py`

## Copied files

These files are copies. The original lives on the branch named.

- `shared/__init__.py` from shared module (original in `main`)
- `shared/data_split.py` from shared module (original in `main`)
- `shared/utils.py` from shared module (original in `main`)

## Experiment log

- `docs/leaderboard-analysis.md` line 1706: exp_021 — the two detectors make the same mistakes (negative result)

## Dependencies not in requirements.txt

The root `requirements.txt` does not list these imported packages: `pytest`, `ultralytics`. No versions are pinned here.

## Licence

Copyright (c) 2026 Ismail Kasozi. This branch uses Ultralytics, which is licensed under
AGPL-3.0, so the branch is licensed under the GNU Affero General Public License v3.0 (`LICENSE`).
