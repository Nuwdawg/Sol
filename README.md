# exp005: centre and offset grouping, ablated with noisy offsets

## What was tried and the decisive result

From `book/chapters/appB_ledger.tex`:

| ledger row | what was tried | decisive result |
|---|---|---|
| exp005 | centre and offset grouping, ablated with noisy offsets | ≤ 0; dropped |
| exp005 | erode every mask by one pixel, confidence 0.35 | 0.4404; public 0.36 |
| exp005 | gradient-boosted emission over shape features | -0.017 against confidence |

## How to run

Run from the repository root, so that `experiments.*` and `shared.*` import.
- `python experiments/exp_005_center_offset/src/grouping_ablation.py`
- `python experiments/exp_005_center_offset/src/prepare.py`
- `python experiments/exp_005_postproc/src/calibrate.py`
- `python experiments/exp_005_postproc/src/nearmiss.py`
- `python kaggle/exp005/nearmiss_kernel.py`
- Kaggle kernel: `cd kaggle/exp005 && kaggle kernels push -p .`
- Tests: `python -m pytest tests/test_pq_matches_official.py tests/test_utils.py`

## Copied files

These files are copies. The original lives on the branch named.

- `shared/__init__.py` from shared module (original in `main`)
- `shared/data_split.py` from shared module (original in `main`)
- `shared/solar_disk.py` from shared module (original in `main`)
- `shared/utils.py` from shared module (original in `main`)
- `experiments/exp_002_yolo_seg/src/predict.py` from `exp/002-yolo-seg`

## Experiment log

- `docs/leaderboard-analysis.md` line 573: exp_005 — mask erosion: 0.33 to 0.36, rank 206 to 85

## Dependencies not in requirements.txt

The root `requirements.txt` does not list these imported packages: `pytest`, `sklearn`, `ultralytics`. No versions are pinned here.

## Licence

Copyright (c) 2026 Ismail Kasozi. This branch uses Ultralytics, which is licensed under
AGPL-3.0, so the branch is licensed under the GNU Affero General Public License v3.0 (`LICENSE`).
