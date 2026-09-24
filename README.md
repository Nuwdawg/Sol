# exp006: rasteriser comparison; half-pixel polygon offset

## What was tried and the decisive result

From `book/chapters/appB_ledger.tex`:

| ledger row | what was tried | decisive result |
|---|---|---|
| exp006 | rasteriser comparison; half-pixel polygon offset | target IoU 0.898 → 0.959 |

## How to run

Run from the repository root, so that `experiments.*` and `shared.*` import.
- `python experiments/exp_006_diagnostics/src/isolate_erosion.py`
- `python experiments/exp_006_diagnostics/src/polygon_offset.py`
- `python experiments/exp_006_diagnostics/src/rasterisation.py`
- `python scripts/show_training_sample.py`
- Tests: `python -m pytest tests/test_pq_matches_official.py tests/test_prepare_yolo.py tests/test_utils.py`

## Copied files

These files are copies. The original lives on the branch named.

- `experiments/exp_002_yolo_seg/src/prepare_yolo.py` from `exp/002-yolo-seg`
- `shared/__init__.py` from shared module (original in `main`)
- `shared/data_split.py` from shared module (original in `main`)
- `shared/utils.py` from shared module (original in `main`)

## Experiment log

- No section heading names this experiment. Search `docs/ACTIVE_EXPERIMENT.md` and `docs/leaderboard-analysis.md` for `exp006`.

## Dependencies not in requirements.txt

The root `requirements.txt` does not list these imported packages: `PIL`, `pytest`, `shapely`, `ultralytics`. No versions are pinned here.

## Licence

Copyright (c) 2026 Ismail Kasozi. This branch uses Ultralytics, which is licensed under
AGPL-3.0, so the branch is licensed under the GNU Affero General Public License v3.0 (`LICENSE`).
