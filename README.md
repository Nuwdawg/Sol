# exp032: GPU kernel: the run exp_010 was supposed to be

## What was tried and the decisive result

The ledger has no row for this experiment. See the log sections below.

## How to run

Run from the repository root, so that `experiments.*` and `shared.*` import.
- `python -m experiments.exp_032_chunked.src.train --help`
- `python kaggle/exp032/at1280_kernel.py`
- `python scripts/audit_resolution.py`
- `python scripts/probe_mask_loss.py`
- Kaggle kernel: `cd kaggle/exp032 && kaggle kernels push -p .`
- Tests: `python -m pytest tests/test_chunked_mask_loss.py tests/test_roi_mask_loss.py tests/test_chunked_training.py`

## Copied files

These files are copies. The original lives on the branch named.

- `shared/__init__.py` from shared module (original in `main`)
- `shared/chunked_mask_loss.py` from shared module (original in `main`)
- `shared/data_split.py` from shared module (original in `main`)
- `shared/roi_mask_loss.py` from shared module (original in `main`)
- `experiments/exp_002_yolo_seg/src/prepare_yolo.py` from `exp/002-yolo-seg`
- `experiments/exp_031_frozen/__init__.py` from `exp/031-frozen`
- `experiments/exp_031_frozen/src/__init__.py` from `exp/031-frozen`
- `experiments/exp_031_frozen/src/train.py` from `exp/031-frozen`
- `scripts/probe_training.py` from `exp/067-chunked-probe`

## Experiment log

- No section heading names this experiment. Search `docs/ACTIVE_EXPERIMENT.md` and `docs/leaderboard-analysis.md` for `exp032`.

## Dependencies not in requirements.txt

The root `requirements.txt` does not list these imported packages: `psutil`, `pytest`, `shapely`, `ultralytics`. No versions are pinned here.

## Licence

Copyright (c) 2026 Ismail Kasozi. This branch uses Ultralytics, which is licensed under
AGPL-3.0, so the branch is licensed under the GNU Affero General Public License v3.0 (`LICENSE`).
