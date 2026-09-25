# exp038: Probe real YOLO segmentation optimizer steps on TPU; GPU disabled

## What was tried and the decisive result

The ledger has no row for this experiment. See the log sections below.

## How to run

Run from the repository root, so that `experiments.*` and `shared.*` import.
- `python kaggle/exp038_yolo_tpu_probe/probe.py`
- Kaggle kernel: `cd kaggle/exp038_yolo_tpu_probe && kaggle kernels push -p .`

## Copied files

These files are copies. The original lives on the branch named.

- `experiments/exp_002_yolo_seg/src/prepare_yolo.py` from `exp/002-yolo-seg`
- `shared/__init__.py` from shared module (original in `main`)
- `shared/data_split.py` from shared module (original in `main`)

## Experiment log

- No section heading names this experiment. Search `docs/ACTIVE_EXPERIMENT.md` and `docs/leaderboard-analysis.md` for `exp038`.

## Dependencies not in requirements.txt

The root `requirements.txt` does not list these imported packages: `shapely`, `torch_xla`, `ultralytics`. No versions are pinned here.

## Licence

Copyright (c) 2026 Ismail Kasozi. This branch uses Ultralytics, which is licensed under
AGPL-3.0, so the branch is licensed under the GNU Affero General Public License v3.0 (`LICENSE`).
