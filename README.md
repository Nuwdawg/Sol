# exp015: CPU kernel: pick an operating point on validation, then predict the test set

## What was tried and the decisive result

From `book/chapters/appB_ledger.tex`:

| ledger row | what was tried | decisive result |
|---|---|---|
| exp015 | CPU inference path | bit-identical to GPU |

## How to run

Run from the repository root, so that `experiments.*` and `shared.*` import.
- `python kaggle/exp015/submit_kernel.py`
- `python scripts/probe_onnx_cpu.py`
- Kaggle kernel: `cd kaggle/exp015 && kaggle kernels push -p .`

## Copied files

These files are copies. The original lives on the branch named.

- `experiments/exp_002_yolo_seg/src/predict.py` from `exp/002-yolo-seg`
- `experiments/exp_005_postproc/src/nearmiss.py` from `exp/005-center-offset`
- `shared/__init__.py` from shared module (original in `main`)
- `shared/data_split.py` from shared module (original in `main`)
- `shared/solar_disk.py` from shared module (original in `main`)
- `shared/utils.py` from shared module (original in `main`)

## Experiment log

- `docs/leaderboard-analysis.md` line 1476: exp_015 — the CPU submission path is bit-for-bit the GPU one

## Dependencies not in requirements.txt

The root `requirements.txt` does not list these imported packages: `onnxruntime`, `ultralytics`. No versions are pinned here.

## Licence

Copyright (c) 2026 Ismail Kasozi. This branch uses Ultralytics, which is licensed under
AGPL-3.0, so the branch is licensed under the GNU Affero General Public License v3.0 (`LICENSE`).
