# exp087: semantic and spine model after Hu et al.

## What was tried and the decisive result

From `book/chapters/appB_ledger.tex`:

| ledger row | what was tried | decisive result |
|---|---|---|
| exp085, 087 | semantic and spine model after Hu et al. | PQ 0.3227 on its own |

## How to run

Run from the repository root, so that `experiments.*` and `shared.*` import.
- Kaggle kernel: `cd kaggle/exp087_morden_continue_t4 && kaggle kernels push -p .`

## Copied files

These files are copies. The original lives on the branch named.

- `experiments/exp_004_spine_tpu/src/decompose.py` from `exp/004-spine-tpu`
- `experiments/exp_033_inference/__init__.py` from `exp/033-inference`
- `experiments/exp_033_inference/src/__init__.py` from `exp/033-inference`
- `experiments/exp_033_inference/src/sweep.py` from `exp/033-inference`
- `experiments/exp_085_morden_spine/__init__.py` from `exp/085-morden-spine`
- `experiments/exp_085_morden_spine/src/__init__.py` from `exp/085-morden-spine`
- `experiments/exp_085_morden_spine/src/evaluate.py` from `exp/085-morden-spine`
- `experiments/exp_085_morden_spine/src/model.py` from `exp/085-morden-spine`
- `experiments/exp_085_morden_spine/src/prepare.py` from `exp/085-morden-spine`
- `experiments/exp_085_morden_spine/src/train.py` from `exp/085-morden-spine`
- `shared/__init__.py` from shared module (original in `main`)
- `shared/data_split.py` from shared module (original in `main`)
- `shared/utils.py` from shared module (original in `main`)

## Experiment log

- `docs/ACTIVE_EXPERIMENT.md` line 848: 2026-09-15 19:35 Nairobi, exp087 continuation complete

## Dependencies not in requirements.txt

The root `requirements.txt` does not list these imported packages: `ultralytics`. No versions are pinned here.

## Licence

Copyright (c) 2026 Ismail Kasozi. This branch uses Ultralytics, which is licensed under
AGPL-3.0, so the branch is licensed under the GNU Affero General Public License v3.0 (`LICENSE`).
