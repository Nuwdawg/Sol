# exp089: Package the exp089 T4 kernel that scores test instances with exp085

## What was tried and the decisive result

The ledger has no row for this experiment. See the log sections below.

## How to run

Run from the repository root, so that `experiments.*` and `shared.*` import.
- Kaggle kernel: `cd kaggle/exp089_prune_features_t4 && kaggle kernels push -p .`

## Copied files

These files are copies. The original lives on the branch named.

- `experiments/exp_004_spine_tpu/src/decompose.py` from `exp/004-spine-tpu`
- `experiments/exp_033_inference/__init__.py` from `exp/033-inference`
- `experiments/exp_033_inference/src/__init__.py` from `exp/033-inference`
- `experiments/exp_033_inference/src/cached_tta.py` from `exp/033-inference`
- `experiments/exp_033_inference/src/sweep.py` from `exp/033-inference`
- `experiments/exp_085_morden_spine/__init__.py` from `exp/085-morden-spine`
- `experiments/exp_085_morden_spine/src/__init__.py` from `exp/085-morden-spine`
- `experiments/exp_085_morden_spine/src/evaluate.py` from `exp/085-morden-spine`
- `experiments/exp_085_morden_spine/src/model.py` from `exp/085-morden-spine`
- `experiments/exp_085_morden_spine/src/prepare.py` from `exp/085-morden-spine`
- `scripts/audit_morden_fusion.py` from `exp/085-morden-spine`
- `scripts/score_submission_instances.py` from `exp/085-morden-spine`
- `shared/__init__.py` from shared module (original in `main`)
- `shared/data_split.py` from shared module (original in `main`)
- `shared/utils.py` from shared module (original in `main`)

## Experiment log

- No section heading names this experiment. Search `docs/ACTIVE_EXPERIMENT.md` and `docs/leaderboard-analysis.md` for `exp089`.

## Dependencies not in requirements.txt

The root `requirements.txt` does not list these imported packages: `ultralytics`. No versions are pinned here.

## Licence

Copyright (c) 2026 Ismail Kasozi. This branch uses Ultralytics, which is licensed under
AGPL-3.0, so the branch is licensed under the GNU Affero General Public License v3.0 (`LICENSE`).
