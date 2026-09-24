# exp085: semantic and spine model after Hu et al.

## What was tried and the decisive result

From `book/chapters/appB_ledger.tex`:

| ledger row | what was tried | decisive result |
|---|---|---|
| exp085, 087 | semantic and spine model after Hu et al. | PQ 0.3227 on its own |

## How to run

Run from the repository root, so that `experiments.*` and `shared.*` import.
- `python -m experiments.exp_085_morden_spine.src.evaluate --help`
- `python -m experiments.exp_085_morden_spine.src.prepare --help`
- `python -m experiments.exp_085_morden_spine.src.train --help`
- `python scripts/audit_morden_fusion.py`
- `python scripts/score_submission_instances.py`
- Kaggle kernel: `cd kaggle/exp085_morden_spine_t4 && kaggle kernels push -p .`
- Tests: `python -m pytest tests/test_pq_matches_official.py tests/test_utils.py`

## Copied files

These files are copies. The original lives on the branch named.

- `experiments/exp_004_spine_tpu/src/decompose.py` from `exp/004-spine-tpu`
- `experiments/exp_033_inference/__init__.py` from `exp/033-inference`
- `experiments/exp_033_inference/src/__init__.py` from `exp/033-inference`
- `experiments/exp_033_inference/src/sweep.py` from `exp/033-inference`
- `shared/__init__.py` from shared module (original in `main`)
- `shared/data_split.py` from shared module (original in `main`)
- `shared/utils.py` from shared module (original in `main`)

## Experiment log

- `docs/ACTIVE_EXPERIMENT.md` line 787: exp086: exp085 semantics as pruning evidence

## Dependencies not in requirements.txt

The root `requirements.txt` does not list these imported packages: `pytest`, `ultralytics`. No versions are pinned here.

## Licence

Copyright (c) 2026 Ismail Kasozi. This branch uses Ultralytics, which is licensed under
AGPL-3.0, so the branch is licensed under the GNU Affero General Public License v3.0 (`LICENSE`).

Parts of the model follow MORDEN (Hu et al.). Its MIT licence is in `third_party/MORDEN_LICENSE`.
