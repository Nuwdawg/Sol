# exp141: soup; epoch 50; eight-view fusion

## What was tried and the decisive result

From `book/chapters/appB_ledger.tex`:

| ledger row | what was tried | decisive result |
|---|---|---|
| exp141, 143, 144 | soup; epoch 50; eight-view fusion | 0.4688, 0.4621, 0.4700; public 0.40 each |

## How to run

Run from the repository root, so that `experiments.*` and `shared.*` import.
- `python -m experiments.exp_141_soup_and_tta.src.emit_fusion --help`
- `python -m experiments.exp_141_soup_and_tta.src.soup_tta --help`
- `python scripts/package_exp141_soup_tta.py`
- Kaggle kernel: `cd kaggle/exp141_soup_tta_t4 && kaggle kernels push -p .`
- Tests: `python -m pytest tests/test_pq_matches_official.py tests/test_utils.py`

## Copied files

These files are copies. The original lives on the branch named.

- `experiments/exp_033_inference/__init__.py` from `exp/033-inference`
- `experiments/exp_033_inference/src/__init__.py` from `exp/033-inference`
- `experiments/exp_033_inference/src/cached_tta.py` from `exp/033-inference`
- `experiments/exp_033_inference/src/sweep.py` from `exp/033-inference`
- `experiments/exp_139_mask2former/__init__.py` from `main`
- `experiments/exp_139_mask2former/src/__init__.py` from `main`
- `experiments/exp_139_mask2former/src/data.py` from `main`
- `experiments/exp_139_mask2former/src/evaluate.py` from `main`
- `experiments/exp_139_mask2former/src/fast_ops.py` from `main`
- `experiments/exp_139_mask2former/src/model.py` from `main`
- `experiments/exp_139_mask2former/src/submit.py` from `main`
- `shared/__init__.py` from shared module (original in `main`)
- `shared/data_split.py` from shared module (original in `main`)
- `shared/utils.py` from shared module (original in `main`)
- `experiments/exp_139_mask2former/tests/__init__.py` from `main`
- `experiments/exp_139_mask2former/tests/reference_data.py` from `main`
- `experiments/exp_139_mask2former/tests/test_equivalence.py` from `main`
- `scripts/package_exp139_gpu_tests.py` from `main`

## Experiment log

- `docs/ACTIVE_EXPERIMENT.md` line 3505: exp141 result: neither the soup nor a re-tuned fusion helps (2026-09-18)

## Dependencies not in requirements.txt

The root `requirements.txt` does not list these imported packages: `kernels`, `pytest`, `scipy`, `transformers`, `ultralytics`. No versions are pinned here.

## Licence

Copyright (c) 2026 Ismail Kasozi. This branch uses Ultralytics, which is licensed under
AGPL-3.0, so the branch is licensed under the GNU Affero General Public License v3.0 (`LICENSE`).
