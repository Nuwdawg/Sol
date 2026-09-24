# exp092: refiner v2 on harvested proposals

## What was tried and the decisive result

From `book/chapters/appB_ledger.tex`:

| ledger row | what was tried | decisive result |
|---|---|---|
| exp091–094 | refiner v2 on harvested proposals | a resampling artefact; -0.0023 |

## How to run

Run from the repository root, so that `experiments.*` and `shared.*` import.
- `python -m experiments.exp_092_refiner.src.deploy --help`
- `python -m experiments.exp_092_refiner.src.diagnose --help`
- `python -m experiments.exp_092_refiner.src.evaluate --help`
- `python -m experiments.exp_092_refiner.src.prepare --help`
- `python -m experiments.exp_092_refiner.src.train --help`
- Kaggle kernel: `cd kaggle/exp092_refiner_t4 && kaggle kernels push -p .`
- Tests: `python -m pytest tests/test_pq_matches_official.py tests/test_refiner_identity.py tests/test_utils.py`

## Copied files

These files are copies. The original lives on the branch named.

- `experiments/exp_033_inference/__init__.py` from `exp/033-inference`
- `experiments/exp_033_inference/src/__init__.py` from `exp/033-inference`
- `experiments/exp_033_inference/src/sweep.py` from `exp/033-inference`
- `experiments/exp_085_morden_spine/__init__.py` from `exp/085-morden-spine`
- `experiments/exp_085_morden_spine/src/__init__.py` from `exp/085-morden-spine`
- `experiments/exp_085_morden_spine/src/model.py` from `exp/085-morden-spine`
- `experiments/exp_085_morden_spine/src/prepare.py` from `exp/085-morden-spine`
- `shared/__init__.py` from shared module (original in `main`)
- `shared/data_split.py` from shared module (original in `main`)
- `shared/utils.py` from shared module (original in `main`)

## Experiment log

- `docs/ACTIVE_EXPERIMENT.md` line 870: 2026-09-15 20:05 Nairobi, exp092 refiner running; exp093 deploy prepared
- `docs/ACTIVE_EXPERIMENT.md` line 901: 2026-09-15 21:17 Nairobi, exp092 result: refiner does not move canonical PQ
- `docs/ACTIVE_EXPERIMENT.md` line 981: exp094 result: the exp092 crop gain was a resampling artifact

## Dependencies not in requirements.txt

The root `requirements.txt` does not list these imported packages: `pytest`, `ultralytics`. No versions are pinned here.

## Licence

Copyright (c) 2026 Ismail Kasozi. This branch uses Ultralytics, which is licensed under
AGPL-3.0, so the branch is licensed under the GNU Affero General Public License v3.0 (`LICENSE`).

Parts of the model follow MORDEN (Hu et al.). Its MIT licence is in `third_party/MORDEN_LICENSE`.
