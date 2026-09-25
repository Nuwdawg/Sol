# exp093: refiner v2 on harvested proposals

## What was tried and the decisive result

From `book/chapters/appB_ledger.tex`:

| ledger row | what was tried | decisive result |
|---|---|---|
| exp091–094 | refiner v2 on harvested proposals | a resampling artefact; -0.0023 |

## How to run

Run from the repository root, so that `experiments.*` and `shared.*` import.

## Copied files

These files are copies. The original lives on the branch named.

- `experiments/exp_033_inference/__init__.py` from `exp/033-inference`
- `experiments/exp_033_inference/src/__init__.py` from `exp/033-inference`
- `experiments/exp_033_inference/src/sweep.py` from `exp/033-inference`
- `experiments/exp_085_morden_spine/__init__.py` from `exp/085-morden-spine`
- `experiments/exp_085_morden_spine/src/__init__.py` from `exp/085-morden-spine`
- `experiments/exp_085_morden_spine/src/model.py` from `exp/085-morden-spine`
- `experiments/exp_085_morden_spine/src/prepare.py` from `exp/085-morden-spine`
- `experiments/exp_092_refiner/__init__.py` from `exp/092-refiner`
- `experiments/exp_092_refiner/src/__init__.py` from `exp/092-refiner`
- `experiments/exp_092_refiner/src/crops.py` from `exp/092-refiner`
- `experiments/exp_092_refiner/src/deploy.py` from `exp/092-refiner`
- `experiments/exp_092_refiner/src/evaluate.py` from `exp/092-refiner`
- `experiments/exp_092_refiner/src/model.py` from `exp/092-refiner`
- `shared/__init__.py` from shared module (original in `main`)
- `shared/data_split.py` from shared module (original in `main`)
- `shared/utils.py` from shared module (original in `main`)

## Experiment log

- `docs/ACTIVE_EXPERIMENT.md` line 870: 2026-09-15 20:05 Nairobi, exp092 refiner running; exp093 deploy prepared

## Dependencies not in requirements.txt

The root `requirements.txt` does not list these imported packages: `ultralytics`. No versions are pinned here.

## Licence

Copyright (c) 2026 Ismail Kasozi. This branch uses Ultralytics, which is licensed under
AGPL-3.0, so the branch is licensed under the GNU Affero General Public License v3.0 (`LICENSE`).
