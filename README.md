# exp081: semantic support admits 77 masks

## What was tried and the decisive result

From `book/chapters/appB_ledger.tex`:

| ledger row | what was tried | decisive result |
|---|---|---|
| exp081, 082 | semantic support admits 77 masks | 0.4594; public 0.38 |

## How to run

Run from the repository root, so that `experiments.*` and `shared.*` import.
- Kaggle kernel: `cd kaggle/exp081_semantic_support_shortlist_t4 && kaggle kernels push -p .`

## Copied files

These files are copies. The original lives on the branch named.

- `experiments/exp_033_inference/__init__.py` from `exp/033-inference`
- `experiments/exp_033_inference/src/__init__.py` from `exp/033-inference`
- `experiments/exp_033_inference/src/sweep.py` from `exp/033-inference`
- `scripts/evaluate_semantic_complement.py` from `exp/077-semantic-complement`
- `shared/__init__.py` from shared module (original in `main`)
- `shared/data_split.py` from shared module (original in `main`)
- `shared/utils.py` from shared module (original in `main`)

## Experiment log

- No section heading names this experiment. Search `docs/ACTIVE_EXPERIMENT.md` and `docs/leaderboard-analysis.md` for `exp081`.

## Dependencies not in requirements.txt

The root `requirements.txt` does not list these imported packages: `torch_xla`, `ultralytics`. No versions are pinned here.

## Licence

Copyright (c) 2026 Ismail Kasozi. This branch uses Ultralytics, which is licensed under
AGPL-3.0, so the branch is licensed under the GNU Affero General Public License v3.0 (`LICENSE`).
