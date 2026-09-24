# exp001: U-Net++ (EfficientNet-B4) at 512, semantic mask then connected components

## What was tried and the decisive result

From `book/chapters/appB_ledger.tex`:

| ledger row | what was tried | decisive result |
|---|---|---|
| exp001 | U-Net++ (EfficientNet-B4) at 512, semantic mask then connected components | 0.2957; public 0.26 |

## How to run

Run from the repository root, so that `experiments.*` and `shared.*` import.
- `python experiments/exp_001_baseline/src/ceiling_analysis.py`
- `python experiments/exp_001_baseline/src/evaluate.py`
- `python experiments/exp_001_baseline/src/predict.py`
- `python experiments/exp_001_baseline/src/prepare_cache.py`
- `python experiments/exp_001_baseline/src/train.py`
- `python kaggle/exp001_kaggle.py`
- notebook: notebooks/colab_exp001.ipynb
- Tests: `python -m pytest tests/test_pq_matches_official.py tests/test_utils.py`

## Copied files

These files are copies. The original lives on the branch named.

- `shared/__init__.py` from shared module (original in `main`)
- `shared/data_split.py` from shared module (original in `main`)
- `shared/preprocessing.py` from shared module (original in `main`)
- `shared/utils.py` from shared module (original in `main`)

## Experiment log

- `docs/leaderboard-analysis.md` line 94: exp_001 — U-Net++ semantic + connected components

## Dependencies not in requirements.txt

The root `requirements.txt` does not list these imported packages: `pytest`. No versions are pinned here.

## Licence

MIT, see `LICENSE`.
