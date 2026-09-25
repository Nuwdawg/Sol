# exp058: first Mask2Former, TPU, 1024 px, about 4 epochs

## What was tried and the decisive result

From `book/chapters/appB_ledger.tex`:

| ledger row | what was tried | decisive result |
|---|---|---|
| exp046–058 | first Mask2Former, TPU, 1024 px, about 4 epochs | 0.300, then 0.247; public 0.20 |

## How to run

Run from the repository root, so that `experiments.*` and `shared.*` import.
- Kaggle kernel: `cd kaggle/exp058_mask2_continue && kaggle kernels push -p .`

## Copied files

These files are copies. The original lives on the branch named.

- `kaggle/exp046_mask2_train/train.py` from `exp/046-mask2`
- `shared/__init__.py` from shared module (original in `main`)
- `shared/data_split.py` from shared module (original in `main`)
- `shared/mask2_checkpointing.py` from shared module (original in `main`)
- `shared/query_mask_loss.py` from shared module (original in `main`)

## Experiment log

- No section heading names this experiment. Search `docs/ACTIVE_EXPERIMENT.md` and `docs/leaderboard-analysis.md` for `exp058`.

## Dependencies not in requirements.txt

The root `requirements.txt` does not list these imported packages: `scipy`, `torch_xla`, `transformers`. No versions are pinned here.

## Licence

MIT, see `LICENSE`.
