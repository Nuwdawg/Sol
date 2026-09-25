# exp026: crop verifier labelled by "any annotator"

## What was tried and the decisive result

From `book/chapters/appB_ledger.tex`:

| ledger row | what was tried | decisive result |
|---|---|---|
| exp025–027 | crop verifier labelled by "any annotator" | 72% versus 17% precision |

## How to run

Run from the repository root, so that `experiments.*` and `shared.*` import.
- `python kaggle/exp026/train_kernel.py`
- Kaggle kernel: `cd kaggle/exp026 && kaggle kernels push -p .`

## Copied files

These files are copies. The original lives on the branch named.

- `experiments/exp_025_verifier/__init__.py` from `exp/025-verifier`
- `experiments/exp_025_verifier/src/__init__.py` from `exp/025-verifier`
- `experiments/exp_025_verifier/src/model.py` from `exp/025-verifier`
- `experiments/exp_025_verifier/src/train_verifier.py` from `exp/025-verifier`

## Experiment log

- `docs/leaderboard-analysis.md` line 1819: exp_025-027 — the verifier is good at the wrong question (negative result)

## Dependencies not in requirements.txt

The root `requirements.txt` does not list these imported packages: `torch_xla`. No versions are pinned here.

## Licence

MIT, see `LICENSE`.
