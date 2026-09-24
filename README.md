# exp099: YOLOv8l-seg at 2048 on an L40, convergence control, T4 rehearsal

## What was tried and the decisive result

From `book/chapters/appB_ledger.tex`:

| ledger row | what was tried | decisive result |
|---|---|---|
| exp097, 099 | YOLOv8l-seg at 2048 on an L40, convergence control, T4 rehearsal | converged at epoch 124; USD 2.74 |

## How to run

Run from the repository root, so that `experiments.*` and `shared.*` import.
- `python scripts/package_exp099_smoke.py`
- Kaggle kernel: `cd kaggle/exp099_exp097_smoke_t4 && kaggle kernels push -p .`

## Experiment log

- `docs/ACTIVE_EXPERIMENT.md` line 1187: exp099 smoke result (2026-09-16, Kaggle T4, free compute)

## Licence

Copyright (c) 2026 Ismail Kasozi. This branch uses Ultralytics, which is licensed under
AGPL-3.0, so the branch is licensed under the GNU Affero General Public License v3.0 (`LICENSE`).
