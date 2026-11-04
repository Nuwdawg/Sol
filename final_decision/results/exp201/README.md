# exp201: the deployed rule on models retrained on all 707 labelled photographs (2026-11-04)

**File:** `submission_exp201.csv` (1,196 instances, 178 photographs, SHA256 `14F581B0...`). It has not
been uploaded yet.

## What it is

The first pick's rule (`../remnant_rescue/README.md`) with every constant unchanged (threshold 0.68).
Only the two Mask2Former models change, both retrained on train + validation (1,154 annotation
records) on the user's free Kaggle T4 quota:
- **Swin-S:** exp139's recipe, seed 7, 50 epochs, in place of exp139 (checkpoint SHA256 `6abc7ac8...`);
- **Swin-B:** the rescue's second opinion, seed 11, 32 epochs (checkpoint SHA256 `bdfafb9e...`).

The two YOLO voters are unchanged. Training ran over two sessions (`kaggle/exp201/`):
- session 1: 20:34 UTC 3 Nov to about 06:30 UTC 4 Nov, stopped on budget, checkpoints kept;
- session 2: 07:08 to 16:42 UTC 4 Nov. Both runs resumed and finished their schedules (Swin-S 50 of 50
  epochs, Swin-B 32 of 32). Both then cached the 180 test photographs (`CACHE_EXIT 0` for each).

The file was built by `experiments.exp_201_alldata.src.build`. On the old models' caches the same
command rebuilt the deployed files byte for byte.

## Why the constants were not re-tuned

No photograph is left out of training, so nothing can be scored on validation. The rule's
constants come from exp139. Applied unchanged, they keep most of their gain on two other models
(`experiments/exp_201_alldata/README.md`):
- exp190b Swin-S (exp201's Swin-S recipe without the validation photographs) keeps 72-75%;
- Swin-B keeps 81%.

On exp190b Swin-S the two folds chose different thresholds (0.67, 0.76), so 0.68 stays, as
pre-registered.

## Checks

| | exp201 | current first pick (remnant + rescue) |
|---|---|---|
| single view (guard's base) | 1,282 | 1,192 |
| exp169 + guard | 1,203 (guard re-adds 20) | 1,150 (guard re-adds 9) |
| final file | **1,196** on 178 photographs | 1,149 on 179 photographs |

- No overlapping masks (checked before writing); at most 17 instances on one photograph.
- Native-resolution review against the current first pick (`review_vs_remnant_rescue.jpg`, the
  16 largest changes; `review_vs_remnant_rescue.json`):
  - 1,117 instances match (median IoU 0.966, 10th percentile 0.927);
  - 79 are added (median 1,368 px; 2.5% near the limb) and 32 removed (median 985 px).
  - Of the 12 largest additions, all are long, clearly dark filaments. Several of them, near the
    limb and up to 14,563 px, are missing from the current file. Others extend a filament the
    current file holds only in part.
  - One added instance (20141120195834Ch) carries small disconnected islands along a gap.
  - The largest removals are a dark irregular blob, a limb-crossing filament, and the long curved
    part of a filament that the new file holds only in part.
  - No merges of separate filaments, no off-disk detections.

## Final-pick rule (pre-registered in `kaggle/exp201/README.md`)

| it prints | final picks |
|---|---|
| 0.42 or more | this file + `../remnant_rescue/submission_exp169_remnant_rescue.csv` |
| 0.41 | the same two |
| 0.40 or less | `../remnant_rescue/submission_exp169_remnant_rescue.csv` + exp139 |

If this file is picked, its two checkpoints go into `weights/` and `reproduce/run.py --stages ... exp201`
rebuilds it. The stage exists and was smoke-tested; its SHA256 constants are filled in when the
weights are added.
