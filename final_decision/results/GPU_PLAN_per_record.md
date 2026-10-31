# GPU plan for the per-record board: a better anchor model (drafted 2026-10-31, not run)

**Why GPU.** Six submissions print 0.41: exp139, exp162, exp162b, exp169, exp169 + guard, and the
YOLO-free file. Their per-record validation ranges from 0.4705 to 0.4889. Rules tuned on validation reach
the board at about 0.845 x validation, so a rule needs validation of about 0.491 to print 0.42. Every
rule change tried on the existing models stays at or below exp169's 0.4889 under cross-validation
(`exp200/README.md`, `m2f_only/README.md`). exp169's rule adds +0.018 to its anchor model (exp139
0.4705 -> 0.4889). A better anchor is the one lever left, and it needs training.

## Run: exp139's recipe on all 707 labelled photographs

- Pod: RunPod H100, the exp139 template (`experiments/exp_139_mask2former/pod/`), about 2 h, about $5
  of the $12 credit. Two free Kaggle T4 sessions also work but need the resume path (CLAUDE.md
  section 8).
- Environment, otherwise as exp139:
  ```
  SOL_FOLD=all
  SOL_TARGETS=record
  SOL_EPOCHS=50
  SOL_EVAL_EVERY=0
  SOL_SAVE_EPOCHS="40 50"
  ```
  - `SOL_FOLD=all` adds the 106 validation photographs to the 601.
  - `SOL_TARGETS=record` keeps per-annotator targets; pooled targets are refuted (CLAUDE.md section 17).
  - `SOL_EPOCHS=50` is exp139's schedule; its epoch 40 was best.
  - `SOL_EVAL_EVERY=0`: there is no held-out set left to score.
- Within the first minute, read back `DATA` and `TARGETS` in the log (CLAUDE.md section 1): records
  1,154, kind "record".

## Build and judge (no held-out set exists, so the board judges)

1. Cache the new epoch 40 on the 180 test photographs, with all 11 views:
   `reproduce/run.py --stages m2f` with the new weights. About 40 min on two T4s; CPU is too slow.
2. Write two files:
   - (a) the single full view at exp139's fixed rule (conf 0.85, area 100);
   - (b) exp169's rule with the new anchor: constants unchanged, plus the guard.
3. Upload both. Adopt a file only if it prints 0.42 or more. Otherwise keep exp169 + guard and exp139.

## Honest odds: low

- Against: training on the validation photographs lifted the score on those same photographs by only
  +0.002 (exp150). The model fits its own training photographs no better than unseen ones, so data
  is probably not what limits it (decision brief, point c).
- For: +17% photographs normally gives a small real gain. File (b) needs only about +0.003 at the
  anchor to cross 0.415, if exp169's +0.018 rule gain carries over.
- Rough chance that either file prints 0.42: one in five. Not worth $5 unless a 0.42 is worth that
  to you.
