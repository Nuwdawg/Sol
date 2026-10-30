# Next steps that need a GPU (planned, not run)

> **Withdrawn 2026-10-30 evening. Do not run section 1.** It rests on pooled truth, which
> exp_028 and exp097 stage1/stage2 refute (`README.md` in this folder). The leaderboard scores
> per record, so pooled targets would train the "would anyone draw this" question that sank the
> exp_026 verifier (CLAUDE.md section 4). Section 2 (all-data retrain) was already judged not
> worth it in `../../decision_brief.pdf` (in-sample about equal to held-out, exp150 and exp173b).

Everything that could run on a CPU has been run (`README.md` in this folder). Under the pooled
truth that explains the leaderboard, exp169 is within about 0.005 of the best rule its model allows.
The remaining levers change the model. Each costs about $5 of the $12 RunPod credit, or a long
free Kaggle T4 session. Run them in this order.

## 1. Retrain exp139 on pooled targets (code ready, smoke-tested on CPU)

**Why.** In photographs with several annotators, exp139 finds most filaments that only some of them
drew, but scores them low (validation, multi-annotator photographs):

| drawn by | has a candidate | median score of that candidate | kept by exp169 |
|---|---|---|---|
| 1 of 3 annotators | 69% | 0.54 | 16% |
| 1 of 2 | 71% | 0.71 | 39% |
| 2 of 3 | 95% | 0.70 | 51% |
| all of them | 97% | 0.86 | 89-93% |

Under pooled truth each of these is a full hit. Per-record training labels a 1-of-3 filament
"background" twice, so the model learns to doubt it. `--targets pooled` trains on one target per
photograph that holds every annotator's filaments, with one annotator's version kept where
versions overlap (`experiments/exp_139_mask2former/src/data.py: pooled_targets`). Used as
predictions on validation, those targets score 0.8085 pooled, so they are a faithful target.

**Honest odds.** The score already ranks candidates by how many annotators draw them, and a
lower threshold recovers part of that. The +0.005 from re-tuning is that part. What retraining
adds is consistent labels: no filament is positive in one copy of a photograph and background in
another. Expected gain: between 0 and +0.02 pooled. The case for it is a mechanism, not a
measurement.

**Run (RunPod H100 NVL, same template as exp139, `experiments/exp_139_mask2former/pod/`).**
Environment changes from the exp139 run:
```
SOL_TARGETS=pooled
SOL_EPOCHS=81          # 601 samples per epoch instead of 974: same optimiser steps as 50 epochs
SOL_EVAL_EVERY=8
SOL_SAVE_EPOCHS="57 65 73 81"
SOL_BUDGET_USD=6.50
```
Read back `TARGETS {"kind": "pooled", "records": 601, "instances": 4667}` and `DATA` in the
log within the first minute (CLAUDE.md section 1). Cost about $4.80, as exp139.

**Judge it before submitting.**
1. Cache each kept checkpoint on validation with exp147's `cache.py --no-parity --tag eNN`
   (modes `full,d4` for validation, plus the shift views, which need the exp157 driver).
2. Score with `experiments/exp_200_test_truth/src/tune.py` (pooled one-to-one, exp168's
   folds) against exp139 e40's 0.4138 for exp169's rule.
3. Adopt only if the pooled gain is at least +0.005 on both halves.
4. Then cache the test photographs (`run.py`'s `m2f` stage with the new checkpoint) and write the
   file with `emit_test.py`.

## 2. Train on every labelled photograph (707 instead of 601)

The same command with `SOL_FOLD=all SOL_EVAL_EVERY=0`. This uses 18% more data and has no
held-out check. Submit only after step 1's comparison says which targets to use.

## Not worth GPU time

- **Swin-B as a second voter:** see `swinb_pooled.json`.
- **Learned re-rankers:** lost under cross-validation.
- **Mask growth or shrinkage:** -0.02 pooled either way.
- **Extra views as new candidates:** about +0.002 each.
