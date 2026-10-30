# exp200: what the leaderboard scores against

## Correction (2026-10-30): the board scores per record, and pooled truth is refuted

The pooled one-to-one hypothesis below was fitted to seven submissions that emit much the same
things. Two pairs that differ a lot in what they emit contradict it:

| submission | public | per record (val) | pooled 1:1 (val) | 0.84 x per record |
|---|---|---|---|---|
| exp_010 detector at conf 0.30 (exp_027's baseline rule) | 0.36 | 0.4274 | 0.3543 | 0.359 |
| exp_028: the same plus exp_026 verifier promotions at 0.50 | 0.32 | 0.3865 | 0.3472 | 0.324 |
| exp097 stage2 YOLOv8l D4 conf 0.35 | 0.39 | 0.4668 | 0.3870 | 0.392 |
| exp097 stage1 YOLOv8l D4 conf 0.25 | 0.38 | 0.4632 | 0.3990 | 0.389 |
| exp063 mask2 probe (exp058 checkpoint) | 0.20 | 0.2468 | 0.2144 | 0.207 |

- exp_028 minus exp_010: public -0.03 to -0.05; per record -0.041; pooled -0.007.
- stage1 minus stage2: public -0.02 to 0.00; per record -0.004; pooled +0.012.

`recall_probe.py` rebuilds exp_028 from its original inputs (dataset `sol-exp010-session1`,
kernel output `sol-exp026-verifier-train`) with `predict_verified.py`'s exact detector call and
rule. On CPU its test file matches all 1,624 submitted masks byte for byte; on validation it
reproduces exp_027's recorded TP/FP/FN at gates none, 0.5, 0.6, 0.7 and 0.8. `history.py` now also
rebuilds exp097 stage1 (exp098's `d4_pool_best.json`) and the mask2 probe (exp063's
`mask2/candidates_0.4.json`); every rebuilt row matches its recorded per-record PQ within 5e-5.

Across all eleven submissions, the least-squares fit public = 0.8395 x per record falls inside the
two-decimal rounding interval for ten and misses exp139 by 0.010 (RMS miss 0.0034). Per record
minus 0.070 misses two (mask2 by 0.018). Pooled one-to-one misses six, by up to 0.022 (exp_028),
and adding a constant does not fix it. The organisers say the test scoring differs from their
per-annotator notebook in "subtle" ways they will not disclose (F:753816, 2026-10-29); whatever
they are, they scale every submission tried by about the same factor and do not reward emitting
more. So:

- validation per record is the objective, as CLAUDE.md said; public is about 0.84 times it;
- the pooled-target retrain (`train.py --targets pooled`) trains the wrong question (CLAUDE.md
  section 4) and must not be run;
- the union files (C1-C3) and the re-tuned files (probe, A, B) do not gain per record
  (C3 -0.0009, the others -0.006 to -0.033). exp169 + guard stays the final file.

Everything below is kept as the record of what was measured under the refuted hypothesis.

---

Every model this project built scores 0.06-0.08 lower on the public leaderboard than on validation,
and other teams report the same gap ("local 0.46, Kaggle 0.40", forum 745880). Nothing tried
before explained it (`final_decision/decision_brief.pdf`, section 2e). Validation scores each
annotator's version of a photograph as a separate record. The submission format has one image per
photograph, so the hidden test truth must combine a photograph's annotators somehow, and the
organisers declined to say how (forum 729487, 753043).

All numbers below are on the canonical validation split (106 photographs, 180 annotation records:
59 photographs with one annotator, 20 with two, 27 with three). They use the validation
predictions of submissions whose public scores are known. CPU only.

## Which truth reproduces the leaderboard (`hypotheses.py`, `rasterisers.py`, `truth_filters.py`, `matching.py`)

| truth, scoring | exp139 | exp169 | exp097 |
|---|---|---|---|
| **public leaderboard** | **0.41** | **0.41** | **0.39** |
| per record (validation as scored) | 0.4705 | 0.4889 | 0.4668 |
| pooled, every pair above 0.5 counted (organisers' notebook) | 0.5103 | 0.5261 | 0.4980 |
| **pooled, one-to-one matching** | **0.4002** | **0.4138** | **0.3870** |
| union of annotators (by overlap / by pixel) | 0.4571 / 0.4575 | 0.4712 / 0.4718 | 0.4520 / 0.4525 |
| majority / intersection of pixels | 0.4909 / 0.4571 | 0.5090 / 0.4775 | 0.4786 / 0.4386 |
| first / last annotator only | 0.4602 / 0.4772 | 0.4817 / 0.4938 | 0.4572 / 0.4678 |
| mean PQ per photograph / per record | 0.4784 / 0.4826 | 0.4924 / 0.4981 | 0.4671 / 0.4777 |
| Left/Right categories only | 0.4142 | 0.4321 | 0.4213 |
| rasterised by OpenCV / PIL (10% / 8% fatter) | 0.4747 / 0.4773 | 0.4902 / 0.4935 | 0.4782 / 0.4793 |
| 1 px thinner / 1 px fatter | 0.3747 / 0.4736 | 0.3884 / 0.4885 | 0.3545 / 0.4852 |
| rasterised at 1024 / 512 | 0.4634 / 0.4289 | 0.4797 / 0.4439 | 0.4595 / 0.4301 |

Only pooled truth with one-to-one matching lands at the leaderboard's level with the leaderboard's
order. In it, every annotator's filaments are instances of one image, and a prediction can claim
one of them. A filament that three annotators drew is three truths, and one prediction can match
only one, so the other two are misses. Left/Right-only truth also lands near the level, but it puts
YOLO above Mask2Former, the reverse of the leaderboard.

## Seven submissions (`history.py`)

| submission | public | per record | pooled one-to-one |
|---|---|---|---|
| exp169 joint rule | 0.41 | 0.4889 | 0.4138 |
| exp139 Mask2Former e40 | 0.41 | 0.4705 | 0.4002 |
| exp097 YOLOv8l D4 | 0.39 | 0.4668 | 0.3870 |
| exp062 YOLO11m D4 c.35 e1 a300 | 0.39 | 0.4559 | 0.3920 |
| exp065 YOLO11m D4 c.35 e1 a100 | 0.38 | 0.4552 | 0.3958 |
| exp057 YOLO11m D4 c.385 e1 a100 | 0.38 | 0.4586 | 0.3927 |
| exp064 YOLO11m D4 c.30 e1 a300 | 0.38 | 0.4501 | 0.3939 |

Pooled one-to-one predicts the public level with no fitted constant (mean residual -0.005, spread
0.010). Per-record scoring needs an offset of -0.072 (spread 0.007). Within the YOLO11m group, all
four sit on the 0.385 rounding boundary, so their order says nothing. This is evidence, not proof:
the order within near-equal submissions does not separate the two explanations.

Things ruled out on the way: temporal leakage (2.8% of validation and 1.1% of test photographs
have a training photograph within an hour; 29% of each within a day), year and site mix (the test
set matches training on both), and how the truth polygons were rasterised.

## What it changes, measured (`tune.py`, `rerank.py`, `additions.py`)

If the truth is pooled, a filament drawn by only one of three annotators is a true positive, not a
false positive against the other two. The misses that pooling adds also lower the bar for any
candidate: one improves PQ when its expected matched IoU exceeds PQ / 2. Both favour more recall.

- **Threshold of exp169's rule** (pooled one-to-one / per record):

  | conf | 0.44 | 0.52 | 0.60 | 0.64 | **0.68 (exp169)** | 0.72 | 0.76 |
  |---|---|---|---|---|---|---|---|
  | pooled | 0.4158 | 0.4218 | 0.4205 | 0.4184 | **0.4138** | 0.3943 | 0.3729 |
  | per record | 0.4555 | 0.4691 | 0.4801 | 0.4827 | **0.4889** | 0.4800 | 0.4680 |

  Under pooled truth exp169 sits at the top edge of a plateau, just before a cliff. Lowering the
  threshold helps half A of exp168's folds by up to +0.013 and half B by at most +0.0045. A
  threshold chosen on one half and scored on the other gains nothing (-0.0001). Only 0.64
  gains on both halves alike (+0.0046 / +0.0045).
- **The whole rule** (YOLO weight 0-8, exponent 0.3-0.7, area 40/100, 31 thresholds): every
  setting reaches the same pooled plateau, 0.420-0.422 in-sample. Cross-validated over the full grid,
  the gain is +0.0009. About a dozen settings gain +0.004 to +0.007 on both halves. The steadiest one
  drops YOLO from the score: weight 0, exponent 0.5, threshold 0.78 (+0.0063 / +0.0067; per record
  0.4806).
- **Re-ranking** by "would any annotator draw this": a perfect ranking of exp169's own candidates
  would give pooled 0.4925. Learned rankers (logistic regression, gradient boosting; scores,
  per-layer probabilities, mask softness, YOLO support, shape, embedding PCA) score 0.4087 / 0.4062
  cross-validated, against 0.4138. Killed.
- **YOLO detections exp169 lacks**: few confident ones exist. Low-confidence YOLOv8l detections
  (0.25-0.50) match a still-unmatched pooled truth 38% of the time, about +0.004 pooled, and lose
  per record. Not used.

So the leaderboard gap is explained, but re-tuning exp169's own predictions gains only about
+0.005. Under pooled truth exp169 is that close to the best rule of its family.

## A second model's uncovered detections (`union.py`, `swinb_infer.py`, `union_test.py`)

In multi-annotator validation photographs exp139 has a candidate for 69% of the filaments that 1 of 3
annotators drew, but scores them low (median 0.54), so exp169 keeps 16% of them. It keeps about 90% of
those everyone drew. Pooled truth counts both alike. Swin-B (exp190b epoch 32) finds some that exp139
misses. Its full-view detections no exp169 instance covers match a still-unmatched pooled truth
40% of the time at score 0.80 and above, against a break-even of about 31%.

| rule | pooled (val) | gain, half A / half B | 95% CI of gain (photo bootstrap) | per record |
|---|---|---|---|---|
| exp169 + guard | 0.4139 | | | 0.4889 |
| C1: B + guard + Swin-B >= 0.90 | 0.4235 | +0.0073 / +0.0123 | +0.0017 to +0.0176 | 0.4825 |
| C2: exp169 + guard + Swin-B >= 0.80 | 0.4224 | +0.0079 / +0.0094 | +0.0008 to +0.0165 | 0.4754 |
| C3: exp169 + guard + Swin-B >= 0.90 | 0.4184 | +0.0020 / +0.0074 | +0.0006 to +0.0088 | 0.4881 |

What else was tried:
- Swin-B as a voter (`swinb_pooled.py`) gains about as much as dropping the YOLO veto, and nothing more.
- Swin-B as both voter and union source (`voter_union.py`, full view at class probability 0.5 as on
  test, Swin-B weight 8 or 16, exponent 0.3 or 0.5): the best worse-half gain is +0.0082 (weight 16,
  exponent 0.5, conf 0.52), against C1's +0.0073, with the threshold chosen on the same halves. Not
  distinguishable from C1, so no file was built.
- YOLO detections on top of C1 add at most +0.0009.
- Growing or shrinking masks by 1 px loses about 0.02 pooled.

Swin-B's test candidates were computed on CPU by `swinb_infer.py`, the same forward pass as its
validation caches (a validation photograph reproduces its 4 candidates at class probability 0.5 or
above exactly). About 75 s per photograph with two processes on four cores.

A model trained on pooled targets (`experiments/exp_139_mask2former/src/data.py: pooled_targets`,
`train.py --targets pooled`) is the next step. It needs a GPU; see
`final_decision/results/exp200/GPU_PLAN.md`.

## Earlier test files (`emit_test.py`; parity: exp169's setting rebuilds its submitted file byte for byte)

| file | what it tests | pooled (val) | per record (val) | expected public: pooled / per record |
|---|---|---|---|---|
| probe: exp169 at conf 0.44 | which truth the leaderboard uses | 0.4158 (+0.002) | 0.4555 (-0.033) | 0.41 / 0.38-0.39 |
| A: exp169 at conf 0.64 + guard | robust pooled gain, one constant moved | 0.4184 (+0.0046) | 0.4827 (-0.006) | 0.41-0.42 / 0.40-0.41 |
| B: no YOLO veto (w0 a0.5 conf 0.78) + guard | steadiest pooled setting | 0.4202 (+0.0064) | 0.4806 (-0.008) | 0.41-0.42 / 0.40-0.41 |

Review sheets (largest added instances, native crops): every large addition is a dark, elongated
filament on the disk; none is removed from exp169.
