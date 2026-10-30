# exp200: what the leaderboard scores against

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

So the leaderboard gap is explained, but the explanation does not unlock a large gain from the
existing predictions. Under pooled truth exp169 is within about 0.005 of the best rule of its
family. A larger gain would need a model whose scores answer the pooled question, for example
one trained on all annotators' filaments of a photograph at once.

## Test files (`emit_test.py`; parity: exp169's setting rebuilds its submitted file byte for byte)

| file | what it tests | pooled (val) | per record (val) | expected public: pooled / per record |
|---|---|---|---|---|
| probe: exp169 at conf 0.44 | which truth the leaderboard uses | 0.4158 (+0.002) | 0.4555 (-0.033) | 0.41 / 0.38-0.39 |
| A: exp169 at conf 0.64 + guard | robust pooled gain, one constant moved | 0.4184 (+0.0046) | 0.4827 (-0.006) | 0.41-0.42 / 0.40-0.41 |
| B: no YOLO veto (w0 a0.5 conf 0.78) + guard | steadiest pooled setting | 0.4202 (+0.0064) | 0.4806 (-0.008) | 0.41-0.42 / 0.40-0.41 |

Review sheets (largest added instances, native crops): every large addition is a dark, elongated
filament on the disk; none is removed from exp169.
