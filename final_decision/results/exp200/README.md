# exp200: why the leaderboard sits 0.07 below validation, and three files to submit (2026-10-30)

**Finding.** The leaderboard behaves as if each test photograph's truth holds every annotator's
filaments at once, matched one-to-one. A filament that three people drew counts as three truths,
and one prediction can claim only one of them. Scored that way on validation, with no fitted
constant, seven past submissions land on their public scores:

| submission | public | validation as we scored it | pooled one-to-one |
|---|---|---|---|
| exp169 | 0.41 | 0.4889 | 0.4138 |
| exp139 | 0.41 | 0.4705 | 0.4002 |
| exp097 (YOLOv8l) | 0.39 | 0.4668 | 0.3870 |
| exp062 (YOLO11m) | 0.39 | 0.4559 | 0.3920 |
| exp065 / exp057 / exp064 | 0.38 | 0.455 / 0.459 / 0.450 | 0.396 / 0.393 / 0.394 |

Every other construction tried (per annotator, merged by union, majority or intersection,
one annotator only, other rasterisers, coarser masks, category subsets, per-photo averaging) misses
the level or the order. Leakage, year mix and site mix are ruled out. Details and code:
`experiments/exp_200_test_truth/README.md`.

**What it changes.** Under pooled truth a filament any annotator drew is a hit, so the best rule
keeps more candidates. exp169 sits at the edge of a cliff (pooled 0.4184 at threshold 0.64, 0.4138
at its 0.68, 0.3943 at 0.72). The gain that holds on both halves of the validation folds is
about +0.005, not more. A learned re-ranker and extra YOLO detections did not beat exp169 under
cross-validation.

## Update, same day: a second model adds about +0.009 (`union.py`, `swinb_infer.py`, `union_test.py`)

Under pooled truth a filament any annotator drew is a hit. exp139 under-scores filaments that only
some annotators drew: in multi-annotator photographs it keeps 16% of those drawn by 1 of 3, against
about 90% of those drawn by all. Swin-B (exp190b epoch 32, validation 0.4698 alone; checkpoint now
in `weights/`) finds some that exp139 misses. Its confident full-view detections that no exp169
instance covers match a still-unmatched pooled truth 40% of the time, above the 31% break-even.
Adding them (painted into unclaimed pixels; the same function on validation and test):

| file | rule | pooled gain (val, 95% CI) | half A / half B | per record |
|---|---|---|---|---|
| `submission_C1_B_swinb090.csv` | B + guard + Swin-B >= 0.90 | +0.0096 [+0.0017, +0.0176] | +0.0073 / +0.0123 | -0.0064 |
| `submission_C2_exp169_swinb080.csv` | exp169 + guard + Swin-B >= 0.80 | +0.0085 [+0.0008, +0.0165] | +0.0079 / +0.0094 | -0.0136 |
| `submission_C3_exp169_swinb090.csv` | exp169 + guard + Swin-B >= 0.90 | +0.0045 [+0.0006, +0.0088] | +0.0020 / +0.0074 | -0.0009 |

Paired photo bootstrap, 5,000 draws; P(gain > 0) is 0.98-0.99 for all three under pooled truth.
Swin-B was run on the test photographs on this CPU with the same forward pass as its
validation caches (a validation photograph reproduces its stored candidates exactly). C3 costs
almost nothing if the board turns out not to be pooled. C1 is the strongest if it is. B was chosen
from many settings, so C1's in-sample gain is somewhat optimistic.

## Submit these three (in this order; all non-overlapping, 180 photographs, SHA256 in `SHA256SUMS`)

| file | what it is | pooled (val) | per annotator (val) | expected public if pooled / if per annotator |
|---|---|---|---|---|
| `submission_probe_conf044.csv` | exp169 at threshold 0.44, 1,672 instances. **Decides which truth the board uses.** Not for final selection. | 0.4158 | 0.4555 | **0.41** / **0.38-0.39** |
| `submission_A_conf064_guarded.csv` | exp169 at threshold 0.64 plus the large-filament guard, 1,255 instances. Adds 114, removes none. | 0.4184 | 0.4827 | 0.41-0.42 / 0.40-0.41 |
| `submission_B_no_yolo_veto_guarded.csv` | Mask2Former views only in the score (YOLO weight 0, exponent 0.5, threshold 0.78) plus the guard, 1,335 instances. Adds 208, removes 14. | 0.4202 | 0.4806 | 0.41-0.42 / 0.40-0.41 |

Review sheets: `review_probe.jpg`, `review_A.jpg`, `review_B.jpg` (largest changes, native crops; red
= added, blue = removed). Every large addition is a dark, elongated filament on the disk. B's
largest removal (4,258 px) is a structure that runs over the limb.

## How to read the results

- **Probe shows 0.41 (or higher)**: the board is pooled. Pick A or B, whichever scores higher, plus
  exp169 as the two final selections.
- **Probe shows 0.39 or lower**: the board scores per annotator as validation does. Keep exp169
  (or `../submission_exp169_guarded.csv`) and drop A and B.
- **Probe shows 0.40**: inconclusive. Keep exp169 first; A second.

The public board prints two decimals and scores about half of the 180 photographs, so a change of
+0.005 shows only if it crosses a rounding boundary.
