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
