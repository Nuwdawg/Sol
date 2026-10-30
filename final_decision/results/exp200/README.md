# exp200: why the leaderboard sits 0.07 below validation, and three files to submit (2026-10-30)

> **Correction, 2026-10-30 evening: the finding below is refuted, and its files are withdrawn.**
> Two pairs of past submissions differ far more in what they emit than the seven in the table, and
> pooled truth gets both wrong (`recall_probe.py`, `history.py`; `recall_probe.json`,
> `history_extended.json`):
>
> | pair | public change | per record (val) | pooled one-to-one (val) |
> |---|---|---|---|
> | exp_010 (0.36) -> exp_028 recall probe (0.32): verifier-promoted extra candidates | -0.03 to -0.05 | -0.041 | -0.007 |
> | exp097 stage2 (0.39) -> stage1 at conf 0.25 (0.38) | -0.02 to 0.00 | -0.004 | +0.012 |
>
> exp_028 was rebuilt from its own weights: the CPU rebuild of its test file matches all 1,624
> submitted masks byte for byte, and the validation rebuild reproduces exp_027's recorded TP/FP/FN at
> every gate. Over eleven submissions with public scores (0.20 to 0.41), **public = 0.84 x the
> per-record validation PQ** lands inside the rounding interval for ten and misses exp139 by 0.010;
> pooled one-to-one misses six, by up to 0.022. The leaderboard scores each annotator separately,
> as validation does, and about 16% lower. This is what CLAUDE.md already said and what exp_028 was
> submitted in September to show; this folder overrode it without checking exp_028.
>
> What follows from it:
> - **Final file: `../submission_exp169_guarded.csv`** (per record 0.48893, expected public 0.41).
> - Do not upload the probe, A, B, C1 or C2: per record they lose 0.006 to 0.033, about 0.005 to
>   0.028 in public. C3 is neutral (-0.0009 per record) and gains nothing either.
> - Do not run `GPU_PLAN.md` section 1: pooled targets train the "would anyone draw this" question
>   that sank the exp_026 verifier (CLAUDE.md section 4).
> - Public 0.42 needs per-record validation of about 0.494 or more; exp169 is 0.4889.
>
> The text below is kept as the record of what was measured.

**Finding (refuted, see above).** The leaderboard behaves as if each test photograph's truth holds every annotator's
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

## What was proposed (withdrawn; the files and review sheets are in `withdrawn/`, SHA256 in `withdrawn/SHA256SUMS`)

Upload these three first:

1. `submission_probe_conf044.csv`: exp169 at threshold 0.44 (1,672 instances). It decides which truth
   the board uses: about 0.41 if pooled, 0.38-0.39 if per annotator. Not a final pick.
2. `submission_C1_B_swinb090.csv`: the strongest file if the board is pooled.
3. `submission_C3_exp169_swinb090.csv`: the safe file, neutral if the board is not pooled.

Optional: `submission_C2_exp169_swinb080.csv`; `submission_A_conf064_guarded.csv` and
`submission_B_no_yolo_veto_guarded.csv` (both superseded by C1 and C3).

| file | instances | pooled (val) | per annotator (val) |
|---|---|---|---|
| exp169 (as submitted, guarded) | 1,150 | 0.4139 | 0.4889 |
| probe | 1,672 | 0.4158 | 0.4555 |
| A | 1,255 | 0.4184 | 0.4827 |
| B | 1,335 | 0.4202 | 0.4806 |
| C1 (B + 18 Swin-B filaments) | 1,353 | 0.4235 | 0.4825 |
| C2 (exp169 + 235 Swin-B filaments) | 1,385 | 0.4224 | 0.4754 |
| C3 (exp169 + 68 Swin-B filaments) | 1,218 | 0.4184 | 0.4881 |

Review sheets (largest changes at native resolution; red = added, blue = removed):
`review_probe.jpg`, `review_A.jpg`, `review_B.jpg`, `review_C1.jpg`, `review_C3.jpg`. The
Swin-B additions in C1 and C3 are dark, elongated filaments on the disk, some beside active regions.
None is removed from its base. One exception: C1's largest addition but one (3,817 px,
20150619133554Bh) runs over the limb. It is the structure B dropped from exp169.

## How the final selections were to be chosen (withdrawn)

- **Probe shows 0.41 or more**: the board is pooled. Select C1 and C3 (or C1 and whichever of C2/C3
  scored higher).
- **Probe shows 0.39 or lower**: the board scores per annotator. Select exp169
  (`../submission_exp169_guarded.csv`) and C3.
- **Probe shows 0.40**: inconclusive. Select C3 and exp169.

The public board prints two decimals and scores about half of the 180 photographs; a gain under
0.01 shows only if it crosses a rounding boundary. The final ranking uses the other half.
