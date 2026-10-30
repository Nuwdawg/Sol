# A Mask2Former-only two-model file: the one bet on public 0.42 (2026-10-30)

**File:** `submission_m2f_only_w8.csv` (1,167 instances, 180 photographs, SHA256 `0EF5A352...`).
exp139's 11 views plus Swin-B e32 as a voter, no YOLO:
score = s^0.3 x mean^0.7, with mean = (11 exp139 view scores + 8 x Swin-B's partner score) / 19.
Keep instances scoring >= 0.84, paint them by score, and drop any smaller than 100 px.
Code: `experiments/exp_200_test_truth/src/m2f_only.py` (validation) and `m2f_only_test.py` (test).

## Validation says it is slightly worse than exp169

| rule | per record, in-sample | per record, cross-validated |
|---|---|---|
| exp169 | 0.48895 | 0.48707 |
| **this file** | **0.48297** | **0.4799** |
| exp139's views alone, no YOLO | 0.4791 | 0.4709 |
| exp139 single view (public 0.41) | 0.47052 | |

Cross-validation uses exp168's two folds (seed 2026). The grid is 5 Swin-B weights x 3 exponents x
22 thresholds. The chosen point sits in a flat region: weights 8-16 and thresholds 0.82-0.84 all
score 0.4810-0.4834.

## Why it could still score higher in public

The public / validation ratio differs by family, across the 17 submissions with both numbers. The
ranges below come from the two-decimal rounding:

| family | submissions | public / validation |
|---|---|---|
| Mask2Former without YOLO | exp139 e40 (0.41), e50 (0.40), e40/e50 soup (0.40) | 0.861-0.864, the only range all three allow |
| Mask2Former rules with YOLO votes | exp162, exp162b, exp169 (all 0.41) | 0.837-0.849 |
| YOLO only | nine submissions, 0.32-0.39 | about 0.81-0.85 |

The first two ranges do not overlap: no single factor fits both Mask2Former groups. This file has no
YOLO in it.
- If it transfers like the first group: 0.4830 x 0.861-0.864 = 0.416-0.417, which prints **0.42**.
- If it transfers like exp169: 0.404-0.410, which prints 0.40 or 0.41.

The pattern rests on seven rounded numbers. Read it as roughly even odds of 0.42, with some chance
of 0.40. One upload settles it.

## Checks

1. The test path reproduces the validation number exactly (0.48296881826980964).
2. The same test caches rebuild exp169's submitted file byte for byte (SHA256 `CC15EA41...`).
3. `reproduce/run.py --stages ... m2f_only` rebuilds this file and checks its SHA256.
4. Review against exp169 at native resolution (`review_vs_exp169.jpg`, the 16 largest differences):
   - 1,084 instances unchanged (median IoU 0.988).
   - 83 added (median 718 px): mostly clear dark filaments that exp169 dropped for lack of YOLO
     support, plus a few faint patches.
   - 57 removed (median 1,218 px): some are real filaments that exp169 kept on YOLO support alone.
   - Neither file is clearly better on these crops.

## How to use it

Upload it once.
- **It shows 0.42:** select it and `../submission_exp169_guarded.csv` as the two finals. They are two
  different strategies, and both rebuild from the repository.
- **It shows 0.41 or 0.40:** keep exp169 + guard and exp139.

## Also tried today, per record (no file)

- Swin-B only as a rescue or veto of exp169's borderline instances (`swinb_gate_val.json`):
  best +0.0010 in-sample; cross-validated -0.002.
- A plain average of exp139's and Swin-B's full views (`pair_ensemble_val.json`): 0.4773 in-sample,
  0.4741 cross-validated.
- Swin-B's support divided by eight views as in exp185 (`m2f_only_val_diluted_voter.json`): the
  test caches hold its full view only, so this dilutes it. Cross-validated 0.471; corrected above.
- Swin-B voting through its full and seven dihedral views (validation caches only;
  `m2f_only_val_swinb_d4_voter.json`): best 0.4825 in-sample, 0.4759 cross-validated. No better
  than its full view alone, so the dihedral views were not computed on test.
- Splitting predictions that cover two filaments: the splits by the other views flag 6 of 11 such
  predictions, and 11 correct ones too.
