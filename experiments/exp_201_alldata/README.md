# exp201: all-data retrain, its submission builder, and a rule-transfer check

The retrain itself is two Kaggle T4 sessions (`kaggle/exp201/`). Pushing them to the account is the
user's step. This folder holds what runs on CPU around it.

## `src/build.py`: retrained models' test caches -> submission (no constant changed)

It runs the deployed remnant + rescue rule (`final_decision/results/remnant_rescue/README.md`), with
the retrained Swin-S in exp139's place and the retrained Swin-B in the old Swin-B's place.

**Rehearsal (2026-11-03).** Given exp139 e40's test cache, the old Swin-B test candidates and the
two YOLO test pools, it rebuilt all three deployed files byte for byte:

| file | SHA256 | rows |
|---|---|---|
| exp139 single view (the guard's base) | `CCA9C06E...` | 1,192 |
| exp169 + guard (submitted 2026-10-30) | `EC00408E...` | 1,150 |
| remnant + rescue (submitted 2026-11-03) | `F9423B74...` | 1,149 |

The full record is in `kaggle/exp201/rehearsal_build.json`. The command for session 2's outputs is in
`kaggle/exp201/README.md`.

## `src/transfer.py`: does the rule's gain hold on another Swin-S?

**Why.** exp201's Swin-S has no held-out photographs, so the rule's constants (threshold 0.68, YOLO
weight 4, exponent 0.3, rescue band, remnant cutoff) cannot be checked on it. exp190b's Swin-S
epoch 50 has the same recipe (seed 7, 50-epoch schedule) trained on the training split only. It is
the closest model on which those constants can be checked. The check also measures how much of the
rule's validation gain over exp139 (+0.020) is real and how much is tuning noise. The rule's
constants were all chosen on exp139.

**Cache.** exp190b Swin-S e50 on the 106 validation photographs, in exp139's 11 views (full, 7
dihedral, 3 half-stride shifts), on CPU with `exp_147_candidate_cache` (two processes, `--subset`).

**Parity on exp139 (smoke test).** Single view 0.47052 at 0.85, exp169's rule 0.48895 at 0.68, the
remnant + rescue rule 0.49095 at 0.68. All three match the recorded values exactly.

**Decision rule, written before the result (2026-11-03 21:00 UTC).**
1. Gate: exp190b Swin-S e50's single view at 0.85 must score 0.4677 +- 0.005 (its training-time
   evaluation on GPU). Otherwise the cache is wrong and nothing below is read.
2. Transfer: report the exp169-rule and combo-rule gains over the single view for both models,
   at the deployed 0.68 and at each model's own cross-validated threshold. If the second model's
   gain at 0.68 is under half of exp139's (+0.0204), most of the rule's validation gain is tuning
   noise. exp139 would then stay a final pick on equal terms, and exp201's file would use the
   single-view rule as its second file.
3. Two-model file: build a test file from the best pair variant (pair22, pair22_w8,
   pair_scores_w8, pair_masks) only if all of these hold against the combo rule on exp139
   (0.49095; halves 0.49618 / 0.48511):
   - its cross-validated PQ (threshold chosen on the other fold) is at least +0.003 higher;
   - its in-sample PQ is at least +0.003 higher;
   - neither half is lower.
   Otherwise no file is built, and the test caching of the second model is stopped.
4. The same bar applies to Swin-B e32 as the second model. It votes through its 8 cached views
   (full + 7 dihedral; exp198 cached no shift views). Variants: pair19, pair19_wkeep (YOLO weight
   6.91 keeps the two YOLO votes at 8 of 19 slots), pair_scores_wkeep, pair_masks. It runs on the
   existing validation caches. Its test file would first need Swin-B's dihedral views cached on test
   (about 15 h of CPU).
5. exp201's file (written 2026-11-04 07:10 UTC, before the exp190b Swin-S result): build it with
   exp169's constants unchanged (threshold 0.68). Move the threshold to t' only if exp190b Swin-S
   e50 (the same recipe without the validation photographs) shows all of these on the combo rule:
   - both of exp168's folds choose the same t' != 0.68;
   - t' beats 0.68 in-sample by at least +0.003;
   - its cross-validated PQ is at least that at 0.68.
   Otherwise keep 0.68. Either way, the second final file stays exp139 unless exp201 is reviewed
   and clearly sound.

**Result, Swin-B e32 as the second model (2026-11-03, `swinb_pair_val.json`).**

Rule transfer: every constant was tuned on exp139, yet Swin-B keeps most of the gain.

| model | single view (0.85) | exp169 rule (0.68) | combo rule (0.68) |
|---|---|---|---|
| exp139 e40 (11 views) | 0.47052 | 0.48895 (+0.0184) | 0.49095 (+0.0204) |
| Swin-B e32 (8 views: full + dihedral) | 0.46796 | 0.48290 (+0.0149) | 0.48468 (+0.0167) |

exp169's rule gives Swin-B 81% of the gain it gives exp139 (+0.0149 against +0.0184), with three
fewer views and constants chosen on another model. The combo row shows 82%, but there Swin-B's
rescue checks Swin-B against itself, so the exp169 row is the clean comparison. So most of the rule's validation gain is real, not threshold noise fitted
to exp139 (decision rule 2: well above half). That favours the remnant + rescue file over exp139
on the private board, and supports using the unchanged rule on exp201's models.

Two-model rule (exp139's candidates, Swin-B's views as extra voters), combo rule. The bar was 0.49095
+ 0.003 in-sample and cross-validated, with no half lower.

| variant | in-sample | cross-validated |
|---|---|---|
| pair19 (YOLO weight 4) | 0.48526 | 0.48280 |
| pair19_wkeep (YOLO weight 6.91) | 0.48764 | 0.48332 |
| pair_scores_wkeep (Swin-B scores only) | 0.48792 | 0.48323 |
| pair_masks (Swin-B masks only) | 0.49067 | 0.49067 |

No variant passes. Swin-B's view scores lower PQ even when YOLO keeps its share of the score, and
its masks in the pixel vote are neutral (-0.0003). This is exp199's finding, now with all eight
views: on validation, a second Mask2Former adds mostly errors that are correlated with exp139's
own. No Swin-B dihedral caching on test was started.

**Result, exp190b Swin-S e50 (2026-11-04, `s7e050_transfer_val.json`).** The cache ran on CPU, one
photograph per call, and finished at 12:15 UTC. The first run had been lost in a container reclaim
(CLAUDE.md section 18).

1. Gate: single view 0.46770 at 0.85, exactly its training-time evaluation (passes).
2. Transfer:

   | model | single view (0.85) | exp169 rule (0.68) | combo rule (0.68) |
   |---|---|---|---|
   | exp139 e40 | 0.47052 | 0.48895 (+0.0184) | 0.49095 (+0.0204) |
   | exp190b Swin-S e50 | 0.46770 | 0.48087 (+0.0132) | 0.48286 (+0.0152) |
   | Swin-B e32 (8 views) | 0.46796 | 0.48290 (+0.0149) | (self-rescue, not comparable) |

   The second Swin-S keeps 72-75% of the rule's gain, and Swin-B keeps 81%. Both are well above
   half, so the gain is mostly real, not tuned to exp139.
3. Two-model file (combo rule; bar 0.49395 in-sample and cross-validated): pair22 0.48692 / 0.48639;
   pair22_w8 0.48861 / 0.48700; pair_scores_w8 0.48950 / 0.48793; pair_masks 0.49005 / 0.49005. None
   passes, so no test caching was run. A second Mask2Former of the same family adds correlated
   errors rather than independent evidence (Swin-B gave the same result).
4. See the Swin-B result above.
5. Threshold for exp201's file: on the combo rule the two folds choose 0.67 and 0.76 (not the same),
   and the best in-sample threshold (0.69) beats 0.68 by only +0.0016. **exp201's file keeps 0.68.**
