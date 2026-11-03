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

**Result.** Pending (the cache takes about 9 h on this CPU).
