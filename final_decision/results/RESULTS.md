# Results of the final experiment (run 2026-10-29, CPU only)

## Primary: Swin-B e32 as a second voter in exp169's rule. Killed.

Code: `experiments/exp_199_swinb_votes/src/votes.py`. Inputs: the cached validation predictions
listed in `../analysis/README.md` plus the five exp198 Swin-B shards (106 photos, checkpoint SHA256
`ac958377...`). Log: `exp199_pilot_log.txt`; numbers: `exp199_report.json`.

| gate | required | result |
|---|---|---|
| 1. parity: weight 0 reproduces exp169 | 0.48894859995, TP/FP/FN 892/304/433 | **passed** (exact) |
| 2. Swin-B alone | 0.4698 +- 0.005 | **passed**: 0.46975 (828/267/497 at 0.90) |
| 3. decision | CV gain >= +0.002 and best row has weight > 0 | **failed**: CV gain -0.0039; best row is weight 0 |

| Swin-B weight | 0 | 1 | 2 | 4 |
|---|---|---|---|---|
| validation PQ, masks not voting | **0.48895** (exp169) | 0.48812 | 0.48804 | 0.48682 |
| validation PQ, masks voting | 0.48874 | 0.48791 | 0.48784 | 0.48665 |

Cross-validated: exp169 0.48895, restricted grid 0.48504. As with the three weaker voters of
exp195a, every unit of weight lowers PQ. A second Mask2Former, even one as strong as exp139, adds
mostly its own errors to this rule. No GPU step was needed, so none was run.

## Fallback, now the final candidate: exp169 with the large-filament guard

Code: `experiments/exp_199_swinb_votes/src/guard.py` (one function for validation and test).

- **Rule.** Re-add every exp139 instance of at least 3,000 px that the vote deleted. One
  restriction was added after looking at the test masks: skip any instance that final instances
  already cover by 10% or more. The first version also re-added 2 remainders of filaments that
  exp169 had kept in part, which split each into two adjacent pieces (`review_guard_test.jpg`
  shows only the final 9). No labels were involved in that change.
- **Validation (non-inferiority bar: no worse than -0.001).** exp169 is reproduced exactly at
  0.48895. Guarded: 0.48893 (TP 893, FP 306, FN 432; 3 instances re-added; delta -0.00002).
  The first version scored 0.48835. `guard_val_report.json`
- **Test.** 9 filaments re-added (3,415-10,657 px each) to exp169's 1,141 instances. That makes
  1,150 rows over 178 photographs, with no overlaps, and exp169's own rows are byte-identical.
  `guard_test_report.json`
- **Review.** All 9 are clear, dark, elongated filaments on the disk, not limb arcs or sunspots
  (`review_guard_test.jpg`).

**File to submit: `submission_exp169_guarded.csv`**, SHA256
`EC00408E7F8ED78A350EEC50FCC1F96824A9851D972310F9408B4B64D7D966A7`. Its expected public score is
0.41, the same as exp169: the change is 9 instances out of 1,150.

## Not done, and why

- **Submission.** The Kaggle credential in this environment can read but not submit: the API
  answered `Permission 'competitions.participate' was denied`. Upload the CSV above by hand
  (Submit Prediction on the competition page). Then pick the final selection(s) on the
  Submissions tab; the rules allow two.
- **Reproducibility rehearsal.** The rules require a notebook that rebuilds this file from
  weights stored in the repository. Running it on the 180 test photos needs a GPU (one T4
  session). See `../HANDOFF.md`, section 3.
