# Handoff: the final experiment (prepared and run 2026-10-29)

**Status (2026-10-29, evening).** Section 1 was run and **killed**: CV gain -0.0039 against the
+0.002 bar, and every Swin-B weight lowers PQ (`results/RESULTS.md`). Section 2 was run: the guard
is validated (0.48893 vs 0.48895, so it passes non-inferiority) and reviewed, and its test file is
`results/submission_exp169_guarded.csv` (SHA256 `EC00408E...`). Two things remain. The credential
here cannot submit (`competitions.participate` denied), so the user uploads the file by hand. Then
comes section 3, reproducibility. Everything in it is done except the rehearsal on a GPU:
- the code is in `experiments/` (exp147-exp169 recovered unchanged; exp185, exp195 and exp199 added);
- the three checkpoints are in `weights/`;
- `reproduce/` holds `run.py`, the notebook and the stored validation predictions;
- `requirements.txt` pins torch, torchvision, transformers, kernels and ultralytics (the rest after
  the rehearsal's `pip freeze`).

On CPU the vote and guard rebuild the submitted file byte for byte. A one-photograph smoke test of
the two GPU stages matched the stored model outputs exactly. The full rehearsal needs one Kaggle
2 x T4 session, about 1.5 h (`reproduce/README.md`).

Read `decision_brief.pdf` first. The plan below was written before any of it ran; the status
above says what has. The user must approve before you start (prompt, Phase 4). Budget: RunPod
$12 (spend none of it on this plan), Kaggle free T4/CPU, 5 submissions per day, Kaggle deadline
2026-11-15 06:00 UTC, report and code 2026-11-15 23:59 CDT.

Follow `CLAUDE.md` (repo root) throughout. In particular: list the account's kernels by run date
before every push (section 14), keep kernel `title` and `id` slug-identical, read back every
resolved argument after launch (section 1), and make every parity check fail loudly (section 5).

## 0. What exists and where

| input | location (Kaggle, account `cosmicwitness`) |
|---|---|
| exp139 e40 validation candidates (full, D4, tiles, scale) | output `val_cache.json` of kernel `sol-exp147-candidate-cache-t4` |
| exp139 e40 test candidates (full, D4) | output `e040_test_cache.json` of `sol-exp151a-e040-test-cache-t4` |
| half-stride shift views, val and test | outputs of `sol-exp157-shift-cache-t4` |
| YOLOv8l D4 pools: val / test | `evaluation/d4_pool_stage2_best.json` of `sol-exp098-exp097-evaluation-t4`; `yolov8l_test_pool.json` of `sol-exp161-yolo-test-pool-t4` |
| YOLO11m pools: val / test views | dataset `sol-exp002-pools` (also on branch `exp/002-yolo-seg:kaggle/exp002_pools_dataset/`) |
| exp168 report (exp169's key and its in-sample row) | output `exp168_report.json` of `sol-exp168-joint-tuning-cpu` |
| exp139 deployed test CSV and exp169 test CSV | outputs of `sol-exp169-joint-submission-cpu` (SHA256 `CCA9C06E...` and `CC15EA41...`) |
| Swin-B e32 checkpoint (431,876,767 bytes) | output `longb/model_e032.pt` of `sol-exp190b-session-long-t4` |
| Swin-B e32 validation caches (full, D4), 5 shards, 106 photos | outputs `swinb_e032_s{0..4}_val_cache.json` of `sol-exp198a..e-swinb-cache-cpu` |
| the code of all of these | `final_decision/recovered_code/<kernel>/` (each file matches its kernel's SHA256 manifest) |

The exp169 rule, for reference: for each exp139 full-view candidate, partners are the best-IoU
(> 0.5) candidates of 7 dihedral and 3 half-stride views; the mask is the majority vote of the
candidate, its partners and its eroded YOLO partners; the score is s^0.3 x mean^0.7 with
mean = (11 view scores + 4 x two YOLO supports) / 19; keep score >= 0.68, area >= 100, paint by score
(`recovered_code/sol-exp169-joint-submission-cpu/experiments/exp_168_joint_tuning/src/joint.py`).

## 1. Primary: a second, equally strong Mask2Former as a voter (exp199)

**Question.** exp139 e40 (Swin-S, val 0.4705) is the only Mask2Former in exp169's vote. exp195a
added three weaker Mask2Former models (0.4518-0.4636) as voters, and PQ fell as their weight rose
(0.48895 / 0.48776 / 0.48707 / 0.48388 for weight 0 / 1 / 2 / 4). Swin-B e32 (exp190b, 0.4698, a
different backbone and seed) is the first voter as strong as e40. Does it help?

**Code (new folder `experiments/exp_199_swinb_votes/`).** Start from
`recovered_code/sol-exp195b-family-votes-cpu/experiments/exp_195_family_votes/src/votes.py`.
Change only:
1. a `--fixed-key d4+shift_w4_a0.3` option that keeps exp169's constants and varies only the new
   voter's weight m in {0, 1, 2, 4} (exp195 searched 72 configurations; its cross-validation
   penalty for that search was about -0.004, which no realistic gain survives);
2. a loader that merges the five Swin-B shards into one cache in exp147's layout and asserts 106
   unique validation photographs, each with `full` and `d4` lists;
3. keep exp195's parity assertion (weight-0 row of exp169's key equals 0.48894859995182627 with
   TP/FP/FN 892/304/433) and its `--with-masks` switch.

**Pilot (CPU, existing caches, about 1-2 h on a Kaggle CPU kernel or locally).** Run twice,
without and with `--with-masks`. Required log lines, in order: `PARITY ... passed: true`;
`SINGLE swinb` with PQ within 0.4698 +- 0.005 (CPU fp32 cache vs the T4 training-time evaluation);
one `CONFIG` row per weight; `DECISION`.

**Kill criterion (fixed before running).** On the 106-photo canonical validation split, with
exp168's two folds (seed 2026): adopt only if the cross-validated PQ of the restricted grid beats
the cross-validated PQ of exp169's key under the same treatment (0.48895) by at least +0.002 AND
the in-sample best row has m > 0. Otherwise stop and go to section 2. Why +0.002: it is the bar
exp169 itself was adopted on (`exp169_report.json`: gain 0.00226, bar 0.002), so the comparison
stays like for like; paired bootstrap noise between rules on these photos is about 0.003
(exp168 `vs_fixed` 95% interval -0.0021 to +0.0086), so a smaller bar would adopt noise.

**Full run (only if adopted; free Kaggle T4, about 1-1.5 h).**
1. Swin-B test cache, modes `full,d4`, with
   `recovered_code/sol-exp198a-swinb-cache-cpu/experiments/exp_147_candidate_cache/src/cache.py`
   on a GPU (the exp198 kernels ran it on CPU at about 14 min per photo). Read back the candidate
   counts per photo against the validation counts' scale.
2. Submission builder: exp169's `joint_submit.py` plus the new voter (support weight m, masks if
   chosen). Gates, all fatal: (a) validation replay equals the pilot's in-sample row for the
   chosen key; (b) with m = 0 the test output is byte-identical to exp169's `submission_joint.csv`
   (SHA256 `CC15EA41...`); (c) exp139 parity CSV SHA256 `CCA9C06E...`; (d) no pixel overlaps;
   (e) review sheet against exp169 (`exp_156_review`), looked at before submitting.
3. Apply section 2's guard to the new rule (its own non-inferiority check), then submit once.

## 2. Fallback (and final step in both branches): never drop a large Mask2Former filament

**Why.** exp169 drops 101 of exp139's 1,192 test instances, 9 of them >= 3,000 px; every large one
inspected is a clearly visible filament (`analysis/review/test_dropped_large.jpg`). On validation
the large drops were real filaments too, but drawn with a different extent, so PQ called them
false positives (`analysis/review/val_dropped_large.jpg`). They are dropped because the YOLO
term carries 8 of 19 votes and YOLO's confidence on them is low (57 of 69 validation drops have a
single YOLO match at IoU > 0.5; `analysis/outputs/veto_mechanism.json`). The judges inspect
predictions (rules_and_forum.md); a missing large filament is the visible failure.

**Rule.** After the final rule paints its instances, re-add every exp139 deployed instance
(single view, score >= 0.85, painted, min area 100) whose area is >= 3,000 px and which no final
instance overlaps at IoU >= 0.5; paint it into unclaimed pixels only; drop the remnant if < 100 px.
The cut is set by what the judges see, not tuned for PQ. One restriction was added on 2026-10-29
after the test review: skip instances that final instances already cover by 10% or more. The
unrestricted rule also re-added the remainders of 2 filaments exp169 had kept in part, which
split each into two pieces. With it the guard re-adds exactly the 9 deleted filaments.
Implemented in `experiments/exp_199_swinb_votes/src/guard.py`.

**Acceptance.** Validation in-sample PQ >= rule PQ - 0.001, no overlaps, and a review sheet of
the re-added test instances showing filament-like shapes (reject limb arcs, off-disk pixels,
sunspots). Measured: unrestricted 0.48835 (-0.0006); restricted 0.48893 (-0.00002, TP +1,
FP +2, FN -1); test 9 re-added, 1,150 rows, all filament-like (`results/`).

## 3. Final selection and reproducibility (needed whatever the pilot says)

- Rules allow up to two final selections (P:rules 2.2.b); the user believes one. Ask. If two:
  the final rule with the guard, plus exp169 as submitted (both reproducible from the repo).
- Weights must be in the repo and loaded by the notebook (P:Final Submission; F:724503). The
  exp139 e40 checkpoint is 275.8 MB and Swin-B e32 431.9 MB (about 4 bytes per parameter, so
  float32 weights), plus the two YOLO checkpoints. GitHub rejects files over 100 MB: use Git LFS
  or split files; store float16 only if a replay still reproduces the submitted CSV; test a
  clean clone.
- The exp147-exp169 code that produced exp169 is not in any branch: promote it from
  `recovered_code/sol-exp169-joint-submission-cpu/` into `experiments/` unchanged, then add the
  notebook and `requirements.txt` (torch 2.8.0, transformers 4.57.6, kernels, pycocotools, scipy,
  and ultralytics 8.4.153, the version the YOLO test pools were rebuilt with in
  `sol-exp161-yolo-test-pool-t4`).
- Rehearse the notebook end to end on a free T4 from the clean clone; it must reproduce the
  submitted CSV's SHA256.

## 4. Do not

- Retrain on all data, add pseudo-labels or neighbouring frames, or start RF-DETR: the evidence
  in the brief (in-sample ~ held-out; domain gap; cost) says none pays in the time left.
- Re-run exp195's 72-configuration grid, or tune the guard's cut on PQ.
- Use the public MAGFiLO release, neighbouring frames at inference, or file-name metadata.
- Push any kernel without first listing the account's kernels by run date (CLAUDE.md section 14).
