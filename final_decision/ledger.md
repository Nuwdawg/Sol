# Sol: ledger of what has been tried (updated 2026-10-29)

Companion to `decision_brief.pdf`. Phase 1 covered the 129 GitHub branches; this update adds the
52 experiments (exp146 to exp198, 2026-09-22 to 2026-10-06) that exist only as Kaggle kernels on
the account `cosmicwitness`. None of their code or results are in any branch of `Nuwdawg/Sol`.

**Source keys.** `AE n` = `docs/ACTIVE_EXPERIMENT.md` line n on `main` (commit `54dcd9f`, HEAD
`f2eff6a` only adds the prompt file). `LA n` = `docs/leaderboard-analysis.md` line n. `K:slug` =
Kaggle kernel `cosmicwitness/<slug>`, read-only (its log and output files). `subs` = the
competition's submission list for this account (34 submissions, read 2026-10-29). `A:file` =
analysis in `final_decision/analysis/` (this folder). **C** = confirmed by a file, log or
submission with numbers. **G** = guessed from a branch name or code only.

PQ (panoptic quality) = SQ x RQ. SQ is the mean IoU of matched pairs, RQ = TP / (TP + FP/2 + FN/2),
a pair matches at IoU > 0.5, pooled over every annotator-image record (checked against the
host's notebook on branch `exp/145-host-evaluation`; our evaluator agrees to 1e-10, AE 3536-3537).

## 1. Best runs

| | value | source |
|---|---|---|
| best model | exp139 epoch 40: Mask2Former, Swin-S (COCO instance weights), native 2048 px, single view, fp32 | AE 3399-3419 |
| its validation | PQ 0.47052, SQ 0.6855, RQ 0.6864, TP 881 / FP 361 / FN 444 (106 photos, 180 records) | AE 3401; A:outputs/diagnose_exp139.json (reproduced to 1e-12) |
| its public score | 0.41 (submission 56309562) | AE 3418; subs |
| best rule | exp169: exp139 e40 + 7 dihedral + 3 half-stride views + YOLOv8l and YOLO11m votes, score s^0.3 x mean^0.7, threshold 0.68 | K:sol-exp168-joint-tuning-cpu, K:sol-exp169-joint-submission-cpu |
| its validation | PQ 0.48895 in-sample, 0.48707 two-fold cross-validated; TP 892 / FP 304 / FN 433 | same; reproduced in A:veto_check.py |
| its public score | 0.41 (2026-09-28) | subs |
| public board resolution | two decimals; exp139, exp162, exp162b, exp169 all print 0.41 | subs |

## 2. Families (Phase 1, branches; abbreviated)

| family | experiments (branch) | best result | verdict | source |
|---|---|---|---|---|
| semantic nets | 001, 004 (U-Net/U-Net++) | val 0.2957; public 0.26-0.28 | dropped | C, RESULTS.md on `exp/001-baseline`; LA 226 |
| YOLO instance | 002, 005, 010, 031, 067-074, 097-100 | YOLOv8l D4 val 0.4668, public 0.39 | superseded | C, AE 1411, 1499; LA 113-573 |
| first Mask2Former | 035, 046-058 (Swin-T, 1024 px) | val 0.300 / 0.247 | failed, undertrained | C, AE 492, 673 |
| Mask2Former native | **139** (`main`) | val 0.4705, public 0.41 | best model | C, AE 3171-3418 |
| full-data retrain | 142 | crashed (ZeroDivisionError), fixed, never rerun | unrun | C, AE 3462, 3558 |
| pre-registered arms | 128 (preprocessing, one annotator per photo) | none | prepared, never run | C, `docs/PREREGISTRATION.md` |
| targets/labels | 006 (rasterisation), consensus labels, 025-027 verifier | target IoU 0.898 -> 0.959; two-annotator consensus predicts a third no better than one annotator (+0.0004) | YOLO-only fix | C, LA 471-497, 639, 837, 1819 |
| post-processing (YOLO era) | 005, 008-029, 049-050, 057-096, 103-138 | 1 px erosion +0.03 public (YOLO); everything else <= control or <= +0.0017 | exhausted | C, LA 573-1490, AE 493-3167 |
| ensembles (YOLO era) | 007, 021-023, 030, 033/037, 048, 059-063, 101, 102/119, 140, 141/143/144 | D4 fusion +0.0166 (YOLO); others <= +0.0013 | | C, AE 253-3523 |
| leakage | canonical split by photo, stratified by site | 43.4% of val photos have a train photo within +-1 calendar day | see section 4 | C, LA 532-560; `shared/data_split.py` docstring |
| label noise | 013-020, 088, 145 | annotator-vs-annotator PQ 0.340 (0.347 on val photos) | | C, AE 3548-3551; LA 741-797 |

Prepared but never run: `exp/039`, `exp/053`, `exp/054`, `exp/093` (C, AE 298, 466, 918).
Status inferred from name/code only (G): `exp/041`, `exp/042`, `exp/044`, `exp/045`, `exp/096`.

**Correction to the Phase 1 note.** Phase 1 said the Mask2Former mask threshold was never tuned.
It was: exp149 tested logit -1 and +1 and both lost (-0.0022, P(gain) 0.007; -0.0013), and 1 px
erosion or dilation cost about -0.03 (K:sol-exp149-evidence-fusion-cpu).

## 3. Kaggle-only experiments exp146-exp198 (all C, from the kernel named)

| exp | idea | result (validation PQ unless stated) | verdict |
|---|---|---|---|
| 146 | every model re-scored by one evaluator | exp139 0.4705 [95% CI 0.4493, 0.4907]; YOLOv8l D4 0.4668; YOLO pair 0.4681; exp139 at 1024 px 0.3419 | parity for all; native resolution matters |
| 147 / 151a / 157 | caches of exp139 e40 candidates (full, D4, tiles, scale, half-stride shifts) for val and test | parity 0.4705249990 | assets |
| 148 | GONG archive from Kaggle | unreachable | closed |
| 149 | rules on the cache: mask threshold, erosion, area, D4 mean/vote, tiles, scale, geometric score, learned combiners | geometric view score +0.0043 (P 0.88); area 200 +0.0005; mask threshold and erosion lose; logistic/boosting combiners lose | exhausted |
| 150 | 4-epoch fine-tune of e40 at lr 1e-5: train fold vs all data | held-out 0.4640 (e4) / 0.4651 (e2); all-data model on the same val photos (now in-sample) 0.4660 | no gain; in-sample ~ held-out |
| 153 | anatomy of the 444 misses | 255 have a full-view candidate with IoU >= 0.5 (median score 0.56); 152 at 0.3-0.5 (median score 0.88); oracle selection PQ 0.626 | diagnostic |
| 154 / 158 | D4 majority-vote masks + geometric score | 0.4788 (+0.0082, P 0.98) | helped |
| 159 | YOLO detections as votes | up to 0.4840 (+0.0135 over exp139, P 0.999) | helped |
| 160 | limb/off-disk pruning | no change | closed |
| 161 / 170 | YOLO test pools | parity | assets |
| 162 / 162b | final rule (+ YOLO mask votes) | 0.4840 / 0.4858; public 0.41 / 0.41 | submitted |
| 163 | protect confident exp139 instances from the vote | protect >= 0.95: +0.0002; >= 0.90: -0.0027 | neutral |
| 164 / 165 | semantic U-Net votes | +0.0007 (P 0.65) | not adopted |
| 166 | YOLO masks (eroded 1 px) in the pixel vote | +0.0018 (P 0.96) | adopted |
| 167 | early-epoch snapshot votes | about 0 | negative |
| 168 | joint tuning of the rule by two-fold photo cross-validation | CV 0.48707 vs 0.48481 for the fixed rule | adopted |
| 169 | joint rule submission | in-sample 0.48895, public 0.41 | **current best rule** |
| 171 | four more YOLO pools as votes | CV 0.4834 (< 0.4889) | negative |
| 172 | solar-rotation views as votes | CV 0.4878 | negative |
| 173a | T4 retrain: control vs IoU class target + EMA | 0.4636 vs 0.4578 | IoU target hurts |
| 173b | Swin-B 20 epochs; all-data Swin-S | Swin-B e20 0.4518; all-data model in-sample on 12 photos 0.4566 | no out-of-sample evaluation |
| 174 / 176 / 182 / 182b / 183a | unlabelled Helioviewer frames, teacher pseudo-labels, domain gap | exp139 on Helioviewer copies of competition scenes PQ 0.08-0.13 vs 0.46-0.48 on the competition JPEGs | domain gap; closed |
| 184a / 185 | three weaker Mask2Former models as votes | no weight > 0 chosen | negative |
| 187 | copy-paste augmentation; quality head | paste 0.4542; head 0.4626 (re-scored 0.4640) | negative |
| 190a / 190b | long schedules on T4: Swin-S 50 epochs (seed 7); Swin-B 40 epochs (seed 11) | Swin-S e50 0.4677; Swin-B e32 0.4698 | parity with exp139 |
| 194a / b / c | RF-DETR-Seg-Large | T4, 1152 px, 7 epochs (75 min per epoch): 0.3967; TPU: unsupported | undertrained |
| 195a / 195b | new-family votes with exp169 reproduced exactly as baseline | three Mask2Former voters: PQ falls as their weight rises (0.48895, 0.48776, 0.48707, 0.48388 for weight 0, 1, 2, 4); RF-DETR: CV -0.0034 | negative |
| 197 | focal + dice mask loss vs a seed-7 control | 0.4607 vs 0.4658 | negative |
| 198a-e | Swin-B e32 validation caches on CPU (full + D4, 106 photos in 5 shards) | never scored | unfinished asset |

Submissions: 34 in total, the last on 2026-09-28; at most 5 per day (subs; competition API).

## 4. New measurements made for this brief (CPU only, existing predictions; A:)

| question | answer | file |
|---|---|---|
| Is the test set less "leaky" in time than validation? | No. Nearest training photo within 1 day: val 29.2%, test 29.4%; within 7 days 94.3% vs 92.8%; median gap 1.66 vs 2.00 days | `temporal_train_test.py` |
| Does PQ depend on how many annotators a photo has? | 1 record: 0.469; 2: 0.500; 3: 0.454 | `outputs/diagnose_exp139.json` |
| Where are exp139's 444 misses? | 245 touch no prediction, 129 are near-misses (IoU 0.3-0.5), 70 below 0.3. On multi-annotated photos, 186 of 254 misses were drawn by that annotator only | same |
| Where are its 361 false positives? | On multi-annotated photos, 127 of 249 match another annotator's filament at IoU > 0.5 | same |
| Merges and splits? | rare: 37 of 1,325 truths overlap 2+ predictions; 40 of 1,242 predictions overlap 2+ truths | same |
| Rubric quantities (host notebook definitions) | mean IoU over overlapping pairs 0.6045; mean Dice 0.7337 | same |
| Photograph bootstrap of PQ | SD 0.0105 (106 photos), 0.0112 (90 photos) | same |
| How good is a committee of two annotators at predicting a third? | 0.295-0.329 PQ; exp139 on the same 81 held-out records: 0.454 | `outputs/committee_ceiling.json` |
| What does exp169 drop from exp139 on test? | 101 deleted + 13 reshaped of 1,192 instances; 9 deleted are >= 3,000 px, 7 had exp139 score >= 0.95 | `outputs/removed_instances.json` |
| Are those drops right on validation? | 69 dropped; 28% of their (instance, annotator) pairs were true matches; all 6 largest are real filaments, 2 were true matches | `outputs/veto_check.json`, `review/val_dropped_large.jpg` |
| Why are they dropped? | Not fragmentation: 57 of 69 (val) and 96 of 114 (test) have a single YOLO instance at IoU > 0.5; YOLO's low confidence pulls the score under 0.68. With no YOLO match an instance needs >= 0.997 in all 11 views to survive | `outputs/veto_mechanism.json` |
| Cost of keeping large exp139 instances (>= 3,000 px, score >= 0.85) that exp169 drops | val PQ 0.48835 vs 0.48895 (-0.0006; TP +1, FP +5, FN -1) | `outputs/veto_check.json` |

## 5. Reusable assets

| asset | where |
|---|---|
| exp139 e40 / e50 weights (275.8 MB each) | Kaggle dataset `cosmicwitness/sol-exp139-mask2former-weights` (AE 3388-3390) |
| Swin-B e32 (431.9 MB), Swin-S long e50 | output of K:sol-exp190b-session-long-t4 (`longb/model_e032.pt`, `long/model_e050.pt`) |
| other T4 checkpoints (control7 e4-e32, focal, paste, head, quality, all-data e12-e26) | outputs of K:sol-exp197..., K:sol-exp187..., K:sol-exp173a/b... |
| exp139 e40 candidate caches: val (full, D4, tiles, scale), test (full, D4), shift views | outputs of K:sol-exp147..., K:sol-exp151a..., K:sol-exp157... |
| Swin-B e32 validation caches (full, D4) | outputs of K:sol-exp198a...e |
| YOLO pools: YOLOv8l val/test, YOLO11m val/test views | K:sol-exp098-exp097-evaluation-t4, K:sol-exp161..., repo branch `exp/002-yolo-seg:kaggle/exp002_pools_dataset/` |
| the exp147-exp198 code, verified against each kernel's SHA256 manifest | `final_decision/recovered_code/` (52 kernels checked, 0 mismatches) |
| out-of-fold Mask2Former predictions | none (one split, one seed) |
