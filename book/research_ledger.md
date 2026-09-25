# Research ledger for the onboarding book

Working notes, transcribed from the project's own records. Each line names its source.
Not part of the book itself.

## From docs/strategy.md
- State 2026-09-05ish: public 0.36, val 0.4404 (exp_005), rank ~92/502, goal >0.46; top cluster 0.55-0.56.
- Ten early experiments each targeted one failure mode; combined contribution ~0: threshold tuning,
  spine seeding, disk masking, sub-pixel trimming, calibrated emission, dihedral TTA, two refiners.
- Error decomposition (exp_013/014/016): ~50% instances one side doesn't acknowledge (annotators disagree at
  same rate); mask on right object but too wrong to match: 195 truths / 150 preds; covered by a
  candidate that confidence discards: 305 truths; invisible to detector: 2.1%; split/merge: 7 of 936.
- exp_002 detector never trained properly: trained 1280, inferred 2048; targets rasterised 11% fatter
  than scorer convention (cv2.fillPoly vs pycocotools; 0.5 px inward buffer: IoU 0.898->0.959,
  area ratio 1.111->0.986); stopped by time=8.5h budget at 149 epochs.
- PQ = SQ x RQ; post-hoc geometry can only move an existing mask; cannot create instances.
- Annotator vs annotator (1024 raster): PQ 0.3361 SQ 0.6348 RQ 0.5296; detector exp_005: 0.4404/0.6843/0.6436.
- Two refiners: synthetic one IoU 0.7208->0.8529 on its task but lost PQ; real one 0.6105->0.6335,
  plateau at epoch 2 for 137 epochs.
- Latent-boundary argument: pairwise disagreement 0.365 of union -> SQ ceiling ~0.82 vs single annotator.
  (Later marked as hypothesis, not fact, per host guidance.)
- exp_010 plan: train at 2048, corrected targets (polygon offset), mask_ratio=1 (loss at 2048 not 512).
  mask_ratio=4 computes mask loss on 512 grid, where 0.5px correction is below representable.
- Kaggle 12h cap chaining: kernel lists itself in kernel_sources; latest successful version mounted.
- exp_015 submit on CPU: CPU inference bit-identical to GPU (same CSV MD5).
- Val->public offset: 0.4064->0.33 and 0.4404->0.36 (gap 0.078 both). Covariate shift excluded
  (year, station); weighting per photograph moves 0.0008.
- Not done: TTA lost 0.037 (early); consensus targets (+0.0004 averaged two annotators predict third);
  dense semantic 0.26/0.28 vs instance 0.32.
- Bugs caught: optimizer='auto' ignores lr0 picks MuSGD 0.01; bare resume=True searches Ultralytics
  settings dir; OOM ladder conceded mask resolution silently.
- Falsification: if exp_010 converges and val doesn't move past 0.4404 -> 32-prototype basis binding.
- Connected components on GT masks score PQ 0.9995 -> instance separation not the obstacle.
- Verifier arithmetic: promoting k right, m wrong: PQ = (578.2 + 0.65k)/(1313 + 0.5k + 0.5m);
  break-even 2 wrong per 1 right -> precision > 33% vs base rate 10.5%.
- exp_005 GBM over 4 geometric features: adds nothing beyond confidence.
- exp_019 mask-field falloff profile (14 cuts): AP 0.7248 vs confidence 0.7150, still cannot promote:
  at most conservative threshold adds 23 FP, 0 TP.
- Correction 2026-09-05: FPs at human disagreement rate (19.8% vs 19.4%, exp_014) but misses are NOT:
  at conf floor 0.05 only 2.1% of GT invisible, 83.2% covered by some candidate at IoU 0.5
  -> misses are a ranking problem.
- Perfect re-ranker: SQ 0.68 x 1102/(1102+0.5x223) = 0.6175 local (~0.55 public).
- Verifier precision table: 100%->0.5298 local; 80%->0.5164; 60%->0.4954; 50%->0.4799.
- Crop classifier idea (exp_011 harvest, exp_012 TPU train).

## From docs/leaderboard-analysis.md, lines 1-1260
- LB shape 2026-08-31: 465 teams, dense mass 0.30-0.40, gap 0.40-0.55, 12 teams at 0.55 (artefact).
- Evaluator exploit (artkomissar/please-fix-the-leaderboard): 5 of 180 images with empty masks scored 1.00
  on old Dice metric; moved to PQ; later Aug-20 host announcement about hacking.
- MAGFiLO v1.0 public release overlaps test -> do not train on it (leak). External imagery for SSL allowed.
- hdjojo public 0.55 notebook: YOLOv8l-seg nc=1 imgsz 2048 conf 0.3 iou 0.0; defect: sorted overlap by
  boxes.cls (all zeros) instead of boxes.conf.
- exp_001 U-Net++ (effnet-b4 @512) semantic + connected components: val 0.2957 -> public 0.26 (rank ~314/466).
- exp_002 yolo11m-seg @1280, 149 epochs (time-capped 8.5h, not converged), mask mAP50 0.709 / 50-95 0.284,
  public 0.32. Threshold sweep 7 conf x 3 min_area: best conf 0.30 area 300 PQ 0.3736 vs default 0.3714
  (+0.0022, flat because FP and FN each cost half a unit).
- exp_001 vs exp_002: SQ 0.629 vs 0.643; RQ 0.470 vs 0.581 (+24% recognition).
- At conf 0.20: 545 FN of 1325 (~40% not detected at any threshold) at 1280.
- Spine seeding ablation (exp_004 spine_ablation.py): erosion 0/3/5/7 -> CC vs spine PQ equal (delta ~0);
  erosion destroys small filaments instead of splitting large ones. exp_001 emitted 8.6 inst/img vs GT 7.4.
- exp_004 U-Net resnet34 @1024 on TPU v5e-8 SPMD, 300 epochs converged (val loss flat 0.1813-0.1815),
  3.8h, 9.50 inst/img, public 0.28. Dense line capped ~0.28.
- exp_005 center/offset grouping ablation (Panoptic-DeepLab style): noise 0/5/15/30 px -> PQ 1.0/0.9938/
  0.9694/0.8336 vs CC 0.9995. Ablations on GT masks can't show decoder gains (CC already 0.9995).
- Ultralytics augment=True on -seg: warning, silently ignored (single-scale).
- Inference resolution sweep exp_002 (trained 1280): 1280 0.3736; 1600 0.3965; 2048 0.4064 (SQ 0.670 RQ 0.607,
  TP 854 FP 635 FN 471) -> public 0.33. 2560 0.3993, 3072 0.3624 (RQ falls; SQ flat) -> 2048 operating point.
- Val->public gap widening: 0.054 then 0.076.
- Inter-annotator PQ 0.3371 (1024 raster, 150px min area, 598 pairs): SQ 0.6341 RQ 0.5317 TP 2122 FP 1873 FN 1865.
  411 photos once, 145 twice, 151 thrice.
- Metric confirmed via host notebook: PQ at IoU 0.5; organiser marks every pair >0.5 as TP; ours greedy
  one-to-one; equivalent when masks disjoint (counting argument: two disjoint preds cannot each cover >half
  of one GT). 25 parametrised cases pinned (tests/test_pq_matches_official.py). 0 of 107 sampled images
  have overlapping GT.
- Consensus fusion (shared/consensus.py: link IoU 0.25, pixel majority vote, painted disjoint). Circular test
  0.6558 vs 0.3361; leave-one-out on 151 three-annotator photos: held-out vs single other 0.3287, vs consensus
  of other two 0.3291 (+0.0004). -> disagreement is structured ambiguity, not noise around a truth.
- Temporal leak: grouped split, val obs with a train obs within +-0d 10.4%, +-1d 43.4%, +-2d 64.2%,
  +-3d 79.2%, +-7d 94.3%. make_temporal_split: 5 contiguous blocks 2011-2022, 7-day buffer: 578/940 train,
  105/173 val, leak 0%. Comparisons within grouped split still valid as rankings.
- exp_005 erosion: 0.33 -> 0.36 (rank 206 -> 85/486). Hypothesis was growing; measured: erosion -1 wins.
  grow -4 0.1156; -2 0.3702; -1 0.4404 (SQ 0.6843 TP 845 FP 456); 0 0.4169; +3 0.2016. Conf peak 0.35.
- Transfer rates: resolution +0.033 val -> +0.010 public (30%); erosion +0.034 -> +0.030 (88%); refiner
  -0.008 -> -0.010 (122%). Sign always transferred.
- Rasterisation (exp_006 diagnostics): pycocotools vs cv2 IoU 0.8744; vs PIL 0.9146; cv2 vs PIL 0.8970.
  Areas: pycocotools 2250.6, PIL +5.71%, cv2 +7.96%. Ultralytics polygon2mask uses cv2.fillPoly; scorer
  pycocotools: IoU 0.9016, area ratio 1.1077. Perimeter/area of filaments 0.216.
  Eroding target 1px: IoU 0.8945 area 0.9004 (overshoots). Offset correlates with perimeter/area r=+0.926:
  Q1 1.155, Q2 1.121, Q3 1.101, Q4 1.074. Distance-transform sub-pixel trim is quantised (no-op).
- Polygon inward offset (shapely buffer): 0.25px IoU 0.9455 area 1.0388; 0.5px IoU 0.9592 area 0.9864;
  1.0px 0.8757. Naive buffer split 42 of 8199 instances -> keep largest piece; erased thin -> keep original.
- mask_ratio check (Ultralytics 8.4.137 v8SegmentationLoss): live code upsamples prototypes to target size.
- Calibrated emission GBM (area, elongation, limb distance, solidity), GroupKFold by photo; 3686 candidates
  37.7% match; prob gate 0.45 PQ 0.4266 vs confidence 0.35 PQ 0.4432 -> raw confidence wins by 0.017.
  Emission rule: predict when P(match)*E[IoU] > 0.5*PQ (~P > 0.32).
- Erosion gain decomposition: with 300px filter +0.0235, without +0.0223 -> 95% boundary. SQ 0.6707->0.6837
  filter off; TP 833->861, FP 533->505, FN 492->464.
- Rim analysis (124 matched pairs): disagreement within 0-1 px 41.4%, 1-2 px cum 64.2%, 2-3 px 77.4%,
  3-5 86.0%, 5-8 90.5%, 8+ 100% (8+ band FP 5062 vs FN 1384: 3.7:1 over-prediction far from boundary).
  Eliminate rim <=2px: SQ 0.8554, PQ 0.4347->0.5474; <=3px 0.5783; interior only >3px 0.4687.
- exp_008/009 refiner: 256px native crops U-Net, synthetic pairs (dilate/translate/smooth GT); own-task
  IoU 0.7208 -> 0.8529 (138 min on 8 TPU cores); PQ 0.4322 @0.6 vs erosion 0.4404; stacking refiner+erosion
  0.3711. Lesson: trained on a model of the error (severity-calibrated, not character-calibrated).
  Refiner public 0.35 (regression datum).
- Dihedral TTA (first impl, voting on binarised masks, greedy IoU link): 0.4035 vs 0.4404 single view.
- exp_013 error decomposition (exp_005 cache, reproduces 0.4403668): FP/FN by best overlap: near 0.25-0.50:
  150 FP / 153 FN; graze 0.10-0.25: 48/28; sliver 0-0.10: 34/14; orphan none: 224 (49%) / 285 (59%).
  Splits 2, merges 5 of 936 failures.
  Oracles: remove orphan+sliver FP 0.4883; near->match 0.4975; all FP removed 0.5329; all FN 0.5389;
  perfect masks on matches 0.6436 (+0.203).
- exp_014: orphan rate (IoU<0.10) of annotator instances by detector 22.6% (299/1325) ...
  annotator instances by another annotator 19.4% (255/1315); detector preds orphaned by annotator 19.8%
  (258/1301). 87 preds orphaned by reference annotator were drawn by a different annotator; 75 missed truths
  drawn by no other annotator. ~42 of 299 orphaned truths attributable to model.

## From docs/leaderboard-analysis.md, lines 1260-2072
- exp_016 recall ceiling (pool to conf 0.05, 23.4 cand/photo): covered at IoU 0.5 by some candidate 1102
  (83.2%); of those already above 0.35: 797 (72.3%); grazed never >=0.5: 195 (14.7%); untouched 28 (2.1%).
  305 truths with covering candidate discarded (median conf of covering candidate 0.545, p10 0.142).
- Polygon offset needs mask_ratio=1: at mask_ratio 4 and imgsz 2048 the loss is on 512 grid; half-pixel
  correction = 1/8 pixel. Rim share: 1275px filament with 200px perimeter: 15.7% of mask on rim at 2048
  vs 62% at 512.
- Distribution shift: train vs test by year (2011-2022 table) and station within 3 points; i.i.d. split.
  Multiplicity: singly annotated 0.4317 (59 records), multiply 0.4454 (121 records), all 0.4404 (180),
  pooled per photo 0.4396 (106) -> weighting moves 0.0008.
- exp_012 refiner on 3429 real pairs (exp_011 harvest): val IoU 0.6108 ep1 -> 0.6335 ep2, never beaten;
  identity + noise. exp_018: real-pair refiner @0.5 PQ 0.4098 (-0.031), mean IoU refined vs coarse 0.674.
- Latent-boundary SQ ceiling ~0.82 (hypothesis).
- exp_015 CPU path reproduced PQ 0.4403668270817509, conf 0.35/area 300/grow -1, 1238 rows, same MD5 as GPU.
- exp_017 logit cut: 14 cuts -0.4..2.0; cut 0.90 no erosion 0.4352; cut 0 + erosion 0.4404; stacking worse.
  "The fatness is not predictable from the field."
- exp_019 field-profile re-ranker: AP 0.7248 vs confidence 0.7150 vs GBM on conf alone 0.6833; emission:
  conf>=0.35 0.4169; prob>=0.35 0.3861; conf or prob>=0.8: 0.4133 (+23 FP, 0 TP).
- exp_010 session 1 (T4 12h): 62/400 epochs at 11.5 min; args: imgsz 2048 batch 2 mask_ratio 2 AdamW lr0 1e-3
  cos_lr; OOM ladder silently fell to mask_ratio 2; LR 9.4e-4 of 1e-3 after 12h (cosine over 400 epochs);
  mask mAP50 0.586 -> 0.619.
- exp_010 on LB: val 0.4274 (SQ 0.6776 RQ 0.6307, TP 766 FP 338 FN 559, conf 0.30 grow 0) -> public 0.36
  (same as exp_002). Offset 0.067 vs 0.078.
- exp_021 two-detector fusion: union 0.4124; agreement gating 0.3740; exp_002 with unconfirmed needing 0.55:
  0.4411 (+0.0007) - removes 7 of 456 FP. Same errors (orphans = real unmarked filaments).
- exp_024 resolution peak for 2048-trained model: 1792 0.4180; 2048 0.4274; 2304 0.4264; 2560 0.4088;
  3072 0.3583 -> peak is the native frame.
- exp_023 fusion public 0.36. Deployment bug: veto ran exp_010 at 0.30 vs 0.05 in validation -> 979 vs 1238
  rows; caught by row count; corrected 1229.
- exp_022 widened operating point: conf 0.30 area 250 grow 0: 0.4274 (interior optimum).
- exp_025-027 crop verifier: 13413 crops (floor 0.05), 128px, 2 channels (photo + mask), 827k-param CNN on
  TPU. AP overall 0.7776 vs conf 0.7770; below conf 0.30: base 0.3442, conf 0.4552, verifier 0.5761.
  Top 100 in-band 72.0% correct; top 300 61.3%. As emission: +100 TP for +496 FP (17% precision).
  Cause: labelled positive if ANY annotator drew it; metric asks THIS annotator.
- exp_028 recall probe (verifier gate 0.50, 1624 instances, val 0.3865): public 0.32 -> test labels not
  more inclusive. Offsets: exp_002 0.078, exp_010 0.067, exp_028 0.067.
- exp_029 boundary snapping: synthetic step-edge filament IoU 0.7986 -> 0.9995; real: guided filter 0.4276
  (inert); intensity snap band 2 0.3933, band 3 0.3491, band 5 0.2739 (destroys). Real filaments fade.
- Five confirmations table (exp_009, exp_018, exp_017, exp_027, exp_029).
- exp_030 mask averaging of two models: A alone 0.4415; distance-transform midpoint 0.4410 (SQ +0.0014);
  union 0.4393; intersection 0.4380. 618 of 836 masks had a partner at IoU 0.5. Correlated errors.
- exp_031: imgsz 1536 batch 1 mask_ratio 1 (grid 1.33 native px), backbone frozen, 50 epochs cosine
  5e-4 -> 5.5e-6: val 0.4343 (SQ 0.6786 RQ 0.6400 TP 831 FP 441) -> public 0.36. Five launches; failing
  allocation is one mask-loss tensor scaling with #instances; empty_cache can't reclaim in-process.
  Completing schedule worth +0.0069. Four models all public 0.36 (local spread 0.0137).
- MAGFiLO public release: 1,593 = annotation passes; 958 unique observations (57.20% once, 19.31% twice,
  23.49% thrice); competition uses 887 (707 train + 180 test) -> ~71 extra; release contains test labels.

## From docs/ACTIVE_EXPERIMENT.md lines 1-900 (operational log, Sep 7-15)
- Host guidance adopted 2026-09-13: judge prediction quality + algorithm logic, visual review required.
- Corrected D4 fusion of exp002 (link 0.5, vote 0.5, conf 0.40, erosion 1px, min area 300): local 0.4570759555;
  20,000-draw photograph-cluster bootstrap gain +0.01662 [0.00387, 0.03032], P>0 = 0.995.
  Test CSVs by exp037 (CPU, 2 shards x 90 photos, 8 D4 views) -> 1,110 instances -> public 0.38 (ref 56096948).
- exp057 fine D4 sweep: 0.4586368117 at conf 0.385, erosion -1, min_area 100 (852 TP, 371 FP, 473 FN) -> public 0.38.
- Intensity (gamma) test-time transforms rejected: identity 0.44106, gamma 0.8 0.42944, gamma 1.2 0.43465.
- TPU forward probe: raw outputs differ from CPU; YOLO TPU training (exp038) timed out in XLA compile.
- exp043 replicated public Mask R-CNN + U-Net notebook byte-for-byte (MD5 09c0eae5...); advertised 0.70 was stale
  (pre-fix); live public 0.34.
- CHAMPION_PAYLOAD: lamhuy8904 public "0.55" notebook copies an embedded CSV; not a model. Not used.
- exp046/047 Mask2Former on TPU (first attempt): custom trainer with CPU Hungarian assignment (shared/query_mask_loss.py),
  1024 px, BF16 + activation checkpointing, deformable sampling in FP32 (grid_sample dtype mismatch fix);
  2,309 steps ~10h TPU; canonical val PQ 0.3001760746 -> rejected. exp058 continuation (2 more epochs) 0.2467558223
  (485 TP 644 FP 840 FN), public 0.20 as a probe.
- exp048 fusion of Mask2 (exp046) with D4: best novel additions IoU<=0.1: 0.4471750055 (TP 868 FP 456 FN 457) vs
  control 0.4570759555 (823/323/502).
- exp055/056 CPU training feasibility: ~32-41 s/step at 1024 -> one epoch in 12h; not launched.
- exp049 erosion 2/3px: 0.4083/0.2981 (1px best).
- exp050 D4 support scoring variants: control .457076, mean .446596, linear support .454846.
- exp051 ideal 256px area-target reconstruction: IoU>.5 fraction at threshold .35 0.9717, .5 0.8255 (target resolution).
- exp059 soft D4 (probability averaging) 0.4569/0.4552/0.4474 vs hard 0.4585 -> loses. exp060 mask NMS best 0.44529
  vs single-view control 0.44145 (below D4). exp061 weight interpolation exp002<->exp031: all worse (0.4349 best).
  Public probes (exp063): NMS 0.37, weight blend 0.36, soft D4 0.38, Mask2 continuation 0.20.
- exp062 D4 conf 0.35 min_area 300 erosion 1: 1233 instances -> public 0.39 (ref 56212792) = first 0.39.
- exp065 (conf .35, minarea 100, +45 small masks) public 0.38; exp064 (conf .30) public 0.38.
- exp066 SAM2 (sam2.1_b) boundary refinement of D4 instances (box / box+3 points): best 0.358379 vs 0.459229 on
  24 photos -> rejected.
- exp067-068 native YOLO retrain at 2048 mask_ratio 1 batch 1 with chunked mask loss (shared/chunked_mask_loss.py);
  FP16 loss NaN -> FP32 validation; zero-gradient-connection fix (proto.sum()*0 -> inf*0=NaN). exp068 5 epochs 2.5h:
  identity 0.4352, D4 0.4470 (best at conf .25). exp071 recall floor .05: D4 0.45098 (< .457).
- exp070 ROI mask loss (shared/roi_mask_loss.py, linearity of interpolation, sample native crop per positive anchor):
  1.576 s vs 2.857 s per step; 5.9 vs 9.4 GiB. exp073 ROI continuation 4 epochs: D4 0.4555 (< 0.4586).
- exp072/075 boundary transfer (donor model masks for matched detections): all variants lose (0.4529-0.4538 vs 0.4559).
- Label-assisted audit: 465 baseline misses: 284 zero overlap, 181 some overlap; only 13 recoverable by joining two
  fragments -> bottleneck is missed-object discovery.
- Hu et al. arXiv:2607.24525 solar-filament workflow (multiscale local-contrast semantic seg, 5x5 kernels, instance
  norm, hierarchical multiscale pooling) -> motivated MORDEN-style semantic stream.
- exp076 physical rescue (dark local contrast, coverage, elongation) of weak D4 proposals: best 0.45463 vs 0.45586.
- exp081 semantic support (frozen exp004 U-Net) for weak D4 proposals: 0.45937 (TP/FP/FN 891/438/434 vs 860/397/465)
  -> exp082 test submission 1,310 instances -> public 0.38 (below 0.39). Adding masks keeps losing publicly.
- exp083 view-count / mean-confidence pruning: all reduce PQ. exp084 area-350 prune: 0.45272, rejected after review.
- exp085 MORDEN-style semantic + spine model (5x5 convs, instance norm, multiscale branches, channel gates, pyramid
  context, 3 channels raw/CLAHE/dark local contrast, focal+Dice): 10 epochs 25.5 min, peak 0.981 GiB, mask Dice 0.703,
  spine Dice 0.635; standalone CC PQ 0.3227 (706/760/619). exp087 continuation 50 epochs: mask Dice 0.7134, spine 0.6735.
- exp088 headroom audit (baseline 0.4558638687, SQ 0.6843 RQ 0.6662, 860/397/465): matched IoU 0.5-0.6: 153,
  0.6-0.7: 318, 0.7-0.8: 320, >0.8: 69. Of 465 misses, 284 zero overlap; 129 overlap at 0.3-0.5 (66 at 0.4-0.5).
  FP: 127 of 397 at 0.3-0.5. Selection oracle over weak D4 pool: 0.4932 (pool best mask per truth mean IoU 0.665).
  Annotator agreement: mean IoU 0.636 over 359 matched pairs; best single mask scored vs every record of its group:
  0.809 (pixel majority), best individual 0.801; our masks SQ 0.680 on same photos -> 0.68 -> ~0.78-0.81 measured headroom.
  Boundary bias on matched pairs: precision 0.817, recall 0.826, area ratio 1.041; 1 dilation 0.4286, 1 more erosion 0.4176.
- exp086 exp085 as pruning evidence: prune semantic_p75 >= 0.15: 0.4593462 (836/332/489); support 0.4575; boundary 0.4409.
- exp090 limb-aware radial prune (semantic_p75 >= 0.30 inside r 0.65, >= 0.12 outside): 0.4606350 (838/328/487);
  bootstrap 500: pruning beats baseline 93.4-95.2%; removal precision 0.68 -> public 0.39.
- exp091/092 refiner v2 on harvested proposals (7442 proposals, 7817 train pairs mean seed IoU 0.5791, val 1017 pairs
  0.6367): crop IoU 0.6485 at epoch 3 (+0.0118).

## From docs/ACTIVE_EXPERIMENT.md lines 900-1450 (Sep 15-16)
- exp092 refiner v2 (crop U-Net on harvested proposals): exact canonical sweep best +0.00068 (one TP) -> not promoted.
  exp094 diagnosis: crop-resolution IoU gain was a nearest-neighbour resampling artefact (+0.0117 with no model);
  refined pasted to native -0.0023. Lesson: measure at native resolution.
- User rule 2026-09-15: download nothing locally; everything on Kaggle.
- LB 2026-09-15: top 0.56, ~20 teams 0.55-0.56; we 0.39. HDJoJo 0.55 recipe: YOLOv8l-seg trained at 2048, single view,
  conf 0.30, NMS IoU 0.0, non-retina, no erosion, area 5, lowest-confidence-first paint. Fususu "LB 0.70" pre-fix.
- exp095 (YOLO inference config probe on exp002 weights): exp033 single view 0.44037; HDJoJo recipe 0.41114; best single
  view (NMS 0.3, conf .35, erode, area 300, lowest-conf-first) 0.44689; D4 emission 0.45586 -> gap is the weights.
- Literature: Reche & Cid 2026 (DETR + per-instance 256 crop U-Net with 3x context and box channel; per-instance IoU 0.70,
  Dice 0.81; radial limb-darkening correction + histogram matching; multi-annotation hurt detector training);
  srivatsav-kannan/solar-seg semantic U-Net holdout PQ 0.346, public 0.31.
- exp097 YOLOv8l-seg @2048 on RunPod (L40 48GB, $0.69/h): convergence-bound controller (constant LR, decay x0.2 after
  12-epoch plateau, stop after 3 decays), RAM cache, preflight batch picker (batch 2: 39.4 GB of 45.5, 9.7 img/s),
  0.5 px polygon offset labels. Pod 1 failed preflight (headroom rule 0.85 rejected only fitting batch; ERR trap;
  host core count). Pod 2 crashed at training start: Ultralytics 8.4.153 _ImageCache has no len(); also final_eval
  calls on_fit_epoch_end twice (phantom epoch), tloss is dict, OOM halves batch silently. exp099 T4 smoke test
  (23 checks) before relaunch. Epochs ~108-110 s. Stage 1 plateau epoch 72 fitness 0.648; decay lifted to 0.686 in
  6 epochs. Converged epoch 124 after 3 decays (1e-3, 2e-4, 4e-5, 8e-6); best raw fitness 0.69496 @ep86; mask mAP50
  0.70395, mAP50-95 0.26242. Cost $2.738 (+$0.22 failed pods) of $9.95.
- exp098 scored stage1_best: single view 0.4295; D4 0.453765 (conf .25 no erosion area 300; SQ 0.6899, 788/283/537)
  vs baseline 0.455864 -> no submission. D4 grid edge bug: fused score = mean conf x (0.5 + 0.5 support) -> sweep from 0.10.
- exp100 stage1 submitted at user's request: public 0.38 (offset 0.0738); 5 test photos had no predictions.
- User direction 2026-09-16: build a pipeline of several models + classical CV + physics, not one model;
  first beat checkpoints/ensembles.

## From docs/ACTIVE_EXPERIMENT.md lines 1450-2050 (Sep 16)
- exp098 v2: stage2_best D4 0.466806 (conf .35 no erosion area 300; 819/269/506; SQ 0.6877 RQ 0.6788) -> submitted,
  public 0.39 (offset 0.0768). Bar for printed 0.40: validation >= 0.472; for 0.46: ~0.537.
- exp098 v3: best.pt == stage2_best == stage3_best (Ultralytics best fitness peaked ep 86-88, copies re-stamped);
  epochs 88-124 bought nothing. last.pt single 0.4497; D4 last 0.4609 (highest recall 837 TP).
- exp103 classical proposals (disk detect, radial-median limb flattening, large-scale background subtraction,
  hysteresis threshold, elongation filter): 0 GT covered on 3 photos. Discriminator "disk kept at 80% truth recall":
  flat darkness 0.374; wide depression (sigma 60) 0.300; elongated black top-hat 6 orientations 0.378;
  Hessian ridge sigma 2/4/8: 0.262 (need < 0.05). Classical proposal branch closed (global); local corridor untested.
- exp101 3-checkpoint 24-view ensemble: 0.466857 (+0.00005) -> same trajectory = same mistakes.
- exp102 cross-architecture (exp002 yolo11m@1280 D4 pool + exp097 YOLOv8l@2048 D4 pool), support denominator 2:
  0.468146 (conf .35, erosion 1, area 100; 852/328/473) -> +0.00134 over best single. Mixed pools need re-swept rules.
  = exp119 submission, public 0.39.
- exp105 component audit (bipartite overlap graph per record): 1:1 953 (55 in 0.4-0.5 band); N:1 frag 29 (5); 1:N merge
  24 (5); N:M 4 (1); 0:1 invisible 286; 1:0 spurious 211. 83% of near-threshold failures are 1:1 shape.
  Oracles: union of preds per truth: 14 cross 0.5; split pred by truth: 40 truths cross.
- PQ leverage (user derivation): object at IoU 0.49 adds 0 to numerator and 1 to denominator (FP/2+FN/2); at 0.51 adds
  0.51 to numerator, denominator unchanged. Rescue of 0.49 miss ~= +0.51 numerator vs polishing 0.55->0.60 = +0.05.
- exp105b near-threshold 1:1 (n=55): median IoU 0.459, missing/inter 0.76 (under-segmented; median 575 px missing vs
  139 spurious); healthy n=845 median IoU 0.692. exp105c geometry AUC: elongation 0.539, solidity 0.424, width CV 0.457,
  combo 0.548. Narrow-neck rho (min/median width) AUC 0.712 but precision 5-7% at every cut -> rejected.
- exp104 morphology on ensemble pool: grow -2 0.4311; -1 0.4681; 0 0.4412; +1 0.3828; +2 0.3055.
- exp106 semantic annulus support (exp085 maps; exterior ring dilate 4 vs inner band erode 4): AUC 0.397-0.543;
  corrected both directions: sem_interior negated 0.635; no net-positive operating point.
- exp108 where missing pixels are: near-threshold: axial fraction 0.590, distance median 10.9 px, p90 30.7 px, beyond
  10 px 51.4%, half-width 2.2 px; healthy: axial 0.132, median 2.0 px, 75.7% within 3 px -> truncation, not thinness.
- exp107 cross-architecture disagreement trigger: AUC 0.595/0.574, all operating points net-negative.
- exp109 cone ahead of endpoint (35 deg, 4-30 px): truth_ahead AUC 0.655 (instrument valid); semantic prob 0.0005 vs
  0.0008; spine 0.0017; orientation 0.736 vs 0.720 -> nothing to walk along; even oracle label precision 0.295.

## From docs/ACTIVE_EXPERIMENT.md lines 2050-2650 (Sep 16)
- exp110 best-path per-ray stats: truth_best_ray label AUC 0.621; best_ray_min 0.0 both populations; dark_best_ray
  (raw absorption vs sigma-25 background) AUC 0.611 (only image feature > 0.6).
- exp111 width profile along principal axis: mean_slope AUC 0.809 (merges vs healthy), saddle depth 0.750 (0.519 vs
  0.165), peak_ratio 0.718; precision wall: 24 targets vs 858 healthy.
- Independent assessment correction: finite cost is not impassable (Dijkstra). 16 approaches: 3 tested, 3 partial,
  10 never implemented. Demotion by shared precondition != refutation.
- exp112 geodesic with finite cost alpha*(1-dark) + beta*(1-orientation) + step, 20 px ahead in 45 deg cone:
  cost AUC 0.559 negated; dark 0.605; orientation 0.518 -> trigger weak; operator untested.
- Testing manual (Appendix B): step 1 oracle ceiling per repair family; step 4 rasterise and score officially.
- exp113/113b ORACLE ceilings (read GT, bounds only), baseline 0.4558638687 (860/397/465):
  growth_full +0.032510 (49 masks, 56,352 px, 49 up/0 down); split_by_truth +0.022342 (28 merges -> 57 pieces,
  40 up); growth_lateral +0.018865; growth_axial +0.018103; union_fragments +0.011579 (14 up).
  axial+lateral 0.036968 > full 0.032510 (overlap).
- exp114 real splitters with oracle gate: neck_cut +0.002846 (12.7% of ceiling; 22 up / 6 down); watershed
  -0.005072; ungated neck -0.2518, watershed -0.4103 (PQ -> 0.0455).
- exp115 watershed constants: best +0.001938; placement (not count) is the problem: rescues 22 (neck) / 18 (ws) of 40.
- exp116 interface cut (min-cost path between width maxima): 1 of 6 split; path 6 px long.
- exp117 growth operators under oracle gate: control dilate 2 +0.006350 (33.7% of lateral ceiling); dilate 1 +0.004548;
  absorption-guided band grow best +0.003283; axial ray best +0.001204. Ungated dilate 1 -0.026434 (11 up / 43 down),
  dilate 2 -0.080075. Gate break-even TPR/FPR > 43/11 = 3.9.
- exp117 addendum 2 arithmetic (baseline IoU sum 588.52): recovering N misses at IoU 0.70: 25 -> 0.4649, 50 -> 0.4738,
  100 -> 0.4911, 200 -> 0.5237, 284 -> 0.5494. Suppressing N pure FP: 50 -> 0.4649, 100 -> 0.4742, 200 -> 0.4941.
  0.472 needs 45 recoveries; 0.530 needs 221.
- exp118 recall ceiling (pool 1414 candidates, 813 above 0.35, 601 in [0.10,0.35)): blind floors 0.30 0.4501
  (+32 TP +95 FP, precision 0.252); 0.25 0.4421 (0.235); 0.20 0.4260; 0.15 0.4027; 0.10 0.3969.
  ORACLE admit: 0.476686 (+0.020822), 132 admitted, 120 misses recovered at mean IoU 0.651.
  Break-even k*q > 0.5*PQ*(k+m): at q=0.651, m < 1.86k -> 35% precision for zero gain; ~50% needed at k~120 for 0.472.

## From docs/ACTIVE_EXPERIMENT.md lines 2650-3270 (Sep 16-17)
- Required verifier precision vs recovered IoU: 0.55 -> 0.414; 0.60 -> 0.380; 0.651 -> 0.350; 0.70 -> 0.326.
- exp120 single-feature verifier screen on 526 band candidates (132 admissible, base 0.2510): score AUC 0.641
  (best op precision 0.396), limb flipped 0.602 (centre more likely real), others 0.50-0.54.
- exp121 score q0.90 AND limb low q0.70: +0.000480 (2.3% of ceiling), crossings down always 0 (painting gives
  contested pixels to higher score). Units lesson: instances per photograph (106) vs pooled per record (180).
- exp122 admission geometry (whole / largest component / all pieces): spread < 0.0007 -> not a lever.
- exp119 = exp102 ensemble submitted: replay 0.46814553 (852/328/473), 1153 rows, public 0.39.
- exp125 learned verifier (9 features, GroupKFold by photo, out-of-fold): logistic top-20 +0.001698 (8.2% of ceiling,
  precision 0.469); gradient boosting all negative (top-20 precision 0.200 < base; memorises 132 positives).
- exp126 isolation features (claimed_fraction, nearest_gap) at chance; +0.002772 dismissed as noise (4 crossings,
  best of 8, OOF AUC fell 0.652 -> 0.647).
- exp124 cross-architecture confirmation (exp097 proposes overlap IoU>=0.5): AUC 0.688 (best single feature);
  operating-point grid too loose -> negative.
- exp129 stage decomposition (best IoU any pool candidate reaches, no floor/erosion/area/paint): >=0.5 matchable 1016
  (76.7%); 0.3-0.5 boundary 194 (14.6%); 0.1-0.3 weak 50 (3.8%); <0.1 never proposed 65 (4.9%).
  Oracles: emission only 0.5057 (+0.0498); boundary only 0.5149 (+0.0590); both 0.5566 (+0.1007).
  Retracted "problem is entirely detection" and "284 zero-overlap" (emitted-set statistic).
- exp130 emission losses (203 truths with raw proposal >0.5 unmatched): below conf floor 168 (82.8%); area floor 13;
  erosion 12; painting 10.
- exp131 raw proposals too FAT for 0.3-0.5 bucket: excess/inter 0.919, 58.8% too large; erode 1 rescues 56 (28.9%).
- exp132 corrected decomposition in eroded representation: matchable 880 (66.4%); lost to conf floor 172 (13.0%);
  true boundary 165 (12.5%); weak 45; never proposed 63 (4.8%). Erosion lifts proposal recall 76.7 -> 79.4%.
  Stricter vote substitutes for erosion.
- exp134 CNN verifier on raw pixel crops (exp091 harvest 7,442 proposals; per-record labels fixed; 5,281 train 19.3% pos,
  1,016 val): AP 0.19-0.25 vs base 0.178, loss stuck at 1.11; all admission rules lose; darkness contrast AUC 0.5345.
  -> low confidence here is correct uncertainty.
- exp135 boundary bucket split: 98 emitted, 67 below floor; half-width 2.80 px, long side 90 px.
- exp136 grid ceilings: stride 2 (1024) 0.976; stride 4 (512) 0.914 (thin 0.896, 99.9% clear 0.5); stride 8 (256)
  0.771 (thin 0.713, 95.8%).
- exp137 mask_ratio 4 target: continuous 0.914, binary-cover 0.818, binary-nearest 0.702 -> mask_ratio 1 recovers
  <= +0.096 vs own GT.
- exp138 annotator A vs B (5,816 pairs on 296 photos): raw full-res 0.576 (73.8% clear 0.5); continuous 512 0.577;
  binary-cover 512 0.564 -> surviving quantisation cost +0.013.
- exp139 design: previous M2F exp046/058 ~4 epochs of 50, 1024 input (stride 8 vs native), swin-tiny, custom Hungarian
  loss -> val 0.2468 says nothing about architecture. New: facebook/mask2former-swin-small-coco-instance, 100 queries,
  HF 4.57.6 loss unchanged; native crops + 25% whole frames; pycocotools native targets (HF samples points ->
  sub-pixel); D4 + gain/gamma; AdamW 1e-4 wd 0.05 bb x0.1, no decay on norms/embeddings/pos tables, clip 0.01;
  warmup 500, cosine; val PQ every 5 epochs.
- Facts: max 23 filaments per train record, 26 per val record (100 queries never short). Filament long side p50 96,
  p90 235, p99 515.5, max 1067; 1.06% > 512. Sine position embeddings normalize=True (fractions of input extent) ->
  whole frames needed. Orientation vs latitude: |y|<0.3 isotropic 50.0% E-W; 0.3-0.6 60.5%; >0.6 88.1% (185, 3.8%).
  bf16: matcher cost promoted to fp32 by torch.minimum with fp32 scalar.
- Memory (CPU measured, full Swin-S M2F bf16 fwd+loss+bwd): 512px 1.629 GiB (ckpt 1.079); 768px 3.168 (2.035);
  linear in pixels. Two 1024 crops ~11.5 GiB; four ~21.9; whole 2048 ~22.0 (14.4 with checkpointing).
