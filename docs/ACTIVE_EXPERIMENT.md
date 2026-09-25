# Active objective and resource constraints

**Mandatory decision context:** [Host guidance](HOST_GUIDANCE.md), explicitly
adopted by the user on2026-09-13. Evaluate individual prediction quality and the
logic of the algorithm alongside PQ. Visually review representative predictions
and changed instances before promoting a variant; do not infer a noise ceiling
or optimize implausible masks solely for score. Existing authorized public probes
may continue. This supplements, and does not cancel, the >0.46 goal below.

Latest user instruction: obtain a verified public leaderboard score **greater
than 0.46** for filament-segmentation-2026. Training, inference, postprocessing and
submission are authorized. As of2026-09-13, Kaggle GPU and TPU use are authorized
again. User explicitly specifies **T4 GPUs and a30-hour GPU budget**. Always
select NvidiaTeslaT4; do not rely on the default accelerator. No further RunPod
access or changes. This supersedes the earlier GPU prohibition.

Verified current public best: **0.39** (Kaggle submission ref `56212792`, 2026-09-13).
Canonical exp_002 validation baseline: 0.4403668271. A code improvement or a local
PQ gain does not complete the objective; only the public submission score does.

The previous RunPod inference was never training. An attempt to cancel just that
process after the new restriction returned SSH connection refused. Do not retry
or change that instance. No current RunPod runtime state is asserted.

Current work (2026-09-14): exp063 probes completed and scored; best remains0.39.
exp065 small-mask D4 probe(ref56222273) COMPLETE at0.38. exp066 SAM2 T4 screen
COMPLETE and rejected: best0.358379 versus unchanged0.459229 on24 photographs.
Artifacts downloaded to data/exp066_output; no full validation or submission.
exp067 v4 COMPLETE: five crowded-record optimizer steps at2048/mask_ratio1/batch1,
FP32 validator on four held-out records atbatch2, checkpoint reload and prediction
all passed. Peak GPU allocation10,116,368,896 bytes. FP16 segmentation validation
loss was NaN; FP32 loss is finite. exp068 v1 COMPLETE after5 epochs/2.503h.
Identity PQ.435155642 and D4 PQ.446964013, both best atconf.25 (lowest searched),
grow0/minarea300. exp071 is RUNNING on T4 to extend the confidence floor to.05
before rejecting this model. exp070 ROI loss comparison COMPLETE and faster;
use measured reports for subsequent training choices, not as score evidence.
Historical sol-exp032-at1280 is COMPLETE but uses
the older exp031 trainer, not experiments/exp_032_chunked/src/train.py.
Preserve canonical seed/split 2026 and score individual annotation records.

2026-09-14 19:30 Nairobi continuation:
- Main exp068 training and evaluation COMPLETE, slug unchanged. Best/last have
  identical predictions. Best weights data/exp068_output/runs/native/weights/best.pt,
  45,624,950 bytes, SHA256 efe905fd14c9089221acb80b8816e349ec2d82c9b2b9226cbad5d1d9f555e95a.
  Metrics downloaded separately to data/exp068_reports/validation/report.json.
- Coarse D4 .446964013 (SQ.690762565,RQ.647058824,759TP262FP566FN) is below old
  baseline, but optimum at inference floor.25 means confidence search incomplete.
- exp071 cosmicwitness/sol-exp071-native-recall v1 RUNNING, T4, no training.
  Same weights SHA asserted remotely; native inference floor.05/max_det1000,
  thresholds.05/.075/.1/.15/.2/.../.6, grow0/-1,minarea100/300, full106 photos,
  identity and D4. Source scripts/package_native_recall.py. Cap3600s/worker3000s.
  Watcher wrapper15292(worker712), stream6284; data/exp071_output and exp071 logs.
  Initial single-view inference reached100/106. No automatic submission.
- exp070 comparison passes both five-step training/FP32 validator/reload gates.
  Warm full optimizer-step means2.857s chunked versus1.576s ROI on same crowded
  records. Reports data/exp070_reports/probe_{chunked,roi}/probe.json. ROI uses
  explicit FP32 mask arithmetic and is not bit-identical AMP training; first loss
  agrees within1e-6, subsequent optimizer trajectories differ slightly. Accuracy
  still needs training evaluation. Correcting zero connections also makes the
  chunked FP16 diagnostic finite, confirming the prior overflow mechanism.
- Large artifact transfers were repeatedly restarted after transport errors and
  the watcher's600s overall deadline. scripts/download_kernel.py now preserves
  per-file progress, verifies completed files, resumes byte ranges bound to the
  immutable Kaggle /kf/<version>/ object id, and never stores access-token path
  segments. Three tests cover interrupted transfer, verified skip and changed
  remote objects. Watcher relies on per-request timeouts, removing600s restarts.
- Restarted ONLY terminal-job download watchers: exp068 wrapper11372,
  exp070 wrapper27452, *_watch_resumable.log/.err. Large epoch checkpoints still
  downloading; .download-complete.json is the all-artifacts completion marker.
- Verified actual Kaggle proxy Range support: HTTP206 with Content-Range
  bytes100-199/1192 on results.csv. Resume identity uses immutable version349760201
  for exp068, local SHA256 for completed-file integrity, never signed-token paths.
- exp070 peak GPU allocation9.422GiB chunked versus5.948GiB ROI; both finite
  FP16 diagnostic losses after the zero-connection fix. No new long training
  launched with ROI yet; first finish exp071 confidence extension.
- exp071 remote source byte-identical and T4/GPU=true/TPU=false verified by pull.
- exp071 COMPLETE. Native recall floor.05 inference and full D4 finished. Its
  trained model's best D4 is0.450982644 (conf.25,grow0,minarea300), below the
  existing D4 .457075956. No submission was generated because validation did not
  justify spending a slot; report and caches are data/exp071_output/validation.
- exp072 COMPLETE local boundary-transfer screen: matched602 base instances to
  exp068 masks at IoU>.6. Native/intersection/union variants scored .451182/.452899/
  .451892 versus unchanged .455864. Best is unchanged baseline; reject. Its
  purpose was to test whether the new model's boundaries help existing detections
  without changing emission. No submission.
- exp064 CSV submitted ref56239220 after validation/file/schema/overlap checks;
  it is still PENDING, monitored by scripts/watch_submission.py (wrapper28724,
  data/submission_56239220.json and logs). Three daily slots remain after upload.
  Do not duplicate while pending.
- exp064 ref56239220 COMPLETE at public0.38. This confirms the lower-confidence
  D4 recall rule (local validation .450091) does not improve the leaderboard.
- GPU quota API currently reports16310.18/21600 seconds used, with exp073
  reserved/running. Remaining allowance is roughly88 minutes (API string is
  `16310.178571.0s`; do not interpret the middle digits as hours). User's broader
  30-hour authorization remains recorded, but Kaggle's current account quota is
  the enforceable runtime limit until its refresh.
- exp073 cosmicwitness/sol-exp073-roi-continuation v1 RUNNING on explicit T4.
  It starts from exp068 best SHA efe905fd14c9089221acb80b8816e349ec2d82c9b2b9226cbad5d1d9f555e95a,
  uses ROI loss, lr0=.0001, four-epoch max with measured time=.9h, native2048,
  mask_ratio1, batch1, FP32 validation. This is a short continuation, not a
  claimed improvement. Watcher26412, stream21288, output data/exp073_output and
  logs exp073_watch/stream .log/.err. No automatic submission; review checkpoint
  and canonical validation before deciding whether the saved weights are useful.
- exp073 v1 was ERROR before first optimizer step: Kaggle has no git checkout and
  the wrapper omitted SOL_SOURCE_REVISION, so metadata's git fallback failed.
  No model state was produced; terminal output is preserved in data/exp073_output.
  v2 pushed only after v1 became terminal, adding the fixed source revision env.
  Observe existing v2 run in data/exp073_v2_output (watcher15160, stream3048).
  It is the same bounded experiment, not a changed training hypothesis.
- exp073 v2 reached `Starting training for 0.9 hours` on Tesla T4 and entered the
  native2048 training loop; no assertion or setup error observed. Keep polling
  its existing watcher; do not push another version.
- After setup, the first real ROI batch produced only a deterministic
  `grid_sampler_2d_backward_cuda` warning and no epoch row for over five minutes.
  GPU quota rose from16580.9 to16666.9 seconds. This is materially slower than
  the five-step crowded probe because training has hundreds of positive anchors;
  keep observing the authoritative RUNNING session through its bounded 0.9h cap,
  then inspect whether any checkpoint exists. Do not restart from logs alone.
- Later v2 logs show epoch1 at batch597/974 with ROI seg_loss1.578, GPU6.38GiB
  and finite losses; quota moved to about17068 seconds. The progress renderer
  has not emitted a newer line on the last poll, but the authoritative Kaggle
  status remains RUNNING and quota continues increasing. Keep the worker alive.

2026-09-14 14:17 Nairobi:
- exp067 v2 proved finite training after three normal AMP loss-scale skips
  (65536 ->8192) but failed FP16 EMA segmentation loss. v3 isolated that loss as
  NaN with finite model state; FP32 components all finite. v3 then hit a probe-only
  missing epoch attribute. v4 fixes this and passes actual FP32 batch2 validator.
- Probe checkpoint construction now writes explicit segmentation train_args and
  clears stale source EMA; CPU lifecycle test catches task-metadata reload errors.
  Both CLI precision modes and the probe lifecycle pass after the fix.
- v4 remote COMPLETE and live report status complete. Watcher wrapper17840,
  worker2220; stream24616. Artifacts data/exp067_v4_output, exp067_v4 logs.
- Production trainer will use warmup_bias_lr0.0, avoiding upstream AdamW default
  0.1 versus lr0=.0005. This is an explicit CLI-only difference from the resource
  probe, which used full lr0 directly. Package gate permits only this exact source
  difference; criterion and validator source remain unchanged. No claim that this
  explains prior public scores without a controlled comparison.
- scripts/package_chunked_train.py requires completed matching GPU probe. Training
  starts exp002, whole model,2048/mask_ratio1/batch1/chunk8, AdamW.0005, FP32
  validation, 2.5-hour Ultralytics measured-time schedule (initial8 epochs, adjusted
  using epoch duration). Its best and last checkpoints get full106-photo PQ
  evaluation and the best gets D4 if identity PQ>=.435. Save RLEs for visual review.
- scripts/evaluate_yolo_checkpoints.py does no submission. Public best stays0.39;
  latest checked submissions unchanged. No test annotations are used.
- exp068 v1 push succeeded after completed/downloaded v4 probe and passing CPU
  lifecycle tests. Kaggle created slug from title; actual slug is
  cosmicwitness/sol-exp068-native-mask-training (not sol-exp068-chunked-train).
  Local metadata/package updated to actual slug, no second push. Pulled code is
  byte-identical; T4/GPU=true/TPU=false confirmed in data/exp068_remote.
- Hidden watcher wrapper24104, stream wrapper24780. Outputs data/exp068_output,
  exp068_watch.log/.err and exp068_stream.log/.err. Kernel cap14400s, training
  worker10000s with 2.5-hour measured schedule, validation worker2000s. No auto
  submission; review canonical PQ and native prediction overlays on completion.
- exp068 API RUNNING. Live resolved metadata saved data/exp068_resolved_live.json:
  ChunkedSegmentationLoss, FP32 validation,2048/mask_ratio1/batch1, AdamW.0005,
  warmup_bias_lr0.0,time2.5,freeze0 all confirmed after trainer setup.
- Concurrent CPU-only exp069 screen reuses frozen exp026 verifier checkpoint
  downloaded to data/exp026_reuse; no new neural training. Source
  scripts/reuse_verifier_scores.py; output data/exp069_verifier_reuse, local
  execution session53196. Uses existing D4 masks, fixed original-confidence
  painting/erosion/minarea300, per-annotator binary match labels and five folds
  grouped by photograph for logistic calibration. Compares confidence-only with
  confidence plus verifier logit; full selection and promotion-only rules.
- This is exploratory: old verifier was selected on canonical validation AP;
  only calibration is held out by photograph. Never present it as fully unseen
  validation or promote without actual pooled PQ and visual review. Historical
  claims that per-annotator relabeling cannot help were not proven; this screen
  tests the reusable scores directly. No public submission is armed.
- exp069 COMPLETE: baseline PQ.455863869 exactly reproduced. Best two-feature
  verifier calibration .455403976, worse; promotion-only best reproduces baseline.
  Confidence-only calibration .456398557 is a negligible +.000535 exploratory
  difference, not grounds for promotion or submission. Reject this route and keep
  the evidence in data/exp069_verifier_reuse/report.json and pool.json. No new GPU
  time used. exp068 remains RUNNING under its existing watcher.
- Fixed local SSE log decoding to force UTF-8 (Requests' text/* Latin-1 default
  split progress glyph bytes into control characters and broke JSON events).
  An inline regression using a U+2585 progress glyph passes through the actual
  Kaggle SDK parser. Reconnected ONLY the log reader, wrapper19372, with logs
  exp068_stream_utf8.log/.err; training and watcher were not restarted. Existing
  JSONL rows are deduplicated on log-reader restart. Progress output may still
  buffer until an epoch newline; lack of a recent line is not terminal evidence.
- exp068 subsequently logged finite losses through batch594/974 of epoch1,
  GPU memory9.7GB, about1.6s/batch. This confirms optimizer-loop progress beyond
  setup. Native run remains unchanged in its packaged remote source.
- exp070 ROI-loss comparison v1 pushed after43 passing tests. Actual slug
  cosmicwitness/sol-exp070-roi-probe, T4 confirmed and pulled source byte-identical.
  Five identical crowded-record updates, real batch2 FP32 validator, save/reload
  for corrected chunked and ROI criteria in separate processes on the same job.
  Watcher wrapper23432, stream7968; data/exp070_output and exp070 logs. Cap3600s.
- New shared/roi_mask_loss.py uses linearity of interpolation to compute a single
  low-resolution logit and sample its native crop per positive anchor. Explicit
  FP32 mask arithmetic; eight loss/gradient parity cases plus actual CLI/probe
  lifecycle pass. CPU component benchmark results/exp070_roi_cpu_benchmark.json:
  warm1.539s chunked versus.556s ROI, matching FP32 losses/gradients. This does NOT
  establish full-model GPU speedup; wait for the real T4 comparison before use.
- Corrected zero-valued autograd connections in shared/chunked_mask_loss.py:
  proto.sum()*0 can become inf*0=NaN when reducing native FP16 prototypes; indexing
  one value times0 preserves zero gradients without overflow. Regression tests
  cover positive and empty batches. Current exp068 uses its old embedded source
  with FP32 validation and is unaffected. New local source has not replaced it.

2026-09-14 exp067 handoff:
- Slug cosmicwitness/sol-exp067-chunked-probe v1, explicit NvidiaTeslaT4,
  GPU=true/TPU=false confirmed by pulled metadata in data/exp067_remote.
- 31 tests pass (chunked loss/gradient parity and actual training CLI lifecycle).
  Probe additionally checks finite gradients, optimizer allocation, crowded EMA
  loss, checkpoint save/reload and held-out forward. This is not PQ evidence.
- Source scripts/package_chunked_probe.py and scripts/probe_training.py;
  exact embedded hashes plus source revision saved remotely. Existing exp002
  checkpoint and competition training annotations must match verified SHA256s.
- Five real steps, 1500-second probe timeout, 2400-second kernel cap; no fallback
  or automatic long training. Size a complete schedule using measured results.
- Hidden watcher wrapper1924 and stream wrapper22064; outputs data/exp067_output,
  logs data/exp067_watch.log/.err and data/exp067_stream.log/.err. Observe the
 existing job; never relaunch because a status/log request times out.
- Older sol-exp032-at1280 API lists no output files; its COMPLETE status alone
  does not prove successful training or a usable checkpoint.
- v1 subsequently ERROR: first backward triggered the probe's combined
  nonfinite-loss/gradient check. It did not OOM, but the old check did not
  distinguish forward failure from normal initial AMP scaling overflow. Original
  terminal outputs are preserved in data/exp067_output (downloaded=true).
- v2 corrects the diagnostic: forward nonfinites fail immediately; AMP may skip
  overflowing updates and lower its scale for at most17 attempts per batch.
  Five finite-gradient updates remain mandatory; scale changes and bad layers
  are recorded. This does not declare the numerical issue solved in advance.
- v2 pushed only after v1 was terminal and fully downloaded. New watcher wrapper
  11456(worker20652), stream wrapper19572, data/exp067_v2_output and corresponding
  exp067_v2 logs. No automatic long training or submission. Review probe.json,
  resolved_experiment.json and source manifest before deciding the next launch.

2026-09-07 current runs:
- `cosmicwitness/sol-exp033-frozen-cpu`: RUNNING; identity/gamma 0.8/gamma 1.2.
- `cosmicwitness/sol-exp034-tpu-forward`: QUEUED; native exp002 TPU forward probe.
- `kaggle/exp035_mask2_probe`: prepared, push rejected because Kaggle permits
  only one batch TPU session. Not launched. Run after exp034 terminates.
- Local `experiments.exp_033_inference.src.cached_tta`: exp005 cache reproduces
  PQ 0.4403668270817509 exactly. Testing the confidence threshold omitted from
  the historical D4 fusion evaluation. First fused cache saved in
  `data/exp033_cached_tta`; resumed with equivalent prefix-paint evaluation.
- Quota API reports 72,000 TPU seconds allowed, 1,082.856 used (~19.7h left).
  GPU remains forbidden regardless of quota reporting.
- Fourteen exp033 tests pass, including prefix painting equivalence for
  overlapping masks, tied confidence, erosion and duplicate annotation records.

Public-method check: `lamhuy8904/solar-filament-unet-segmentation-0-55`
does not generate its submitted predictions with the displayed trained U-Net.
Its test loop copies an embedded compressed CSV (`CHAMPION_PAYLOAD`). The
payload's provenance is unverified; do not use it as model-quality evidence or
as our submission. Source inspected read-only under ignored `data/public_code`.

Measured progress:
- Corrected D4 fusion (link 0.5, vote 0.5, confidence 0.40, erosion 1px,
  min area 300) reaches local PQ **0.4570759555**. Public score remains 0.36.
- Sparse RLE voting reproduces all 1,414 first-rule fused masks byte-for-byte,
  including scores, in 1.73s. Remaining fusion settings are still being scored.
- CPU ONNX probe completed; warm timings are noisy and do not establish a speed
  gain. Do not switch deployment backend on this evidence. Existing PyTorch used.
- `cosmicwitness/sol-exp037-test-0` and `sol-exp037-test-1`, version 1, pushed:
  CPU only, 90 test photographs each, eight D4 views, source floor confidence
  **0.25** (matches historical exp007), native 2048, IoU 0.60, max_det 100.
  Both first compare three validation photographs with the historical cache;
  reject candidate-count, mask-IoU or confidence discrepancies. They only harvest
  candidates; final fusion configuration is selected from validation results.
- Local active sweep was restarted to use sparse voting and resume saved rows.
  Current terminal session: 27687. ONNX session 78193 finished successfully.

2026-09-08 update:
- Full corrected D4 sweep complete: best remains PQ 0.4570759555. A 20,000-draw
  photograph-cluster bootstrap estimates gain +0.01662, 95% interval
  [+0.00387, +0.03032], P(gain>0)=0.99515.
- CPU identity/intensity sweep completed. Current-runtime identity PQ 0.44106;
  gamma 0.8 PQ 0.42944 and gamma 1.2 PQ 0.43465. Intensity transforms rejected.
- TPU forward probe completed: warm eight-image 2048 forward 0.794s, finite,
  but raw outputs differ substantially from CPU. Do not use TPU predictions
  until end-to-end canonical PQ validates them.
- `sol-exp038-yolo-tpu-probe` is RUNNING: two real optimizer steps at 1024/mask2,
  1024/mask1, and 2048/mask1. It contains GPU-disabled metadata.
- CPU test shards `sol-exp037-test-{0,1}` v5 are RUNNING. Historical-cache gate
  accepts the verified version skew: identical candidate count, max confidence
  difference 2.1e-6, min mask IoU 0.9613 and mean IoU 0.9861 on the gate photo.
- User explicitly authorized new model training on TPU and TPU CSV generation.

2026-09-08 continuation after usage-limit interruption:
- `sol-exp037-test-{0,1}` v5 completed and both artifacts passed their historical
  cache gates. The selected D4 rule produced 1,110 disjoint instances for all 180
  test photographs. Submission ref `56096948` scored 0.38; local PQ 0.4570759555.
- `sol-exp038-yolo-tpu-probe` completed. All three configurations timed out during
  XLA initialization/compilation before an optimizer step. Do not launch the full
  YOLO TPU trainer without changing the graph/runtime approach.
- `sol-exp043-public-cpu` reproduced the imported public submission byte-for-byte
  (MD5 `09c0eae5fdff29a91f8a740475ecaf36`). The live public score for ref
  `56091487` is 0.34, disproving the artifact's stale advertised 0.70 score.
- `sol-exp039-tpu-validation` v1 was pushed to test end-to-end PQ for TPU inference
  from the existing exp002 weights. `exp035_mask2_probe` remains next because the
  account permits only one concurrent batch TPU session.

Continuation: user deleted queued exp039. exp035 v1 then ran, but tiny1024,
tiny1536 and small1024 all hit 900s timeout without two reported optimizer steps.
The logs do not establish the cause: tiny1536 reached loss scalar conversion.
exp035 v2 pushed: tiny256 then tiny1024, explicit synchronized stage timings,
three finite optimizer steps required, stop on first failed configuration.
Hidden monitor downloads v2 artifacts to data/exp035_output_v2 and ends for review.
Long exp046 training must not launch from this synthetic probe alone: its real-data
loss differs, schedule/checkpoint handling needs validation, and the prior 3300s
launch timeout contradicted its 10h budget. Monitor now requires an exp046-validated
marker after a matching real-data lifecycle check; that marker does not exist.
Fusion sweep completed: D4 control PQ .457076 remains best; best cross-model fusion
.450529 is rejected. Public best remains .38; no subsequent submission.

2026-09-08 training correction:
- Replaced exp046's unvalidated fixed-order query objective with CPU Hungarian
  assignment and fixed-shape class/BCE/Dice loss in shared/query_mask_loss.py.
  Four tests cover target/query permutation, gradient equivalence, inactive
  padding, empty records, nonfinite input and excess-instance rejection.
- New source kaggle/exp046_mask2_train/train.py completed three real-image CPU
  lifecycle steps using a small random 128px model, then saved/reloaded a checkpoint
  and ran a held-out photograph. This is code validation, not accuracy evidence.
- The trainer uses canonical split, one annotation record/sample, fractional area
  targets, measured step-budget cosine schedule and atomic checkpoint every100 steps.
- scripts/package_mask2_train.py embeds identical code for exp047 real TPU probe
  and exp046 training. Neither has launched yet.
- Replaced the PowerShell monitor body with launcher for scripts/monitor_training.py.
  It retains current exp035v2, downloads its diagnostics, launches exp047 only on
  finite 1024px completion, and launches exp046 only on matching-code real-data
  TPU probe success (3 finite steps, warm steps<30s, checkpoint reload+val forward).
  Three monitor tests prevent duplicate launches and restart on observation errors.
- Monitor logs/status: data/monitor/training-monitor.log and training-state.json.
  It automatically stops for review on failed probes and after training download.

exp048 CPU evaluation prepared and packaged. Uses identical preprocessing, native
mask resizing, query mask-quality scores and confidence painting for validation
and test. Canonical pooled PQ selects among 24 coarse rules; CSV generation only
runs when all 106 validation photographs complete and PQ exceeds .4570759555.
CPU lifecycle checkpoint smoke exercised inference/rasterization/scoring and
correctly refused submission generation. Monitor now advances finished exp046 to
exp048 and downloads its results, then stops for submission review or a new idea.
Kaggle quota checked through authenticated CLI: 7072.447 TPU seconds used of72000,
reserved0; exp035v2 remains QUEUED with failureMessage=null. MCP returned
Unauthenticated, so CLI remains the functioning connection. Queue cause unknown.

2026-09-09 00:18 Nairobi continuation:
- exp035v2 COMPLETE: 256px completed three finite optimizer steps (447s,265s,1.27s).
  1024px FP32 failed at first forward synchronization with TPU HBM allocation OOM.
- Added BF16 autocast and activation checkpointing to exp046. Checkpoint wrapper
  flattens nested tensor arguments so decoder feature lists retain gradients.
  Nine loss/checkpoint/monitor tests pass, including FP32/BF16 gradient equivalence.
  Four CPU smoke steps on real records also completed save/reload and held-out forward:
  data/exp046_cpu_bf16_checkpoint_fixed/train_report.json. This verifies code only.
- exp047 real-data 1024px TPU probe v1 pushed and currently QUEUED. Pulled remote
  metadata confirms GPU false, TPU true, Tpu1VmV38; embedded trainer and helper
  match local bytes. Probe requests four steps, BF16 and checkpointing, 5400s cap.
- Background Python monitor restarted (worker PID40132), reviewing exp035 OOM as
  reason for corrected probe. It requires four finite real TPU steps, last two
  below30s, matching trainer, BF16/checkpointing and reload+held-out inference before
  launching exp046, then CPU exp048 evaluation. Failed gates stop for review.
- TPU quota: 8354.295 of72000 seconds used, reserved0 before probe push.
- exp049 erosion screen completed. Two/three-pixel erosion worsened PQ to .4083/.2981;
  one-pixel control .4570759555 remains best. No new CSV or submission; public .38.

2026-09-09 00:21 Nairobi:
- Previous goal turn made progress: corrected exp047 pushed, remote code/config
  verified and monitor live. Current authoritative status remains QUEUED.
- Started scripts/screen_d4_support.py in background (worker37868, launcher11620).
  Reconstructs exact cached D4 clusters and asserts original masks/scores, then
  screens mean confidence and confidence multiplied by view support against the
  deployed half-strength support penalty. Uses canonical per-annotator pooled PQ,
  no new inference or training. Output data/exp050_d4_support; logs support.log/err
  under data/monitor. Review results before any CSV or submission change.
- exp050 completed in300s: control .457076, mean .446596, linear support .454846.
  Both alternative rankings rejected; retain deployed rule. No submission spent.
  exp047 remained QUEUED at the final authoritative status check; monitor40132 live.

2026-09-09 00:34 Nairobi:
- Previous turn was a verified wait on queued exp047 and live monitor40132;
  authoritative submission listing reconfirmed ref56096948 public .38.
- exp051 measured ideal 256px area-target reconstruction on 32 seeded TRAIN-only
  records (212 instances). IoU>.5 fraction: threshold .35=.9717, .4=.9340,
  .5=.8255, .6=.6226; .2=.8019. This measures target resolution, not model quality.
- Added .35 to exp048 evaluation mask thresholds, retaining .4/.5/.6 (32 rules).
  Packaged CPU evaluator updated; one-photo smoke completed all rules and refused
  CSV generation as required. Queued trainer/probe source remains unchanged.
  Diagnostics: data/exp051_target_resolution/report.json;
  evaluator smoke: data/exp048_cpu_threshold_smoke/validation.json.

2026-09-09 00:38 Nairobi:
- exp047 remains QUEUED and TPU quota unchanged at8354.295/72000 seconds used.
- Launched exp052 CPU fallback probe v1 (1800s cap), now authoritatively RUNNING.
  Four real 1024px steps with same BF16/checkpoint trainer, save/reload and held-out
  inference. Remote source bytes and GPU=false/TPU=false verified by pull.
  This measures feasibility before deciding whether CPU training is worthwhile.
- Observer scripts/watch_kernel.py runs hidden (worker41376, launcher15928), never
  launches or replaces jobs; downloads terminal output to data/exp052_output.
  Logs: data/monitor/cpu-probe.log and cpu-probe.err. TPU monitor40132 remains live.
- Prior verifier artifacts retain union labels and no record/candidate mapping;
  they cannot be safely relabeled without a new harvest. No reranking run launched.

2026-09-09 00:44 Nairobi:
- exp052 still RUNNING; exp047 still QUEUED. Previous turn made progress by
  launching and remotely verifying the CPU fallback probe, without touching TPU job.
- Prepared CPU exp053 training and exp054 evaluation from identical trainer/eval
  payloads. Nothing full-length launched without successful probe evidence.
- Generalized scripts/monitor_training.py with isolated --cpu jobs/output/state/lock.
  Five monitor tests pass including backend mismatch, throughput, no duplicates,
  failed status observations, and CPU/TPU state separation.
- Replaced download-only CPU observer41376 with CPU pipeline monitor (launcher8104;
  actual worker PID in data/monitor/cpu-training-state.json). Old observer stopped
  deliberately; remote CPU job preserved. Existing TPU monitor40132 remains live.
- CPU gate requires four finite real1024px steps, exact code, BF16/checkpointing,
  reload+held-out forward and max last-two-step time<25s, leaving room for >1 epoch
  within the 10h trainer budget. On success advances exp053 then exp054; stops
  after downloaded evaluation for review. CPU logs cpu-training-monitor.log.

2026-09-09 06:23 Nairobi, results after interruption:
- exp052 BF16 CPU ERROR: first step1138.9s, timed out1680s during step2. Reject.
- exp055 FP32 CPU counterpart COMPLETE: 4 finite steps43.25/40.57/41.14/41.46s,
  losses5.575/5.003/4.973/4.077; remote report confirms reload+held-out forward.
  Too slow for one complete epoch within the current10h budget with reserve.
  No CPU full training launched. Local downloaded checkpoint.pt is0bytes: do not
  treat it as a usable checkpoint. Reports/events are intact; quality not evaluated.
- exp047v1 ERROR in first forward: grid_sample received BF16 values and FP32 grid.
  Added enable_fp32_sampling() around deformable attention with autocast disabled
  and values/locations/weights FP32. Keeps grid precision and gradients; rest BF16.
  12 tests pass, including mixed-dtype forward/gradient equivalence; four CPU smoke
  steps saved/reloaded and ran held-out forward (data/exp046_cpu_fp32_sampling).
- exp047v2 pushed with same1024px/four-step/5400s settings. Pulled remote source
  verifies GPU=false, TPU=true, exact corrected trainer/helper. Current QUEUED.
  Output isolation data/exp047_output_v2, full-train packaging updated to same code.
  Restarted TPU pipeline monitor worker1776; CPU pipeline remains deliberately stopped.
- Live logs require api.kernels_logs_stream(), not kernels_logs() which returns
  completed-session output and was empty while running. scripts/stream_kernel.py
  captures live logs after QUEUED ends to data/monitor/exp047-v2-started-live.jsonl;
  status pipeline still handles automatic launch/download. No remote job restarted
  solely from observation failures; v1's terminal dtype failure justified v2.
- TPU quota8481.890/72000 seconds used, reserved0 before v2. Public best still .38.

2026-09-09 06:28 Nairobi:
- exp047v2 remains authoritatively QUEUED; live-log collector9576 is waiting for
  execution. Previous turn made progress fixing and relaunching terminal TPU probe.
- Fixed output retrieval after discovering CLI charmap encoding errors and0-byte
  checkpoint downloads. scripts/download_kernel.py uses streamed HTTP with status,
  length and nonempty-artifact checks, atomic replacement, SHA256 manifest and UTF8
  logs. Monitor uses this downloader and requires verified output before advancing.
- Real integration recovered exp055 checkpoint:190110567 bytes, SHA256 matches,
  torch weights-only load succeeds, all47430154 tensor elements finite, steps4,size1024.
  Earlier warning about unusable exp055 local checkpoint is now resolved; this is
  still only a four-step feasibility model, not a quality result.
- Six monitor tests pass including rejecting empty/unverified artifacts. Restarted
  only local TPU pipeline worker to load download fix; remote queued job unchanged.
  New worker PID is in data/monitor/training-state.json. CPU pipelines stay stopped.

2026-09-09 07:03 Nairobi:
- exp047v2 now RUNNING. Live log verifies FP32 sampling active,27 checkpointed
  layers and first real optimizer step completed:661.80s, finite loss5.9337.
  First-step compile cost is not steady-state throughput. Waiting for steps3/4.
- exp056 CPU FP32 without activation checkpointing COMPLETE; same exp055 trainer
  payload, only checkpoint flag removed. Remote metadata/source verified CPU-only.
  Four steps35.93/34.32/32.37/31.93s, finite losses5.575/5.003/4.973/4.077;
  checkpoint reload+held-out inference passed. Strict output download completed.
  Output data/exp056_output; live log data/monitor/exp056-live.jsonl.
- Prepared exp053 CPU fallback from EXACT exp056 trainer payload, FP32/no checkpoint,
  with41400s trainer budget/43080s subprocess cap; would require43200s kernel cap.
  At measured speed with25% margin and20min reserve it can fit roughly990 steps
  (one epoch), unlike prior10h CPU plan. exp054 evaluator prepared, unchanged974-step
  minimum/full canonical validation. NOT launched: prefer TPU if warm probe passes.
- CPU monitor remains stopped and its prior BF16 gate is incompatible with this
  new fallback. Before any CPU launch, explicitly validate exact packaged trainer
  digest against exp056, update CPU gate/source/output paths and time budget, and
  retain objective-based validation/submission gates. Do not restart old CPU state.

2026-09-09 07:06 Nairobi — TPU feasibility passed, full training submitted:
- exp047v2 COMPLETE. Warm steps3/4:12.3224/12.3592s; all four losses finite,
  checkpoint reload and held-out inference passed. Strict downloader verified
  artifacts including190110567-byte checkpoint under data/exp047_output_v2.
- Monitor automatically pushed exp046v1 at07:05:30, currently QUEUED. Pulled remote
  metadata confirms TPU=true,GPU=false. Extracted trainer SHA256 matches successful
  probe report; helper bytes also identical.10h trainer budget,39600s kernel cap.
- Pipeline monitor worker7912 remains live; full-training live-log collector2116
  writes data/monitor/exp046-live.jsonl. Automatic next action after training is
  exp048 CPU full validation and conditional CSV, then download for submission review.
- Keep CPU fallback unlaunched while TPU route advances. No new submission;
  verified public best remains .38. Goal >.46 is still incomplete.

2026-09-09 14:06 Nairobi:
- Verified training is still RUNNING, not stalled: restarting only the local log
  collector exposed step1780 after a prior stream pause. Latest completed event is
  step1780 at11.7435s; next progress events are every20 steps. Remaining schedule
  is529 steps (target2309), roughly1.8h at observed throughput.
- Full-run quota check: TPU32445.500465s used,9622.309535s reserved at14:01;
  GPU quota untouched by this workflow. No RunPod operation performed.

2026-09-09 17:15 Nairobi:
- exp046 TPU training and exp048 CPU validation completed. The trained Mask2 model
  scored PQ0.3001760746 on the canonical106-photo validation and is rejected.
- Fine local D4 sweep over existing fused predictions found PQ0.4586368117 at
  conf0.385, erosion-1, min_area100 (852 TP,371 FP,473 FN), a +0.00156 gain over
  the previous0.4570759555 control. The corresponding test CSV is
  data/exp057_submission.csv:1188 rows,178 contributing images, and no mask
  overlaps. The Kaggle CLI was restored in the local venv; submission56125279 then
  completed with public score0.38. A TPU-only continuation, exp058, was pushed from the
  exp046 checkpoint (two additional epochs, 1024px, BF16+activation checkpointing,
  27000s cap, no GPU) and remains QUEUED; quota shows 9.08 TPU hours available.
  The watcher is running at data/exp058_output/watch-state.json.

2026-09-09 17:57 Nairobi, existing-weight audit:
- The two public 0.38 submissions exp037 and exp057 share exp002 weights; they
  differ in emission thresholds, not trained models. The +0.00156 local change
  was insufficient evidence for a meaningful public gain. No further training
  was launched in this audit; exp058 was already queued.
- No completed exp048/fusion result existed. The old monitor log stopped after
  standalone evaluation despite the newer on-disk fusion code. Launched the
  existing canonical Mask2/D4 fusion screen explicitly with the actual unpacked
  annotation path. Session49631; output data/exp048_output/fusion. It tests novel
  instances, agreement filtering, painting priority, matched intersection/union.
  All 106 photographs required; exact D4 control is checked before decision.json.
- Earlier explanations of undertraining/resolution as established root causes,
  and the 0.35-0.42 forecast, exceeded the evidence. They are hypotheses, not
  measured forecasts or isolated causal findings.

2026-09-09 19:06 Nairobi:
- Cached Mask2/D4 fusion completed at18:52:45, approximately55min after launch;
  the earlier10-20min remaining estimate was too optimistic. decision.json
  verifies unchanged D4 control PQ0.4570759555 and no improvement.
- Best fusion: novel Mask2 additions at IoU<=0.1, PQ0.4471750055;
  TP868/FP456/FN457 vs control823/323/502. Added45 matches but133 false positives.
  No submission generated from this screen. Public best remains0.38.
- Direct Kaggle status at19:06 still reports exp058 QUEUED. No new training
  launched during the existing-weight audit.

2026-09-09 19:23 Nairobi, user requested all three remaining existing-weight ideas:
- Implemented scripts/existing_weight_trials.py and launched CPU-only exp059
  soft D4, exp060 mask NMS, exp061 weight interpolation. All three returned
  RUNNING; pulled metadata confirms GPU=false and TPU=false. Live RESOLVED logs
  confirm native2048, original exp002 hash d9c6641e..., canonical106 photographs
  /180 annotation records, conf floor0.25 and training=false.
- exp059 matches identical hard-D4 seed clusters and score/support formula,
  compares probability averages at0.4/0.5/0.6 against regenerated hard voting.
  exp060 disables box suppression (IoU1) then tests mask NMS0.3/0.5/0.7;
  raw proposal cap1000 fails explicitly if saturated. exp061 interpolates
  exp002 with compatible exp031 frozen weights at alpha0/.25/.5/.75/1;
  fresh unfused model per alpha, floating state interpolated, integer buffers
  retained from baseline. Downloaded donor finite,711 matching states, SHA
  c08091f0df8c6b294baf1c4e9debc77c9ba022eff73ebd5cefb363e9558690aa.
- Native streamed-mask decoder matched official decoding for both CPU cropping
  branches. Real photograph20110220082634Lh.jpeg matched native and historical
  masks exactly (minIoU1.0);36 raw proposals,4 kept after mask NMS0.5.
  Fixed Ultralytics grayscale HxWx1 reader handling before launching.
- Every experiment evaluates full canonical validation with unchanged controls,
  confidence0.30/.35/.385/.40/.45/.50 and erosion0/-1, minarea300. Historical
  control mismatch >0.001 invalidates the result. No new submission yet.
- Background monitor scripts/monitor_existing_weight_trials.py worker13760
  (wrapper10316), state data/existing_weight_monitor.json, checks every180s,
  downloads terminal output, records best/control scores, NEVER relaunches or
  submits. Live logs data/exp059_output/live.jsonl, exp060_output/live.jsonl,
  exp061_output/live.jsonl. Job manifest data/existing_weight_jobs.json.
- Previously queued exp058 TPU continuation now RUNNING, independently of
  these three inference jobs. No new training or RunPod operation this turn.

2026-09-09 22:10 Nairobi, exp060/061 control-check recovery:
- Both exp060 mask NMS and exp061 weight interpolation completed ALL106-photo
  inference and scoring, then raised at the stale control scalar. Saved output
  is intact, including every candidate cache and report. Original notebooks
  remain ERROR; do not relaunch expensive inference merely to clear that label.
- Root cause: new identity proposal floor0.25 was checked against a historical
  scalar from floor0.05. Ultralytics CPU crop_mask changes round-versus-ceil
  behavior at50 detections. Exactly four photographs crossed that boundary
  (58->27,54->23,52->26,59->24). Official native inference reproduced both
  historical and new masks exactly on all four; CPU score differences across
  Windows/Linux were <1.8e-6. All retained scores in the two saved106-photo
  caches matched exactly; other photographs had identical masks.
- Native verification script scripts/audit_existing_weight_control.py writes
  data/existing_weight_control_audit.json after pooled PQ replay. Recovery
  script scripts/recover_existing_weight_trials.py requires that audit to pass
  and both independent controls to match before creating recovered_report.json.
  Read data/existing_weight_recovery.json for the recovered decision summary;
  original report.json and original failure remain preserved for provenance.
- Mask NMS best0.44528791295620856 atIoU0.3/conf0.35/erosion-1/minarea300,
  versus matched single-view control0.44145370603652173. This remains below
  existing D4 best0.4586368117. Weight interpolation alpha.25/.5/.75/1 best PQ
  .43490545/.43267268/.43202969/.43242007, all worse than its unchanged control.
  No new submission. Soft D4 still RUNNING (latest observed69/106).
- Local source now references the correct floor0.25 scalar, with tighter1e-6
  tolerance for identity; running remote exp059 was not changed. Monitor was
  restarted (wrapper15688) with support for recovered reports and persistent
  downloaded state. It will not repeat completed jobs or their downloads.
- Recovery completed successfully: audit passed, pooled historical cache
  PQ0.4410584549910743 and new controlPQ0.44145370603652173 verified. Both
  recovered_report.json files and data/existing_weight_recovery.json saved.

2026-09-10 00:08 Nairobi, all three existing-weight screens finished:
- Direct Kaggle status confirms exp059 COMPLETE. Monitor downloaded outputs
  at23:49:56; results/report.json SHA25665755013ea064731aa26c700e8e211f7ff62b2f02ea295aa1ee1c03884eeb1a7.
  Full106 photographs,15922.65s total. Fixed hard control0.4565275605 was within
  the preregistered0.001 tolerance of historical0.4570759555.
- Matched sweep best hard control0.4585190942; soft thresholds0.4/0.5/0.6
  scored0.4568985112/0.4551735397/0.4474300284. Soft averaging loses. Best soft
  and hard use conf0.385,erosion-1,minarea300; soft has identical841 TP but
  six more FP (354 versus348) and slightly lower SQ.
- All three requested methods have now been evaluated. No tested configuration
  beats existing best validation0.4586368117. No new CSV/submission justified;
  verified public best remains0.38. Direct status exp058 TPU continuation still
  RUNNING; watcher state refreshed00:07:13. Goal >0.46 public is not achieved.

2026-09-13 21:12 Nairobi, explicit user request to submit what's on hand:
- Submitted data/exp062_d4_recall.csv after complete-file schema, identifier,
  test-stem and overlap validation. D4 link.5/vote.5/conf.35/erosion-1/minarea300,
  1233 instances over177 contributing test images (all180 processed). SHA256
  11fbeb79be12bcafc7fa7aef4a2b34c84cbd3053ea12952ed832aa6ec3643fdd.
  Relative to exp057:82 new masks,37 old-only masks. Submission56212792 accepted,
  initially PENDING; four submission slots remain. No new public score yet.
- exp063 submission probes v1(defaultP100) failed during GPU warmup: current
  Kaggle CUDA build lacks a P100-compatible kernel image. No prediction emitted.
  Retried SAME slug asv2 with explicit --accelerator NvidiaTeslaT4 and14400s cap;
  pulled metadata confirms NvidiaTeslaT4/GPU=true/TPU=false. Direct status RUNNING.
  Initial log stream500 is an observation error, NOT evidence job failed.
- v2 produces three existing-weight test variants using identical validated
  CPU crop-rounding rules, FP32 GPU inference, and four complex validation-photo
  parity checks per variant; no new training. Then evaluates exp058 checkpoint
  on106 canonical validation photos and generates its best-rule test CSV using
  --force-submission (user explicitly overrides local-score submission gate).
- Automatic authorized uploader: scripts/submit_authorized_probes.py, worker18008
  (wrapper13776), state data/exp063_submission_state_v2.json; logs
  data/exp063_submit_v2.log/.err. Watches only exp063, downloads to
  data/exp063_v2_output, checks hashes/schema/IDs/test membership/no overlap,
  submits each of the four named probes once and reads public scores back.
  Exact descriptions reconcile retries; an ambiguous upload is never blindly
  repeated. Does NOT launch or restart compute. Live log collector wrapper32464
  writes data/exp063_v2_output/live.jsonl. v1 outputs preserved separately.
- User states30 GPU hours; quota SDK unexpectedly returned total21600s(6h)
  before launch while TPU72000s(20h). Treat discrepancy as unresolved, not as
  a reason to ignore user's budget or T4 instruction. No new training launched.
- v2 T4 inference now confirmed progressing: mask-NMS parity passed on four
  complex held-out photographs, all maskIoU1.0, maximum score delta<1.91e-6.
  First10/180 test photos processed at45.8s after module start. First submission
  status is also saved in data/exp062_submission_result.json.

2026-09-13 21:16 Nairobi, first authorized public probe improved the score:
- Kaggle submission56212792 is COMPLETE with public score0.39, verified through
  the authenticated submission API and saved to data/exp062_submission_result.json.
  This supersedes the0.38 best. Same original exp002 weights; lower D4 emission
  confidence0.35 plus minarea300 versus exp057 conf0.385/minarea100. Because
  two settings differ, do not attribute the gain solely to confidence as proven.
- Public result changes prioritization: local threshold ordering is not enough
  to reject nearby emission rules. A further lower-confidence D4 probe is a
  reasonable follow-up once today's four already-authorized variant submissions
  complete. Do not consume their four reserved submission slots implicitly.
- exp063v2 remains RUNNING on T4; native test inference observed atNMS60/180,
  182.4s. Await its four controlled public probes and exp058 full validation
  before choosing another training schedule. Goal >0.46 is still incomplete.

2026-09-13 21:29 Nairobi, host-guided visual review and follow-up:
- Completed native-image review of exp057 versus exp062. Full RLE comparison
  finds82 added and37 removed instances, zero matching-mask boundary changes.
  Six deterministic size-stratified examples reviewed; details and limitations
  in docs/exp062-quality-review.md, panels in data/exp062_quality_review.
- Added curved structure20120402105814Th has clear image support. Removed299px
  structure20210127110130Ch is also visibly coherent and filament-like, so the
  area300 cutoff is NOT promoted as a quality improvement merely from0.39 PQ.
- Preparing exp065_d4_small_masks.csv atconf.35/minarea100 (same D4/erosion),
  which isolates the confidence change against exp057 and preserves small masks.
  Local builder session93884; logs data/exp065_build.log/.err. NOT submitted;
  today's four remaining slots stay reserved for the authorized exp063 probes.
  exp064_d4_recall.csv(conf.30/minarea300) prepared1364 rows, lower priority now.
- Latest direct status: exp063 RUNNING, soft20/180 observed; NMS and weight-blend
  CSVs complete and overlap-checked. No submission currently being scored; last
  accepted result56212792 remains0.39. Automatic upload waits for kernel output.

2026-09-14 08:20 Nairobi, resumed after user usage limit:
- Direct API confirms exp063 COMPLETE. All four automatic submissions completed:
  NMS56213729=0.37, weights56213786=0.36, soft56213830=0.38,
  Mask2 continuation56213874=0.20. State data/exp063_submission_state_v2.json.
  Historical Unicode logging error did not prevent submission/readback.
- exp058 continuation full106-photo validation is0.2467558223 (485TP,644FP,840FN),
  worse than exp046's0.300176. Do not blindly extend that training. Checkpoint
  hash6c941d0717a881d2299f08cc39e2b2ea94891fe985027c8668b1a5e06b8a2974.
- Today's allowance reset to5. Submitted exp065 after complete-file overlap,
  schema and test-name checks:1278 instances,178 contributing images, all180
  processed. Preserves all1233 exp062 masks and adds45 small masks. SHA256
  6496f25a115d2fd472340e06ec0bbb490e4ba2e96161c64c7e2e7418133dc94c.
  Ref56222273 initially PENDING; four slots remain. No duplicate submissions.
- New bounded experiment exp066 uses pretrained sam2.1_b.pt to refine cropped
  existing D4 instances, comparing box versus box+three interior point prompts.
  Confidence scores/detections are retained, with explicit seed-IoU/area guards
  falling back to original masks. No training or test annotations. Canonical
  24-photo screen selected without labels; >0.005 gain gates full106 evaluation.
  Both raw and eroded refined masks scored, minarea100/conf.35 fixed. Save raw
  caches and per-instance changes for required visual review.
- Local crop/prompt assertions passed on12 real masks. Pushed
  cosmicwitness/sol-exp066-sam2-boundary-probe v1 with explicit NvidiaTeslaT4,
  7800s kernel cap; pulled metadata verifies T4/GPU=true/TPU=false. Live inference
  observed at9/24 screen photographs. Watcher worker2612(wrapper23204) downloads
  terminal artifacts to data/exp066_output; live stream wrapper22012. No auto
  submission for SAM2. Source: scripts/sam2_boundary_probe.py and package_sam2_probe.py.
- watch_kernel now uses authenticated SDK status and sys.executable for the
  downloader, removing dependence on an ambiguous bare python/kaggle PATH.
- exp06556222273 subsequently COMPLETE with public0.38, saved in
  data/exp065_submission_result.json. Best remains exp062 at0.39. Preserving
  those45 small instances did not improve public PQ; visual plausibility alone
  does not establish their test labels or justify calling all removals errors.

2026-09-15 01:02 Nairobi, resumed after usage interruption:
- Direct authenticated Kaggle status confirms `cosmicwitness/sol-exp073-roi-continuation` is RUNNING on the explicit T4 configuration. The ROI continuation starts from the verified exp068 best checkpoint; live output shows finite loss through epoch 1 and epoch 2 has started.
- Exactly one background watcher (PID 26108) and one UTF-8 log streamer (PID 6792) are attached. The watcher polls status every 60 seconds and downloads terminal artifacts to `data/exp073_v2_output`; the stream persists `live.jsonl`.
- No submission is being made while the continuation is running. After download, evaluate the checkpoint against the canonical validation and only then build a schema-checked submission.
2026-09-15 01:07 Nairobi, monitor recovery:
- The authoritative job remains RUNNING; live logs now show epoch 2/4 completed with finite losses. A transient SSE chunk error was recovered by restarting only the log streamer; exactly one watcher (PID 26108) and one streamer (PID 31108) are active. No training job was restarted.

2026-09-15 06:23 Nairobi, exp073 ROI continuation evaluation:
- exp073 completed all four bounded ROI-loss continuation epochs from exp068's verified native checkpoint. Its official Ultralytics mask mAP50-95 was 0.206; training output includes an immutable `best.pt`.
- T4 evaluation kernel `cosmicwitness/sol-exp074-roi-validation` measured canonical identity PQ 0.43636273095914957 and D4 PQ 0.4554893230860615 at conf0.25/grow0/minarea300 (798 TP, 285 FP, 527 FN). D4 improves exp068's 0.44696401267887576, but remains below the retained validation best 0.4586368117.
- Do not create a test CSV or submission from exp073. Preserve its outputs and use the result as evidence that the ROI loss is viable but this short low-rate continuation is not the promotion candidate.
- exp074 terminal report and both 106-photo prediction caches are downloaded at `data/exp074_output/validation`; watcher reports COMPLETE/downloaded=true. Next CPU-only experiment exp075 reuses the existing boundary-transfer evaluator with exp074 as donor and the original D4 candidates as baseline. It compares donor, intersection and union masks for matched detections, holding emission fixed and requiring the unchanged baseline PQ check to pass. Output `data/exp075_roi_boundary_transfer`, logs `data/exp075.log` and `data/exp075.err`. No additional GPU or training is needed for this experiment.

2026-09-15 06:53 Nairobi, shift from more continuation training to complementary discovery:
- exp075 completed. Baseline/native/intersection/union PQ are 0.4558638687 / 0.4529408772 / 0.4532713931 / 0.4538087042. All 640 matched-boundary substitutions lose; reject the entire boundary-transfer family.
- A label-assisted instance audit found 465 baseline misses: 284 have zero overlap with any accepted prediction, 181 have some overlap, and only 13 are recoverable by joining a pair of fragments. The immediate bottleneck is missed-object discovery, not boundary replacement or graph joining.
- A newly published solar-filament workflow (Hu et al., arXiv:2607.24525) supports a complementary semantic route: multiscale, local-contrast segmentation first, image-driven detail refinement second, instance integration last. Its ablation attributes the largest changes to 5x5 kernels, instance normalization and hierarchical multiscale pooling. This is stronger evidence for a separate semantic proposal stream than for another short YOLO continuation.
- exp076 now audits a cheaper first step: retain the pretrained D4 detections, then rescue only weak pretrained proposals that also have dark local contrast, dark-pixel coverage and elongated shape. The 512px feature pass is followed by an approximate candidate-utility screen and exact canonical PQ on only 32 distinct rules. Worker PID 25844 and monitor PID 31280; durable state `data/exp076_v7-monitor.json`, terminal report `data/exp076_physical_rescue_v7/report.json`. No GPU or submission is involved.

## 2026-09-15 interruption recovery and semantic complement

- Kaggle's account UI is authoritative for accelerator balance: the supplied screenshot showed 24 h 13 m GPU remaining out of 30 h and the full 20 h TPU balance. The earlier interpretation of `quota_view()` as only 13.8 minutes remaining was wrong.
- exp076 completed after checkpointed resume. All 32 exact global physical-rescue rules underperformed the unchanged baseline: best PQ 0.4546324363 versus baseline 0.4558638687. Although the best rule raised TP from 860 to 882, FP rose from 397 to 446. This confirms that admitting visually plausible weak proposals without sufficiently selective independent evidence loses under the competition's loose multi-annotator scoring convention. Do not promote exp076.
- The queued TPU version of the frozen semantic-complement validation was removed from the critical path because TPU allocation was not starting. The same no-training experiment is running as T4 kernel `cosmicwitness/sol-exp078-semantic-complement-t4`. It uses the existing exp004 U-Net only to support weak D4 proposals or add non-overlapping semantic components, with canonical validation before any submission.
- The full semantic-component branch in exp078 stalled after image 60 and was deleted. A support-only, exact-shortlist rerun completed as `cosmicwitness/sol-exp081-semantic-support-shortlist-t4`: best canonical PQ 0.4593728336 at D4 confidence 0.26 plus frozen exp004 semantic mean >=0.80, versus the unchanged 0.4558638687 baseline. TP/FP/FN changed from 860/397/465 to 891/438/434. This is a small validated gain using existing weights only.
- Test inference is running as `cosmicwitness/sol-exp082-semantic-support-submission-t4`. It reuses the two completed exp037 D4 test-proposal kernels and exp004 checkpoint; no training or YOLO inference is repeated. Output must pass overlap/schema checks and native-mask review before submission.
- exp082 completed with 1310 instances over 177 contributing test images. It adds 77 masks to exp062 across 56 photographs with no removals or boundary changes. Full schema/identifier/overlap checks passed, and deterministic native review of the largest, median, and smallest additions found visible dark filament support in all three. CSV SHA256 `9C07C4217C36255260E3D97C3C95663E43D4F6452DC9CE052428F05357ED2A54`. Submission ref 56250449 was accepted and is PENDING; background result file `data/exp082_submission_result.json`.
- exp082 public score is 0.38, below exp062's 0.39 despite canonical validation improving from 0.45586 to 0.45937. Together with exp064 and exp065, public evidence consistently penalizes extra low-confidence or small masks. Do not promote proposal-rescue-by-addition. Shift the next public probe toward conservative pruning and the main experiment toward a consensus-aware proposal-quality head.
- exp083 reconstructed the retained D4 baseline exactly and audited view-count and mean-confidence pruning. Every pruning rule reduced canonical PQ; most retained instances already have support from all eight D4 views, so transform-consensus does not separate useful from harmful predictions.
- exp084's area-350 candidate would remove 26 instances from exp062 and reduced canonical PQ to 0.45271518. Native review of size-stratified removals at areas 300, 327 and 349 showed clear filament structure in all three. It was rejected before submission under the host's prediction-quality guidance.
- exp085 is the next independent model family: a compact MIT-licensed adaptation of MORDEN's 5x5 convolutions, instance normalization, direct multiscale image branches, channel gates and pyramid context, with an added spine head trained from the provided physical centerlines. It uses three solar channels (raw, CLAHE and dark local contrast), the unchanged grouped canonical split, per-annotation targets, focal/Dice losses, and epoch checkpoints. Kernel `cosmicwitness/sol-exp085-morden-spine-t4` version 1 is RUNNING on a read-back `NvidiaTeslaT4`; GPU=true, TPU=false. Watcher PID 18072 downloads terminal output to `data/exp085_output`; streamer PID 30144 persists `data/exp085_live.jsonl`. The bounded trainer has a 4.25-hour budget, followed by standalone canonical PQ evaluation. Existing pretrained D4 proposals remain the later instance/proposal anchor if the semantic-spine field is complementary.
- exp085 completed 10 epochs in 25.5 training minutes and downloaded cleanly. Peak allocation was only 0.981 GiB. The best epoch reached validation mask Dice 0.70303 and spine Dice 0.63473, proving the solar-specific field learned, but standalone connected-component PQ was only 0.32270 (706 TP, 760 FP, 619 FN). Do not submit standalone semantics: instance decomposition, rather than pixel recognition, is the failure.
- exp086 is the intended pretrained-centered audit. It holds the exact 0.4558638687 D4 emission as the instance anchor and tests exp085 only as independent evidence for pruning/weak-proposal support or as a seeded semantic boundary field. Unseeded semantic components can never enter. Local worker PID 27496, persistent state `data/exp086_monitor.json`, terminal report `data/exp086_morden_fusion/report.json`.

2026-09-15 after four-hour usage interruption:
- The interrupted exp076 worker had reached 8/32 exact rules but had not persisted its feature table. The evaluator now atomically saves `feature_stage.json` and `exact_progress.json`, accepts the same output directory, and resumes completed stages/rules. The resumed v7 worker uses PID 26072 with monitor PID 16032. Its feature checkpoint is already complete. The proposal oracle is material: 168 of 465 missed truth instances have a low-confidence D4 proposal above IoU 0.5, although the first simple physical rules remain below baseline.
- Recovered the frozen converged exp004 1024px semantic U-Net checkpoint from completed Kaggle output. `best.pt` is 97,891,199 bytes under `data/exp004_output/checkpoints`; the resumable downloader also verified `last.pt` and the old submission.
- Kaggle quota readback is GPU 20,773.727/21,600 seconds used (826 seconds left) and TPU 0/72,000 used, refresh 2026-09-19. Preserve the remaining T4 minutes.
- Launched `cosmicwitness/sol-exp077-semantic-complement` v1 on TPU, currently QUEUED. Remote pulled metadata confirms TPU=true, GPU=false, Tpu1VmV38, with frozen exp004 and exp066 kernel outputs mounted. It evaluates the dense model as independent support for weak D4 proposals and as a source of non-overlapping semantic candidates; no training and no test annotations. Watcher PID 29228 and streamer PID 24844 download to `data/exp077_output`.

## 2026-09-15 17:20-18:10 Nairobi, exp088 headroom decomposition and exp086 outcome

Process error first: a duplicate exp087 was launched because this session
trusted the handoff's "not launched" claim, which had queried the wrong slug.
Kaggle derives the slug from the title, so the previous agent's push had landed
on `sol-exp087-morden-continuation-t4`. The duplicate ran about ten T4 minutes
and the user cancelled it. `CLAUDE.md` section 14 records the rule.

### exp088: where the gap actually is

`scripts/audit_headroom.py` reproduces the exact 0.4558638687 baseline and
decomposes it. All numbers below are canonical pooled PQ on the 106-photograph,
180-record validation split.

- Baseline SQ 0.6843, RQ 0.6662, TP/FP/FN 860/397/465.
- Matched IoU clusters just above the threshold: 153 in 0.5-0.6, 318 in 0.6-0.7,
  320 in 0.7-0.8, only 69 above 0.8.
- Of 465 missed truth records, 284 have zero overlap with any prediction, but
  129 already overlap one at IoU 0.3-0.5 and 66 of those are in 0.4-0.5.
- The false positives mirror them: 127 of 397 sit at IoU 0.3-0.5 against truth.
  Many FP and FN pairs are the same object failing the 0.5 threshold.
- Selection oracle: taking the best available mask per truth record from the
  entire weak D4 pool, with today's false-positive burden unchanged, gives PQ
  **0.4932**. The pool's best mask per truth averages only 0.665 IoU.

`scripts/audit_annotator_agreement.py` and `scripts/audit_consensus_ceiling.py`
calibrate what boundary quality is reachable. Two annotators drawing the same
filament agree at mean IoU 0.636 over 359 matched pairs, but the best single
mask scored against every record of its group reaches **0.809** (pixel majority;
best individual annotator mask 0.801). On those same multiply annotated
photographs our masks reach SQ 0.680. So 0.68 to roughly 0.78-0.81 is measured
headroom, not annotation noise.

Consequence for planning: a proposal-quality head over the existing pool is
capped near 0.4932 and cannot by itself reach the score the goal needs. That
does not make it worthless, but segmentation quality is the larger unclaimed
term and has never been attacked with a model trained on the per-record
question. exp066 used generic SAM2 and exp075 substituted a donor model's whole
mask; neither learned the annotators' boundary convention.

`scripts/audit_boundary_bias.py` rules out the cheap version of that fix. On
matched pairs precision is 0.817, recall 0.826 and the area ratio is 1.041,
so the masks are very slightly large but essentially unbiased. Uniform
morphology is destructive: one dilation gives PQ 0.4286 and one further erosion
0.4176. The boundary error is local shape, not a global size offset.

### exp086: exp085 semantics as pruning evidence

The audit completed. Holding the exact D4 emission as the instance anchor:

- Best prune: `semantic_p75 >= 0.15` gives PQ 0.4593462, TP/FP/FN 836/332/489.
- Best support (additions): 0.4574850. Best D4-seeded boundary: 0.4409271.
- Boundary substitution loses again, consistent with exp075.

So of the three roles exp085 can play, only pruning helps.

### exp088 follow-up: limb-aware pruning

Native review of the 58 instances the flat rule removes
(`data/exp088_headroom/removal_review/removed_instances.png`) shows the errors
are spatial. It correctly deletes limb-edge artifacts, including a 4924-pixel
mask tracing the disk edge with zero annotator overlap. It wrongly deletes
genuine dark filaments near the limb and in bright plage, where limb darkening
depresses the semantic probability: three removals have ground-truth IoU 0.78,
0.76 and 0.74.

`scripts/sweep_radial_prune.py` therefore lets the threshold vary with the
instance centroid's normalised radius. The IoU table is computed once and every
rule rescored over the retained rows, because pruning never repaints a mask.

| rule | retained | PQ | TP | FP | FN | removals real |
|---|---|---|---|---|---|---|
| baseline | 770 | 0.4558639 | 860 | 397 | 465 | - |
| flat 0.15 | 712 | 0.4593462 | 836 | 332 | 489 | 20 of 58 |
| flat 0.12 | 716 | 0.4597150 | 839 | 337 | 486 | 18 of 54 |
| radial 0.65 / 0.30 / 0.12 | 710 | **0.4606350** | 838 | 328 | 487 | 19 of 60 |
| radial 0.60 / 0.60 / 0.12 | 698 | 0.4607940 | 831 | 314 | 494 | 25 of 72 |

Photograph-level bootstrap, 500 resamples: pruning beats the baseline in 93.4%
(flat 0.12) and 95.2% (radial 0.60) of draws. The radial refinement beats flat
0.12 in only 76-88% of draws, so its extra ~0.001 is not robust; the robust
claim is that semantic pruning helps by about +0.004.

Deployed rule: `semantic_p75 >= 0.30` inside radius 0.65, `>= 0.12` outside.
It has the best removal precision of the family (0.68), a stated physical
reason, and validated PQ 0.4606350484956118.

Why this is worth a public probe even though it is small: every public loss so
far came from adding masks (exp064, exp065, exp082), and the one public gain
came from emitting fewer, higher-confidence masks (exp062 over exp057). Pruning
has never been tested publicly, because every earlier pruning rule lost locally.
This one gains locally with independent evidence from a separately trained
solar-specific model.

### Live compute

- `cosmicwitness/sol-exp087-morden-continuation-t4` RUNNING, epoch 25/50 at
  1.107 h, mask Dice around 0.701. Watcher PID 2248, streamer PID 14196.
- `cosmicwitness/sol-exp089-prune-features-t4` RUNNING. It scores the 1233
  exp062 test instances with the exp085 checkpoint and, before that, recomputes
  the 770 stored validation features from the same checkpoint and refuses to
  continue unless they reproduce within 0.02. Watcher PID 28072, streamer
  PID 13680. Output `data/exp089_output`.
- Deployment: `scripts/build_pruned_submission.py` replays the identical rule
  over the stored validation features and asserts PQ 0.4606350484956118 before
  it writes any CSV.

## 2026-09-15 19:35 Nairobi, exp087 continuation complete

`cosmicwitness/sol-exp087-morden-continuation-t4` finished 50 epochs in about
2.2 hours on one T4. Best checkpoint is epoch 46 at
`data/exp087_output/training/best.pt`, SHA256
`27c21ea63a6163260a179c70a1e4f72b82fb6aebe63498bb08850c1435ec563a`.

| metric | exp085 (10 epochs) | exp087 (50 more) |
|---|---|---|
| validation mask Dice | 0.703035 | **0.713350** |
| validation spine Dice | 0.634728 | **0.673546** |

The spine head gained most. Both heads improved with no divergence, so the
longer schedule was worth it. This checkpoint is NOT yet wired into any
deployed rule. The exp090 pruning rule was tuned against exp085 probabilities
and deployed against exp085 probabilities, and that pairing must stay intact
unless the rule is retuned end to end.

A local footprint note, after the machine ran low on memory: only three small
Kaggle pollers run locally, about 140 MB in total. The 318 MB exp092 validation
crop cache was deleted because the Kaggle kernel rebuilds it.

## 2026-09-15 20:05 Nairobi, exp092 refiner running; exp093 deploy prepared

`cosmicwitness/sol-exp092-refiner-t4` is RUNNING on a read-back T4 (GPU true,
TPU false). Watcher PID 4880 downloads to `data/exp092_output`; streamer PID
17112 appends `data/exp092_live.jsonl`.

- exp091 harvest completed: 601/601 training photographs, 7442 proposals,
  parity 1.0 against the stored validation pool.
- Train crop cache: 7817 (proposal, record) pairs, mean seed IoU 0.5791.
- Validation crop cache: 1017 pairs, mean seed IoU 0.6367181341128024,
  bit-identical to the local build.
- Epochs take about 1.5 minutes. After 5 of 30, best crop IoU is 0.64849 at
  epoch 3 against seed 0.63672 (+0.0118), pushing 48 pairs above 0.5 and 13
  below. This is a checkpoint-selection proxy only. Promotion is decided by the
  exact canonical PQ sweep that runs at the end of the kernel.

Deployment is prepared as exp093 and must not be pushed until exp092's report
exists. `scripts/package_refiner_deploy.py` takes `--threshold`, `--guard` and
`--expected-pq` from `data/exp092_output/validation/report.json`. The kernel
refines the validation emission first and asserts that PQ before it touches
the test CSV, then asserts row count and pixel-disjointness per image.

Ordering check, read from code: `shared.utils.paint_panoptic` sorts candidates
by descending score and returns them in painting order, and both CSV writers
(`scripts/build_submission_from_views.py`, `scripts/existing_weight_test.py`)
number rows `_1, _2, ...` in that order. So `deploy.load_submission`, which
assigns scores by row position, reproduces the true claim order.

Gate before any exp093 submission: exact PQ above 0.4558639, native review of
changed boundaries, `scripts/submit.py --dry-run` overlap and schema checks.

## 2026-09-15 21:17 Nairobi, exp092 result: refiner does not move canonical PQ

`cosmicwitness/sol-exp092-refiner-t4` COMPLETE. 30 epochs; best checkpoint is
epoch 20 (crop IoU 0.64967 against seed 0.63672). The kernel's unchanged
baseline check reproduced 0.45586386872921264 exactly.

Exact canonical PQ sweep over threshold x guard (30 rules), best rows:

| threshold | guard | PQ | delta | SQ | TP | FP | FN |
|---|---|---|---|---|---|---|---|
| 0.40 | 0.50 | 0.4565395 | +0.00068 | 0.6843 | 861 | 395 | 464 |
| 0.40 | 0.40 | 0.4565012 | +0.00064 | 0.6842 | 861 | 395 | 464 |
| 0.45 | 0.50 | 0.4561169 | +0.00025 | 0.6836 | 861 | 395 | 464 |
| 0.40 | 0.00 | 0.4560121 | +0.00015 | 0.6843 | 860 | 396 | 465 |

Worst rule: threshold 0.60, guard 0.30, PQ 0.449375.

Decision: NOT promoted, exp093 deploy NOT pushed, no submission spent. A
+0.0007 local change is one true positive and far below what the two-decimal
leaderboard can show (`CLAUDE.md` section 11).

Contradiction to explain: at crop level 52 validation pairs crossed IoU 0.5
and 12 fell below, yet native PQ moved by one TP and SQ did not move. Being
measured: whether the crop target truncates annotator masks that extend past
the proposal-sized window, which would let crop IoU rise while native IoU
cannot. Result in `data/exp092_analysis/truncation.json`.

## 2026-09-15 evening: leaderboard reality check and change of direction

User rule, effective now: download nothing to the local machine and spend no
local bandwidth. Every job runs on Kaggle. Kernel outputs stay on Kaggle; only a
compact summary printed to the kernel log is read back. Local watchers that
download outputs (`scripts/watch_kernel.py`) must not be used.

Public leaderboard read via the Kaggle CLI: top **0.56**, about twenty teams at
0.55-0.56. We sit at 0.39.

- `artkomissar/please-fix-the-leaderboard` (responsible disclosure, July) showed
  the evaluator of that time gave 1.00 to five empty masks, 0.88 to one empty
  mask per image and 0.70 to seven. Some public notebooks (for example the
  "LB 1st" U-Net baseline) write an empty mask for every image without a
  detection. Scores from before the fix are not model evidence. We will not use
  empty masks or any evaluator defect.
- The host self-evaluation notebook now computes pooled per-annotator-image PQ
  at IoU > 0.5; predictions for images without annotation rows are not visited.
  This matches `shared.utils.aggregate_pq`.
- HDJoJo, at 0.55, publishes inference for a YOLOv8l-seg trained at imgsz 2048:
  single view, conf 0.30, NMS IoU 0.0, default non-retina masks, no erosion,
  area floor 5, and overlap removal that effectively claims the lowest-confidence
  mask first. CORRECTED by reading `data/checkpoints/exp002/outputs/args.yaml`: exp002 is yolo11m-seg (medium) trained at imgsz 1280 with optimizer auto, batch 4, mask_ratio 4, time 8.5 h. It is not YOLOv8l-seg. HDJoJo's model differs in both size (large) and training resolution (2048).
  Their training data is unknown; the public MAGFiLO release contains test
  labels, so a leaked-training explanation cannot be excluded.
- Fususu's "LB 0.70" notebook (Mask R-CNN at 1024 plus a ResNet18 256-crop
  U-Net refiner with topology-safe flip TTA) dates from 5 August, before the
  evaluator fix, so its 0.70 is not comparable.

Decision: stop the exp092 refiner line (exact PQ +0.00068, not promoted) and
first measure how much of the 0.39 vs 0.55 gap is inference configuration alone.
exp095 (`scripts/probe_yolo_inference_config.py`, packager
`scripts/package_yolo_inference_probe.py`) runs our frozen exp002 weights on the
canonical validation split under six (NMS IoU x retina) configurations and
scores confidence, erosion, area floor and paint-order rules with exact PQ.
Parity rule reproduces exp033's single-view 0.4403668270817509 (NMS 0.60,
retina, conf 0.35, one erosion, min area 300; tolerance 0.001 as in exp033).

Literature notes (user supplied Reche & Cid 2026, Acta Astronautica; SABR, VisionX,
EdgeAttNet, solar-seg READMEs):
- Reche & Cid: DETR detection plus a per-instance 256 crop U-Net with 3x
  context and a box channel; MAGFiLO per-instance IoU 0.70 / Dice 0.81. That is
  close to our matched SQ 0.684, so their gain is not boundary quality. Their
  preprocessing (radial limb-darkening correction, histogram matching inside the
  disk to a fixed three-peak reference, zeroed off-disk) is a candidate for a
  future detector retrain. They found MAGFiLO's multiple annotations per
  filament hurt detector training.
- srivatsav-kannan/solar-seg: width-24 semantic U-Net, holdout PQ 0.346, public
  0.31. A plain semantic U-Net is far below the 0.55 cluster.

exp094 (refiner diagnostic) completed on Kaggle; its summary has not been read.
Local watcher PIDs 4880, 17112 and 27420 were stopped under the new rule.

### exp094 result: the exp092 crop gain was a resampling artifact

Read from the kernel log only (184 KB; future kernels print a single summary).
Over 1017 validation (proposal, record) pairs, mean IoU relative to the native
seed IoU 0.63672:

| stage | mean delta | crossed up / down at 0.5 |
|---|---|---|
| unrefined seed scored at crop resolution | +0.01165 | 35 / 4 |
| refined, crop resolution (training monitor) | +0.01295 | 52 / 12 |
| refined, pasted to native | -0.00228 | 17 / 18 |
| refined, native, TTA | -0.00130 | 17 / 18 |
| refined, native, TTA, threshold 0.40 | +0.00093 | 17 / 14 |

Nearest-neighbour resampling to 256 inflates IoU with no model involved. The
refiner's own contribution was about +0.001, matching the exact-PQ result of
one extra TP. Two earlier explanations (window truncation, fragments of matched
truth) were measured and refuted. Lesson for any crop model: monitor IoU at
native resolution, never at crop resolution against a native baseline.

### exp095 result: the leaderboard gap is the weights, not the inference recipe

`cosmicwitness/sol-exp095-yolo-inference-probe-t4`, Ultralytics 8.4.153, frozen
exp002 weights (SHA256 d9c6641e...), canonical validation, exact pooled PQ. Log
read only (25.6 KB). Parity reproduced exp033's single view: 0.4403667 against
0.4403668.

| rule | PQ | TP | FP | FN |
|---|---|---|---|---|
| exp033 single view (NMS 0.6, retina, conf 0.35, erode, area 300) | 0.44037 | 845 | 456 | 480 |
| HDJoJo's exact recipe (NMS 0.0, non-retina, conf 0.30, no erosion, area 5, lowest-confidence-first paint) | **0.41114** | 835 | 565 | 490 |
| best single view: NMS 0.3, conf 0.35, erode, area 300, lowest-confidence-first paint | 0.44689 | 846 | 419 | 479 |
| best at NMS 0.0 | 0.44194 | 823 | 405 | 502 |
| current D4 emission | 0.45586 | 860 | 397 | 465 |

Retina and non-retina masks gave identical PQ in every configuration. The public
0.55 team's recipe is worse than ours on our weights, so its advantage lies in
its trained model (training resolution, data, or undisclosed training data).
NMS 0.3 helps single view by +0.0065, which is below what the leaderboard can
show; no submission. The next lever is detector training, not postprocessing.

Also: `kaggle kernels output` on Windows must run with `PYTHONUTF8=1`, or a
non-cp1252 character leaves a 0-byte log.


Correction to the exp095 note above: the inference probe used exp002, a yolo11m-seg trained at 1280, not a same-architecture model. The measured conclusion stands (the public recipe scores 0.41114 on our weights), and the remaining gap is attributable to model size and/or training resolution and data, which exp095 cannot separate.

## 2026-09-15 23:30 Nairobi, exp097 prepared (RunPod, not launched)

User direction: spend at most $9.95 of RunPod compute on the strongest possible
pretrained detector for the 0.39 pipeline. Train until converged, not to a clock.
Keep GPU utilization above 70%. Document everything. No local downloads.

Plan and full rationale: `docs/exp097_runpod_training_plan.md`.

- Architecture: YOLOv8l-seg, COCO-pretrained, native imgsz 2048, canonical split,
  labels with the 0.5 px polygon offset.
- Length: convergence-bound, with a constant rate, decay ×0.2 after a 12-epoch
  plateau, and a stop after 3 decays. There is no time or epoch schedule.
- Speed: RAM cache (asserted), workers = vCPUs − 2, and the preflight picks the
  fastest batch on the densest images. User clarification: 70% utilization is a
  speed aim, not a limit. Utilization is logged every epoch and never blocks or
  stops training.
- Transfer: code arrives as a private Kaggle dataset verified by SHA256. Weights
  leave as the private dataset `cosmicwitness/sol-exp097-yolov8l-2048`, uploaded
  at every decay and at the end, then verified. Evaluation runs on a Kaggle T4
  (exp098).
- Verified locally: 18 unit tests pass, all modules compile, both shell scripts
  pass `bash -n`. Code tarball SHA256
  `cfd0626310ce6f47b7250498d10689e434b504e18fb73525d4bdd97df26dd792` (29,316 bytes).
- Nothing uploaded, nothing launched, nothing billed. Waiting on: the user's
  RunPod secret `kaggle_key`, and approval to upload the code dataset and create
  the pod.

### 2026-09-15 23:45 Nairobi, exp097 code uploaded; pod on hold

- User decisions:
  - pass the Kaggle key directly in the pod request, with no RunPod secret;
  - the code upload is approved;
  - pod creation is on hold;
  - 70% GPU utilization is a speed aim, not a gate.
- The preflight now picks the fastest batch that fits and fails only when nothing
  fits. The low-utilization stop was removed from `train.py`. Tests: 18 pass.
- First upload attempt: Kaggle CLI 2.2.4 on Windows failed on a forward-slash
  `-p` path before sending anything.
- Second attempt: Kaggle auto-extracted the `.tar.gz`, which would have broken
  the pod's SHA256 check.
- Fix: a reproducible gzip-compressed tar named `sol_exp097_code.payload`,
  uploaded from inside the folder (`CLAUDE.md` section 15).
- Verified listing of `cosmicwitness/sol-exp097-code`: exactly
  `sol_exp097_code.payload` (29,279 bytes, SHA256
  `a56bbf593f012afa563c89b4fdcb27163186aa88dd4e96f1c6d8d710f0d13509`) and
  `manifest.json` (1,838 bytes).
- `experiments/exp_097_runpod_yolo/pod/pod_request.json` was regenerated with that
  hash and a key placeholder. The real key is absent from the file. Nothing is
  launched or billed.

### 2026-09-16 00:03 Nairobi, exp097 LAUNCHED on RunPod

- Pod `y2cc247f9sq3b5`, name `sol-exp097-yolov8l-2048`, created
  2026-09-15T21:03:32Z, status RUNNING.
- Hardware: 1× NVIDIA L40 (48 GB) on community cloud at $0.69/h; host CUDA 12.8,
  9 vCPUs, 151 GB RAM, 30 GB container disk, 60 GB persistent `/workspace`.
- GPU change from the plan: community L40S hosts reported only CUDA 12.4, which
  cannot run the cu128 image. The L40 was chosen as the cheaper 48 GB equivalent,
  and the user supplied the key after being told.
- Pre-launch fixes: the RunPod PyTorch image is Ubuntu 24.04 with PEP 668, so the
  launcher and pod environment now use `--break-system-packages` and
  `PIP_BREAK_SYSTEM_PACKAGES=1`. Only the start command changed; the payload is
  unchanged (SHA256 `a56bbf59...13509`).
- Tests before launch: 28 pass (unit, packaging consistency, real-data pipeline).
- The Kaggle key was supplied by the user directly in chat. It sits in the pod
  environment and in no repository file.
- Budget cap $9.95 (about 14.4 h at this price). The run is convergence-bound;
  the pod uploads to `cosmicwitness/sol-exp097-yolov8l-2048` and stops itself.

### 2026-09-16 00:20 Nairobi, first pod stopped at preflight; fixed and republished

Pod `y2cc247f9sq3b5` ran about 3 minutes (≈$0.04), uploaded diagnostics to
`cosmicwitness/sol-exp097-yolov8l-2048` and stopped itself. Self-stop and
diagnostics both worked as designed. Everything before preflight succeeded:

| check | result |
|---|---|
| environment | L40 (45,468 MiB usable), CUDA 12.8, torch 2.8.0+cu128, Ultralytics 8.4.153, shapely 2.1.2 |
| weights | `yolov8l-seg.pt`, SHA256 `ca3ec94c...` |
| data | annotation hash matched, 707/180 photographs, 974 train / 180 val labels, 6,874 / 1,325 instances |
| preflight | batch 2 on the densest images (up to 23 instances): 9.91 img/s, data wait 0.17%, peak reserved 39,418 MiB (86.7%) |

Causes:
- **Headroom rule:** 0.85 rejected batch 2, the only batch that fits a 48 GB card, so "nothing fits" was reported.
- **ERR trap:** the trap fired on the preflight's expected non-zero exit even under `set +e`, which gave the generic "line 171".
- **Measurement window:** 4 s, too short for stable throughput or any GPU samples.
- **Worker count:** read the host's 96 cores instead of the pod's 9 vCPUs.

Fixes:
- **Headroom:** 0.92, with batch 1 added as a candidate.
- **Exit capture:** `|| PREFLIGHT_RC=$?`.
- **Measurement:** at least 240 images per candidate.
- **Workers:** from the cgroup CPU quota.
- **Regression test:** built from the real measurement. Tests pass (24 + packaging).

The failed pod was terminated (404 confirmed). Code dataset version 3 holds payload
SHA256 `f232b5a37b4191ea4b754f3ba23d0f06b34a85e186a822341812b92abc239b89`
(29,889 bytes).

### 2026-09-16 00:18 Nairobi, exp097 relaunched

- Pod `qsatuc7fdharyi`, created 2026-09-15T21:17:54Z, status RUNNING.
- Hardware: 1× NVIDIA L40 on community cloud at $0.69/h; host CUDA 13.0,
  24 vCPUs, 250 GB RAM, 60 GB persistent `/workspace`.
- Code: dataset `cosmicwitness/sol-exp097-code` version 3, payload SHA256
  `f232b5a37b4191ea4b754f3ba23d0f06b34a85e186a822341812b92abc239b89`.
- The weights dataset `cosmicwitness/sol-exp097-yolov8l-2048` already exists
  from the first pod's diagnostics, so this run's uploads become new versions.

## exp097 pod 2 crashed at training start; four Ultralytics 8.4.153 defects fixed (2026-09-16)

Pod `qsatuc7fdharyi` (1x L40, community, $0.69/h) passed every stage and then failed
about 35 seconds into training. The attempt cost roughly ten minutes of pod time.

Evidence, from the artifacts it uploaded to `cosmicwitness/sol-exp097-yolov8l-2048`:

- `error.json`: `TypeError: object of type '_ImageCache' has no len()` at
  `train.py:220`, inside `on_train_start`.
- `events.log`: the preflight passed and chose batch 2 (9.36 img/s, 39,396 MiB of
  45,458, data wait 0.18%). Batch 1 gave 8.61 img/s at 80.8% GPU utilization and
  90% of the power limit. Batch 3 gave 9.38 img/s but 42,396 MiB, above the 0.92
  headroom, which stopped the ladder.
- `train.log`: AMP checks passed, the RAM cache completed (11.4 GB, 974/974
  images), `AdamW(lr=0.001)` with 106/117/116 parameter groups, then the crash.

Cause: the callbacks were written against an assumed Ultralytics API and tested
only against that assumption (CLAUDE.md sections 6 and 8). Reading the 8.4.153
source found four defects. Only the first was fatal.

1. A complete RAM cache is packed into an `_ImageCache`, so `dataset.ims` is no
   longer a list and has no `len()`. The driver now reads `shapes`, and the
   object's type is itself the proof that every image is cached.
2. `final_eval` validates best.pt after the loop and calls `on_fit_epoch_end` a
   second time with `trainer.epoch` advanced by one. The old code would have
   recorded a phantom epoch, and could have decayed the rate or stopped on it.
   That call is now recorded as the final validation.
3. `tloss` is a dict keyed by loss name, not a tensor. Every per-epoch loss would
   have been written as null.
4. An out-of-memory error in the first epoch halves the batch and rebuilds the
   loaders, optimizer and scheduler, with only a log warning. The driver now
   reports a `batch_reduced` event, re-checks the rebuilt pipeline and records the
   reduction in the summary.

Two related changes:

- The learning-rate read-back moved from `on_train_epoch_start` to the first batch
  of the epoch, because Ultralytics calls `scheduler.step()` between the two. It
  now checks the rate the optimizer actually trains with.
- The preflight's validation split now holds one full validation batch of the
  densest images, twice the training batch, which is what Ultralytics validates
  with. The measured peak memory therefore covers the validation pass that ends
  every real epoch.

exp099 (Kaggle T4 kernel `sol-exp099-exp097-smoke-t4`, pushed 2026-09-16) runs the
pod code, from the payload bytes the pod verifies, against Ultralytics 8.4.153: the
data build, the preflight, a converged run with two decays and stage snapshots, and
a run with an injected first-epoch out-of-memory error followed by a budget stop.
The pod relaunches only after the kernel prints `SMOKE_RESULT {"passed": true}`.

### exp099 smoke result (2026-09-16, Kaggle T4, free compute)

`SMOKE_RESULT {"passed": true, "failed": [], "warnings": [], "checks": 23}`, on
payload `bce589a9...`, the bytes the pod verifies. What the 23 checks proved:

- data: 974/180 labels and 6,874/1,325 instances, identical to the pod's build;
- preflight: both candidates ran, and the validation split held one full
  validation batch (2 images at batch 1, 4 at batch 2);
- run A: `_ImageCache` read back as 240/240 cached, AdamW at lr0 with warmup of
  120 iterations, decisions `warmup, improved, decay, decay, stop`, rates
  1e-3 -> 1e-3 -> 2e-4 -> 4e-5, four loss terms per epoch, one FINAL_VALIDATION
  and exactly five epochs, `stop_reason converged`, both stage snapshots loading
  as segment models, three upload hand-offs with all 11 artifacts, exit code 0;
- run B: the injected first-epoch out-of-memory error took Ultralytics' own retry
  path (`Reducing to batch=1`), the driver reported `batch_reduced` with the
  re-checked pipeline (240 batches, warmup 240, AdamW, cache complete), and the
  budget stop fired with `epochs_completed 1`;
- GPU monitor: 55 samples, peak utilization 93%.

RunPod spend before the relaunch: $0.2245 (pod 1 $0.037, pod 2 $0.118, disk
$0.085). Pod `qsatuc7fdharyi` is terminated, so its disk charge has stopped. The
relaunch carries `SOL_BUDGET_USD=9.70`, which keeps the total under the $9.95 cap.

### exp097 relaunch and its monitoring limits (2026-09-16)

Pod `hndexs5w71u2wd` started 03:18:22 UTC: 1x L40 community, $0.69/h, CUDA 12.8,
9 vCPU, 151 GB RAM, `SOL_BUDGET_USD=9.70`, code v4 `bce589a9...`.

The MCP `stream-pod-logs` tool failed twice with 300-second idle timeouts, so the
pod's `STAGE` and `EPOCH` lines cannot be read while it trains. What remains
observable:

- `get-pod` runtime: at 626 s uptime it read GPU 99%, VRAM 72% of 48 GB, CPU 25%,
  host memory 18% of 151 GB. Sustained load with a large host-memory footprint is
  consistent with training after the RAM cache filled, but it is an inference, not
  a reading of the configuration.
- the weights dataset `cosmicwitness/sol-exp097-yolov8l-2048`, which gains a full
  artifact snapshot at every learning-rate decay, on an error, and at the end. A
  poller checks it every two minutes. No new file has appeared since the failed
  run's 21:27 diagnostics, so nothing has crashed.

Evaluation is ready in advance: `tests/test_exp097_evaluator_parity.py` passes, so
`evaluate_checkpoints.py` reproduces the exp062 baseline 0.4558638687 from the
stored exp002 D4 pool, and `kaggle/exp098_exp097_evaluation_t4` is regenerated
from the current source. Push it only after the pod's final upload is listed.

Correction on monitoring: `stream-pod-logs` does work. It stalls with
`source: "container"`, and with `since`, but returns immediately with
`source: "both"`, a `tail` count and `maxWaitMs` near 3000. Use that shape.

Training start, read back at 03:25:12 UTC rather than inferred:

`batch 2, batches_per_epoch 487, cache {container _ImageCache, mode ram, cached
974, samples 974}, imgsz 2048, optimizer AdamW, initial_lr [0.001],
warmup_iterations 1461, epoch_ceiling 1000`.

The preflight took batch 2 (9.66 img/s, 39,396 MiB of 45,468) over batch 1 (9.74
img/s, 36,418 MiB) on the 1% tie rule, and stopped at batch 3 (42,396 MiB, above
the 0.92 headroom). Its projection was 126 s per epoch at $0.0241.

First four epochs, each about 110 s after the first:

| epoch | fitness | mask mAP50 | mask mAP50-95 | lr | reason | spend |
|---|---|---|---|---|---|---|
| 1 | 0.32373 | 0.37926 | 0.11601 | 0.00033 | warmup | $0.170 |
| 2 | 0.29412 | 0.37561 | 0.11024 | 0.00067 | warmup | $0.191 |
| 3 | 0.36914 | 0.45525 | 0.12414 | 0.00100 | warmup | $0.212 |
| 4 | 0.39705 | 0.48239 | 0.14801 | 0.00100 | improved | $0.233 |

Data wait is 0.2% with 16 workers, so the input pipeline is not the constraint,
and `get-pod` reads GPU utilization at 99-100%. One harmless flaw: the nvidia-smi
sampler records nothing on this host, so `gpu_util`, `power_fraction` and
`vram_mib` are null in the epoch events. It is diagnostic only and never gated
training; `get-pod` supplies the same figure.


### exp097 first stage ended at epoch 75 (2026-09-16 05:42 UTC)

Stage 1 ran 75 epochs at a constant 1e-3, then the controller decayed to 2e-4.
Read from `epochs.jsonl`, which the pod uploaded with the snapshot:

| epoch | fitness | smoothed | best epoch | note |
|---|---|---|---|---|
| 10 | 0.4575 | 0.4751 | 10 | improved |
| 20 | 0.5824 | 0.5499 | 20 | improved |
| 30 | 0.5726 | 0.5732 | 29 | plateau counting |
| 40 | 0.6057 | 0.5972 | 35 | plateau counting |
| 50 | 0.6117 | 0.6202 | 50 | improved |
| 60 | 0.5821 | 0.6068 | 52 | plateau counting |
| 70 | 0.6239 | 0.6242 | 63 | plateau counting |
| 75 | 0.6313 | 0.6087 | 63 | decay to 2e-4 |

Best epoch so far is 72: fitness 0.64795, mask mAP50 0.65139, mask mAP50-95
0.23538. Epochs take 108 s, spend at the decay was $1.715 of the $9.70 cap. The
learning rate printed for epoch 75 is still 1e-3 because the rate is recorded
before the decay is applied; 2e-4 takes effect at epoch 76.

The snapshot upload proves the checkpoint path works end to end:
`stage1_best.pt` (92,809,857 bytes) plus args.yaml, resolved_args.json,
epochs.jsonl, events.log, results.csv, train.log and manifest.json, listed in
`cosmicwitness/sol-exp097-yolov8l-2048` at 05:42:41. A checkpoint now exists off
the pod, so a host failure no longer costs the whole run.

Kernel `sol-exp098-exp097-evaluation-t4` was pushed at 05:45 against that
snapshot. It re-checks the exp062 parity (0.4558638687) and then scores
stage1_best.pt with the single-view sweep and the D4 pipeline, so the run's
progress is read in exact PQ rather than in Ultralytics fitness.

### The first decay worked, and the submission path is ready (2026-09-16 05:54 UTC)

Stage 2 began at epoch 76 with the rate at 2e-4. Every epoch since has improved:

| epoch | fitness | mask mAP50 | mask mAP50-95 | spend |
|---|---|---|---|---|
| 76 | 0.65830 | 0.67648 | 0.24287 | $1.736 |
| 77 | 0.66913 | 0.67330 | 0.24504 | $1.756 |
| 78 | 0.67774 | 0.67963 | 0.25054 | $1.777 |
| 79 | 0.67723 | 0.67641 | 0.24845 | $1.798 |
| 80 | 0.68552 | 0.68582 | 0.25400 | $1.819 |
| 81 | 0.68618 | 0.68504 | 0.25022 | $1.839 |

Stage 1 plateaued at 0.648 (epoch 72). The decay lifted it to 0.686 within six
epochs, so the plateau was a learning-rate limit rather than the model's ceiling.

Submission path, written while training continues:

- `scripts/emit_exp097_submission.py` emits a test CSV from an exp097 checkpoint
  in either mode: `single` (2048, floor 0.05, NMS 0.60, retina) or `d4` (eight
  dihedral views at floor 0.25 through `predict_views`, fused by
  `cached_tta.fuse(0.5, 0.5)`), painted with `sweep.evaluate`'s exact steps.
- It replays the rule on validation before touching a test photograph, and
  aborts unless it reproduces exp098's PQ within 5e-4. The replay reads exp098's
  own saved D4 pool through `kernel_sources`, so it costs seconds rather than a
  second forty-minute validation pass.
- `scripts/package_exp097_submission.py` builds the T4 kernel, mounts the weights
  dataset, and verifies the checkpoint's SHA256 against the pod's manifest before
  inference.
- Parity is proved on real data, not asserted: emitting the stored exp002 D4 pool
  at conf 0.35, erosion 1, minimum area 300 reproduces `sweep.evaluate` exactly,
  TP/FP/FN 860/397/465 and PQ 0.4558638687292126.
  `tests/test_exp097_emission_parity.py` locks that property.

### exp098 scored stage1_best.pt: 0.0021 below the baseline, so no submission (2026-09-16 06:06 UTC)

The evaluator reproduced the exp062 baseline exactly before scoring anything:
PQ 0.45586386872921264, TP/FP/FN 860/397/465.

| path | PQ | conf | erosion | min area | TP | FP | FN | SQ | RQ |
|---|---|---|---|---|---|---|---|---|---|
| exp002 D4 baseline | 0.455864 | 0.35 | 1 px | 300 | 860 | 397 | 465 | 0.6843 | 0.6662 |
| stage1 single view | 0.429547 | 0.30 | none | 300 | 721 | 241 | 604 | 0.6813 | 0.6305 |
| stage1 D4 | 0.453765 | 0.25 | none | 300 | 788 | 283 | 537 | 0.6899 | 0.6578 |

Delta against the baseline: **-0.0021**. Validation has run 0.067-0.078 below
public, so this predicts about 0.38 and would not beat the standing 0.39. No
submission was made. CLAUDE.md section 11: a measurement that says a change is
negligible must change the action, not be overridden by it.

The failure is specific rather than general. The 2048-trained model prefers no
erosion, as expected, and it is the cleaner detector: SQ 0.6899 against 0.6843
and 283 false positives against 397. It loses on recognition: 788 true positives
against 860, and 537 misses against 465. That is a recall problem at epoch 72,
which is the *first* plateau; training has since improved fitness from 0.648 to
0.695 and continues.

One defect in the sweep itself: the D4 optimum landed on the grid's lower edge,
0.25, which is also the view floor. `cached_tta.fuse` scales a cluster's score by
0.5 + 0.5 * support, so a mask seen by one view of eight keeps barely more than
half its confidence and fused scores below 0.25 are ordinary. The grid could not
see them. `evaluate_checkpoints.py` now sweeps from 0.10 in both modes, and the
exp098 kernel is rebuilt from that source.

Next: score `stage2_best.pt` when the second decay uploads it, and submit only
if D4 PQ clears 0.455864 by a material margin.

### exp100: stage1 submitted at the user's direction (2026-09-16 06:12 UTC)

The user asked for a submission from the first checkpoint regardless of the
measured delta: "submit the first checkpoint regardless. i want to see something".
Kernel `sol-exp100-exp097-submission-t4` was pushed at 06:12 with exp098's own
winning rule for `stage1_best.pt`: mode D4, confidence 0.25, no erosion, minimum
area 300, expected validation PQ 0.4537645693100445.

What the kernel guarantees before it writes a row:

- the checkpoint's SHA256 matches the pod's manifest entry;
- the rule is replayed on the canonical validation split, reusing exp098's saved
  D4 pool, and the run aborts unless it reproduces 0.453765 within 5e-4;
- every emitted instance has positive area, and masks within a photograph are
  pixel-disjoint, checked by decoding as the CSV is written.

Expected outcome, stated before the fact: validation 0.453765 sits 0.0021 under
the exp002 baseline, and validation has run 0.067-0.078 below public, so this
should land near 0.38 and not beat the standing 0.39. It costs one of the day's
submission slots. The value is a leaderboard reading of a 2048-trained YOLOv8l-seg
at its first plateau, against a model whose recall is still improving: at epoch 86
fitness had reached 0.695 from the 0.648 that produced this checkpoint.

### exp100 scored 0.38 public, and it calibrates the new model (2026-09-16 06:41 UTC)

Submission **56271947**, file `data/exp100_submission/submission.csv`, message
"exp097 stage1_best.pt D4 conf0.25 no-erosion area300, validation PQ 0.453765".
Public score **0.38**, against the standing 0.39. Four slots remain today.

The whole chain verified itself before spending the slot:

- the checkpoint's SHA256 matched the pod manifest
  (`cd717809746bec5022725f47f11f72cdef1a75ebe6fa82f6951bb1380952adbf`);
- the replay reproduced exp098's PQ bit-for-bit, 0.4537645693100445, from
  exp098's own saved pool, so the emitted rule is provably the scored rule;
- 1025 rows over 175 photographs, 5.86 instances each, pixel-disjoint.

Two results worth keeping:

1. **The validation-to-public offset holds for this architecture.** 0.453765
   validation gave 0.38 public, an offset of 0.0738, inside the 0.067-0.078 band
   measured on the exp002 model. Public 0.46 therefore needs about 0.53
   validation PQ, which is 0.077 above what stage1 reached.
2. **Five test photographs received no prediction at all** at confidence 0.25.
   Under PQ an empty prediction on a non-empty photograph is pure false negative.
   That is consistent with the recall deficit exp098 measured on validation
   (788 true positives against the baseline's 860), and it says the confidence
   floor, not the mask quality, is what suppresses instances.

### exp097 converged at epoch 124 and the pod stopped itself (2026-09-16 07:11 UTC)

`summary.json`, uploaded by the run itself:

- `stop_reason` **converged**, 124 epochs, 3 decays (1e-3, 2e-4, 4e-5, 8e-6),
  final `lr_scale` 0.008;
- controller best at epoch 88 by smoothed fitness; best raw fitness 0.69496 at
  epoch 86;
- final validation of best.pt: mask mAP50 **0.70395**, mask mAP50-95 **0.26242**,
  mask precision 0.69576, mask recall 0.65414, box mAP50 0.70385;
- `batch` 2 throughout, `batch_reductions` empty, so Ultralytics never silently
  halved the batch;
- spend at train end **$2.738** of the $9.70 cap. With the two failed pods
  ($0.2245) the whole exp097 effort cost about $2.96 of the user's $9.95.

Five checkpoints are on `cosmicwitness/sol-exp097-yolov8l-2048`, each with a
SHA256 in `manifest.json` that every evaluation and submission verifies:
best.pt and last.pt (92,779,844 bytes), stage1_best.pt (92,809,857),
stage2_best.pt and stage3_best.pt (92,812,417).

The pod uploaded, verified its own listing, stopped itself (status EXITED), and
was terminated after I read the listing independently. No pods remain.

Open question the numbers pose: the model gained fitness from 0.648 at stage 1 to
0.695, yet stage1's D4 PQ was 0.453765, only 0.002 under the exp002 baseline.
Fitness is mAP, not PQ, so the next evaluation has to say whether those gains
converted into PQ. exp098 v2 is scoring stage1 and stage2; a third run covers
best.pt, last.pt and stage3_best.pt with the confidence grid widened to 0.10.

### Direction from the user: a pipeline, not one model (2026-09-16)

Recorded verbatim in intent, because it governs what comes after the current
checkpoints: train several models for different jobs and combine them into a
pipeline that reaches a decision, rather than loading detection, boundary
placement and instance separation onto one segmentation network. The user
expects non-ML components too, drawn from solar physics and classical computer
vision or computer science, and says this is not just an ML-only competition.
Sequencing is explicit: first beat everything available out of the trained
checkpoints and their ensembles, then build the pipeline.

Three measurements in this repository already point the same way:

1. `docs/HOST_GUIDANCE.md`: judging weighs individual predictions and algorithmic
   logic, not a scalar PQ alone.
2. exp098 on stage1: the 2048-trained model draws better masks than exp002
   (SQ 0.6899 against 0.6843, 283 false positives against 397) but finds fewer
   filaments (788 true positives against 860). The binding constraint is
   recognition, and more of the same detector keeps meeting it.
3. Line 717 above: Hu et al. (arXiv:2607.24525) reach a solar-filament workflow
   by decomposition - multiscale local-contrast segmentation, then image-driven
   detail refinement, then instance integration.

Candidate stages, to be judged each against the specific error it removes rather
than against pooled PQ: a recall-oriented proposal stream (local contrast or
hysteresis thresholding across the H-alpha disk, which owes nothing to a learned
detector's confidence floor), a spine tracer for elongated structures, a splitter
and merger for instance decisions (watershed, or graph linking along the spine),
and a learned verifier that decides keep or drop. The exp092 refiner and the
exp077-exp082 semantic-support work are earlier attempts at single stages of this
shape; their null results belong in the design, not in the bin.

### exp098 v2: stage2 clears the baseline by 0.0109 (2026-09-16 07:25 UTC)

Second evaluation, confidence grid widened to 0.10-0.50 in both modes. Parity
reproduced first: 0.45586386872921264, TP/FP/FN 860/397/465.

| checkpoint | mode | PQ | conf | erosion | area | TP | FP | FN | SQ | RQ |
|---|---|---|---|---|---|---|---|---|---|---|
| exp002 baseline | D4 | 0.455864 | 0.35 | 1 px | 300 | 860 | 397 | 465 | 0.6843 | 0.6662 |
| stage1_best.pt | single | 0.429547 | 0.30 | none | 300 | 721 | 241 | 604 | 0.6813 | 0.6305 |
| stage1_best.pt | D4 | 0.453765 | 0.25 | none | 300 | 788 | 283 | 537 | 0.6899 | 0.6578 |
| stage2_best.pt | single | 0.452597 | 0.30 | none | 300 | 849 | 386 | 476 | 0.6824 | 0.6633 |
| **stage2_best.pt** | **D4** | **0.466806** | **0.35** | none | 300 | 819 | 269 | 506 | 0.6877 | 0.6788 |

**+0.010942 over the baseline**, and the optimum is interior to the widened grid,
so it is a real peak rather than the boundary artefact the first run reported.

What changed between the two checkpoints is recall, exactly the deficit exp098 v1
identified. Single view went from 721 true positives to 849 against the baseline's
860, and D4 from 788 to 819 while false positives fell from 283 to 269. Twenty-five
more epochs at a decayed rate converted into found filaments, not just tidier masks.

At this model's measured offset of 0.0738 (stage1: validation 0.453765, public
0.38), 0.466806 predicts about 0.393 public: level with the standing 0.39 at two
decimals, 0.40 if the offset lands at the favourable end of its 0.067-0.078 band.
No slot was spent on it, because exp098 v3 is already scoring all five
checkpoints, including best.pt from epoch 88 and stage3_best.pt.

### stage2 submitted: public 0.39, tying the standing best (2026-09-16 08:14 UTC)

Submission **56273757**, `data/exp100_stage2/submission.csv`, message "exp097
stage2_best.pt D4 conf0.35 no-erosion area300, validation PQ 0.466806". Public
**0.39**. Three slots remain today.

Verified before the slot was spent: checkpoint SHA256
`5771693bc1f986c0194ea8b402633d6351a26a69ba3cadf8349a850d59d6f9c5` matched the
pod manifest; the rule replayed to 0.4668060104154499 exactly, from exp098's own
`d4_pool_stage2_best.json`; 1037 rows over 177 photographs, pixel-disjoint.

The arithmetic that matters now:

- stage1: validation 0.453765 -> public 0.38, offset 0.0738.
- stage2: validation 0.466806 -> public 0.39, offset 0.0768.

Both sit inside the historical 0.067-0.078 band, so the predictor holds for this
architecture. The leaderboard prints two decimals, so 0.39 means public is in
[0.385, 0.395) and the +0.011 validation gain fell inside one printed bucket.
CLAUDE.md already records this trap: a local spread of 0.013 is invisible there.

**The bar is now explicit.** To print 0.40 the submission needs public >= 0.395,
which at an offset of 0.0768 means validation PQ **>= 0.472**, about +0.005 beyond
stage2. To reach the user's 0.46 the requirement is validation ~0.537. Every
remaining idea should be judged against those two numbers rather than against
"is it better than the baseline".

### exp098 v3: training after epoch 88 produced nothing (2026-09-16 08:43 UTC)

All five checkpoints scored with the grid widened to 0.10; parity reproduced at
0.455863868729 first.

| PQ | mode | checkpoint | conf | erosion | area | TP | FP | FN |
|---|---|---|---|---|---|---|---|---|
| 0.466806 | D4 | **best.pt** | 0.35 | none | 300 | 819 | 269 | 506 |
| 0.466806 | D4 | stage2_best.pt | 0.35 | none | 300 | 819 | 269 | 506 |
| 0.452597 | single | stage3_best.pt | 0.30 | none | 300 | 849 | 386 | 476 |
| 0.452597 | single | stage2_best.pt | 0.30 | none | 300 | 849 | 386 | 476 |
| 0.452597 | single | best.pt | 0.30 | none | 300 | 849 | 386 | 476 |
| 0.449723 | single | last.pt | 0.35 | none | 300 | 792 | 296 | 533 |
| 0.429547 | single | stage1_best.pt | 0.30 | none | 300 | 721 | 241 | 604 |

**best.pt, stage2_best.pt and stage3_best.pt predict identically** - the same
TP/FP/FN in two independent modes. They are the same weights: Ultralytics' best
fitness peaked at epoch 86-88 and was never beaten, so every later stage snapshot
copied an unchanged best.pt. Their SHA256s differ only because `strip_optimizer`
stamps a fresh timestamp into each copy.

Two consequences:

1. **Epochs 88 to 124 bought nothing.** The convergence controller was right to
   stop and could have stopped 36 epochs earlier. A future run should snapshot on
   *improvement* rather than on decay, because a decay snapshot can duplicate the
   previous one and hide that fact behind a different hash.
2. **No single checkpoint clears the 0.472 bar** that a printed 0.40 requires.
   The measured ceiling of this training run is 0.466806.

The three genuinely distinct models are best.pt (epoch 88), stage1_best.pt
(epoch 72) and last.pt (epoch 124), and their error profiles differ: 269, 241 and
296 false positives against 819, 788 and 792 true positives. exp101
(`sol-exp101-exp097-ensemble-t4`, pushed 08:44) pools all three as 24 dihedral
views, fused with the support denominator set to 24 rather than the hardcoded 8,
to attack the recall deficit directly.

Process note: this entry had to be written twice. The first attempt ran beside a
push that `cd`-ed the shared working directory, so its relative path did not
resolve and only the trailing `echo` reported success. CLAUDE.md 15 records that
exact hazard; use absolute paths whenever Bash calls run in parallel.

### exp103: a single-pixel classical proposal stage cannot separate filaments (2026-09-16 09:37 UTC)

First component of the pipeline direction, and a negative result worth keeping.

`experiments/exp_103_classical/src/proposals.py` implements the physics route:
detect the disk, divide out limb darkening with the disk's own radial median
profile, subtract a large-scale background, hysteresis-threshold so weak filament
tails grow from confident seeds, then keep elongated components. Run on three
validation photographs it produced 1-3 proposals each and **covered no ground-truth
filament at all**, best IoU 0.0 even at a 0.1 threshold.

The diagnosis is not the thresholds. On `20110220082634Lh.jpeg`:

- flattening works: 96.8% of disk pixels valid, flat-field median 0.9981;
- at the 0.035 seed threshold the mask keeps 76% of truth pixels but also
  **1,123,227 disk pixels**, about a third of the disk, so hysteresis grows one
  enormous blob and the elongation filter then discards it;
- contrast inside filaments (median 0.099, p75 0.161, p90 0.210) is not separable
  from contrast outside them (p90 0.132, p99 0.475).

Three discriminators measured the same way - the fraction of disk retained when
the threshold keeps 80% of truth pixels, averaged over three photographs:

| quantity | disk kept | filament median | disk p90 |
|---|---|---|---|
| flat darkness (1 - flattened intensity) | 0.374 | 0.0886 | 0.1509 |
| wide depression (sigma 60 background) | **0.300** | 0.0952 | 0.1181 |
| elongated black top-hat, 6 orientations | 0.378 | 0.1202 | 0.2641 |

A usable proposal stage needs that fraction well under 0.05. At 0.30 the "proposal"
is a third of the sun. Filaments are not separable from the chromospheric network by
any of the tested single-pixel intensity criteria applied **globally over the whole
disk**, which is consistent with the project's history: the largest classical gain
here came from erosion of learned masks (exp_005, +0.03 public), not from
thresholding raw intensity.

SCOPE CORRECTION (2026-09-16, per the testing manual, method 5): this result tests
global proposal generation only. It does not test local ridge evidence sampled
inside a forward endpoint corridor at scales matched to filament width with the
correct dark-ridge polarity, which remains untested. Global disk occupancy of 26.2%
must not be cited as a rejection of local vesselness.

What the numbers redirect toward: none of the three quantities measures *length*.
A filament is a connected dark ridge tens to hundreds of pixels long; network
darkness is speckle at 5-15 px. The next test is therefore structural - Hessian
ridge response across filament widths, scored by the identical measurement - and
it is cheap and local. If ridge response also keeps 30% of the disk, the classical
first stage is dead as a *proposal* source, and the pipeline's non-ML component
should move to where classical methods already proved themselves here: refining
and splitting learned masks, not finding filaments from raw pixels.

**Ridge test result (same session).** Hessian ridge response, computed at sigma
2/4/8 with the line-likeness weighting and scored by the identical measurement:

| photograph | ridge keeps | wide depression keeps |
|---|---|---|
| 20110220082634Lh | 0.387 | 0.372 |
| 20110324082614Th | 0.194 | 0.211 |
| 20110325171014Mh | 0.204 | 0.319 |
| **mean** | **0.262** | 0.300 |

Structure at the pixel level does no better: 0.262 of the disk retained where a
proposal stage needs under 0.05. **The classical-proposal branch is closed.**
Filaments are not findable from raw H-alpha intensity by thresholding any of
darkness, large-scale depression, orientation top-hat, or ridge response - the
chromospheric network occupies the same value range at every one of them.

Where the non-ML component belongs instead, on this project's own evidence: not
at proposal time but *after* the detector. Classical methods already produced the
single largest gain here by operating on learned masks - exp_005's one-pixel
erosion, +0.03 public - and the same class of operation is still unexploited for
instance decisions: splitting a merged detection along its spine, joining
fragments that a detector broke, and choosing a boundary where annotators
disagree. Those act on ~1000 candidate masks per submission rather than on 3.4
million disk pixels, which is why they can afford to be precise.

### exp101: pooling checkpoints from one run buys nothing (2026-09-16 09:38 UTC)

Three distinct checkpoints, 24 dihedral views, fused with the support denominator
set to 24. Each model was re-scored from the same predictions inside the run.

| source | PQ | conf | TP | FP | FN |
|---|---|---|---|---|---|
| best.pt alone | 0.466806 | 0.35 | 819 | 269 | 506 |
| last.pt alone | 0.460914 | 0.30 | 837 | 332 | 488 |
| stage1_best.pt alone | 0.453765 | 0.25 | 788 | 283 | 537 |
| **24-view ensemble** | **0.466857** | 0.30 | 824 | 285 | 501 |

**+0.000050 over the best single checkpoint.** Five parts in a hundred thousand,
against a leaderboard that cannot resolve 0.013 and a bar of 0.472. The ensemble
gained five true positives and paid sixteen false positives for them.

The reason is diversity, or its absence. exp098 v3 established that best.pt,
stage2_best.pt and stage3_best.pt are the same weights; stage1_best.pt and
last.pt are earlier and later points on one optimisation trajectory. Their
mistakes are the same mistakes, so pooling them mostly re-votes on what they
already agreed about. Note that last.pt alone has the highest recall of the three
(837 true positives) while scoring lower overall - a detector tuned for recall is
available, which matters for a pipeline with a separate verifier.

**Where diversity actually exists**: exp002 is a different architecture
(yolo11m-seg) at a different resolution (1280) trained separately, and it finds
860 filaments to exp097's 819. exp102 pools those two at instance level. Its
first run died on my own bug - `--link` and `--vote` interpolated into a
subprocess argument list as floats rather than strings, `TypeError: expected str,
bytes or os.PathLike object, not float` - fixed and re-pushed.

### exp102: cross-architecture pooling is the first ensemble that works (2026-09-16 10:12 UTC)

exp002's D4 pool (yolo11m-seg, trained at 1280) and exp097's best.pt D4 pool
(YOLOv8l-seg, trained at 2048), fused at instance level with the support
denominator set to the model count. CPU only, no inference: both pools already
existed. Each pool was also swept alone in the same run.

| source | PQ | conf | erosion | area | TP | FP | FN |
|---|---|---|---|---|---|---|---|
| exp002 D4 alone | 0.457492 | 0.40 | 1 px | 100 | 835 | 346 | 490 |
| exp097 best.pt D4 alone | 0.466806 | 0.35 | none | 300 | 819 | 269 | 506 |
| **two-architecture ensemble** | **0.468146** | 0.35 | 1 px | 100 | 852 | 328 | 473 |

**+0.001340 over the best single model, +0.012282 over the baseline.** Small, but
27 times the gain from pooling three checkpoints of one training run (+0.000050),
and it confirms the diagnosis those two results imply together: **ensemble value
comes from different training, not different epochs.** Two networks that differ in
architecture, resolution and optimisation disagree about marginal filaments;
snapshots of one trajectory do not.

The recall arithmetic is the point. exp097 alone finds 819 filaments, exp002 alone
835, the union 852, and misses drop to 473 from 506. False positives sit between
the two parents rather than adding up, which is what the pixel vote is for.

One operating-point surprise worth acting on: the ensemble's best rule uses
**erosion 1 px and minimum area 100** - exp002's preference - not exp097's grow 0
and area 300. Mixed pools do not inherit either parent's emission rule, so the
rule has to be re-swept per pool rather than carried over. exp104 was already
sweeping that axis, including dilation, which the frozen scorer refuses to try.

Still below the 0.472 that a printed 0.40 needs, so no submission slot was spent.

### exp105: the near-threshold failures are shape, not topology (2026-09-16 10:26 UTC)

`scripts/audit_threshold_structure.py`. Motivation, from the user's derivation:
under PQ = sum(IoU)/(TP + FP/2 + FN/2), an object at IoU 0.49 contributes 0 to the
numerator and a full unit to the denominator; at 0.51 it contributes 0.51 and the
denominator is unchanged. A threshold crossing is worth about ten times a
comparable polish of an already-matched pair. The audit builds the bipartite
prediction-truth overlap graph per record - not per-truth maxima, because matching
is global - and classifies its connected components. Baseline reproduced exactly
first: PQ 0.4558638687, TP/FP/FN 860/397/465.

| component | count | in the 0.4-0.5 band |
|---|---|---|
| 1:1 | 953 | **55** |
| N:1 fragmentation | 29 | 5 |
| 1:N merge | 24 | 5 |
| N:M | 4 | 1 |
| 0:1 invisible | 286 | 0 |
| 1:0 spurious | 211 | 0 |

**83% of near-threshold failures are 1:1: one prediction, one truth, wrong shape.**
Topology accounts for 11 of 66. This argues *against* the spine-graph decomposition
that was about to be prioritised, and the order matters: building junction pairing
first would have targeted an eighth of the opportunity.

Oracle bounds, using truth only to set a ceiling and never as an inference rule:

- **union** the predictions overlapping one truth: 30 candidates, **14 cross 0.5**.
  Consistent with exp088's 13 for pair-joining, so that earlier number was not an
  artefact of only testing pairs.
- **split** a prediction covering several truths by truth assignment: 28
  candidates, **40 truths cross 0.5**. The split direction is about three times the
  merge direction - an asymmetry no previous audit had measured, since exp088
  tested joining only.

Two populations are untouchable by any reshaping: 286 of 465 misses have zero
overlap with any prediction, and 211 false positives touch no truth. Those are
detection errors, not geometry errors, and they are why mask surgery alone cannot
reach the goal.

**Revised plan.** Deterministic geometry remains the right instrument, but aimed
at 1:1 shape rather than junctions: local boundary correction conditioned on an
observable proxy (width variation along the spine, tangent discontinuity, semantic
support disagreeing with the mask edge), since the 0.4-0.5 band itself is only
visible with ground truth and using it directly would be an oracle. The 40-rescue
split direction stays worth a bounded implementation afterwards, because splitting
needs a narrow-neck test on one mask rather than a full spine graph.

### exp105b: near-threshold masks are too small, but nothing observable marks them (2026-09-16 10:31 UTC)

Follow-up to the component audit, restricted to strict 1:1 pairs - one prediction,
one truth, no other overlap either way.

| population | n | median IoU | excess/inter | missing/inter | elongation | width CV | solidity |
|---|---|---|---|---|---|---|---|
| near threshold 0.40-0.50 | 55 | 0.459 | 0.18 | **0.76** | 4.49 | 0.149 | 0.494 |
| matched >= 0.50 | 845 | 0.692 | 0.20 | 0.19 | 4.00 | 0.154 | 0.552 |

**Direction is clear: these masks are under-segmented**, median 575 truth pixels
missing against 139 spurious. Growing exactly these 55 objects would add about
0.51 each to the numerator. CORRECTION (user, 2026-09-16): the denominator does
NOT fall. An unmatched pair contributes FP/2 + FN/2 = 1 and a match contributes
TP = 1, so the denominator is unchanged and the entire gain is the numerator
appearing. The leverage argument stands on that alone: rescuing a 0.49 miss adds
about 0.51 to the numerator, polishing a 0.55 match to 0.60 adds 0.05.

**The trigger does not exist in the geometry tested.** Elongation separates by
12.4% of the healthy median, solidity by 10.5%, spine width variation by 2.9% -
medians that close with overlapping distributions, not a decision rule. At
inference time there is no ground truth to say which masks sit in the band, so
without a discriminator the only option is growing everything, and exp088 already
measured that: one uniform dilation drops PQ from 0.4559 to 0.4286.

This is a negative result about the *proxy*, not about the opportunity. Two
observable discriminators remain untested and both are now motivated by measured
results rather than intuition:

1. **Cross-architecture edge disagreement.** exp102 showed two independently
   trained networks carry complementary information (+0.001340 pooled, against
   +0.000050 for three checkpoints of one run). Where exp002 and exp097 disagree
   about a boundary, the mask edge is uncertain; where they agree, it is not.
2. **Semantic support contradicting the mask edge.** exp085's field reaches mask
   Dice 0.703 and spine Dice 0.635. Foreground extending beyond a mask boundary is
   observable evidence that the mask is short - exactly the error measured here.

Sequencing that follows from the two audits: the split repair comes first among
deterministic components, because a narrow neck is directly observable on a single
mask and the oracle showed 40 rescues; 1:1 growth waits on a discriminator, since
its oracle is 55 rescues but its trigger is currently unavailable.

### exp105c: geometry is dead, and the neck trigger fails on precision (2026-09-16 10:44 UTC)

Two audits the user required before any deterministic repair could be built.

**1. Does the geometry carry joint signal, not just similar medians?** Measured by
rank AUC, near-threshold 1:1 (n=55) against healthy 1:1 (n=845):

| feature | AUC | distance from chance |
|---|---|---|
| elongation | 0.539 | 0.039 |
| solidity | 0.424 | 0.076 |
| width CV | 0.457 | 0.043 |
| fixed combination | 0.548 | 0.048 |

All within 0.08 of chance, individually and combined. Geometry is genuinely dead
as a trigger for "this mask stopped too early", not merely unpromising in its
marginals.

**2. Is "narrow neck" a measured trigger or an intuition?** rho = minimum interior
width / median width, from the distance transform projected on the principal axis
(no skeleton: skeletons are unstable under small boundary perturbations).

Oracle-splittable merges median rho 0.629, healthy 1:1 median 0.750, **AUC 0.712** -
which sounds usable until the operating points are examined:

| rho cut | merges caught | healthy masks cut | precision | net objects |
|---|---|---|---|---|
| 0.45 | 2/24 | 30 | 0.062 | -28 |
| 0.55 | 5/24 | 100 | 0.048 | -95 |
| 0.65 | 16/24 | 223 | 0.067 | -207 |
| 0.70 | 17/24 | 312 | 0.052 | -295 |

**Every operating point is strongly negative.** 24 positives against 845 negatives
means even a good-looking AUC yields 3-7% precision, and each false split destroys
a true positive while adding a false positive. The 40-rescue oracle bound was an
upper bound on a repair whose trigger does not exist at usable precision. The
narrow-neck splitter is rejected before implementation, on measurement.

Corrected arithmetic, from the user: crossing 0.5 leaves the denominator unchanged
(FP/2 + FN/2 = 1 becomes TP = 1); the gain is entirely the numerator appearing.
The earlier claim in exp105b that it "removes a full unit of denominator" was wrong
and has been patched in place.

What survives: the two discriminators that are causally connected to the measured
error rather than correlated with it - semantic foreground persisting immediately
outside a mask boundary (directional evidence of under-segmentation), and
cross-architecture disagreement at that boundary (uncertainty gating). Both need
assets that live on Kaggle, so they are a kernel, not local work.

### exp104: dilation is dead on the 2048 ensemble too (2026-09-16 10:45 UTC)

160 grid cells over the exp102 ensemble pool, parity reproduced before any cell
was believed. Best per morphology level, each at its own best confidence and area:

| grow | best PQ | conf | area | TP | FP | FN |
|---|---|---|---|---|---|---|
| -2 | 0.431059 | 0.35 | 100 | 812 | 367 | 513 |
| **-1** | **0.468146** | 0.35 | 100 | 852 | 328 | 473 |
| 0 | 0.441242 | 0.35 | 300 | 820 | 357 | 505 |
| +1 | 0.382756 | 0.35 | 500 | 737 | 415 | 588 |
| +2 | 0.305542 | 0.35 | 800 | 589 | 489 | 736 |

The optimum is interior at -1 and falls away symmetrically. **The hypothesis that a
sharper 2048-trained model would prefer dilation was wrong**, and the guard in
`sweep.evaluate_thresholds` that refuses positive grow was protecting a real
result rather than an outdated one. Minimum area 100 beats 300 by 0.0006; 500 and
800 lose clearly.

Nothing clears the 0.472 needed to print 0.40, so no submission slot was spent.
Best measured configuration for the session stays the two-architecture ensemble at
0.468146 (conf 0.35, erosion 1 px, area 100), +0.012282 over the baseline.

This also sharpens the surviving hypothesis rather than weakening it. Near-threshold
masks are under-segmented by a median 575 pixels, yet *uniform* dilation costs
0.027 PQ here. Growth must therefore be triggered per instance or not at all, which
is exactly what the semantic-support and cross-architecture-disagreement audits are
built to test.

### exp106: semantic support does not mark the under-grown masks either (2026-09-16 11:06 UTC)

The strongest surviving hypothesis, tested with the only independent foreground
evidence in the repository: exp085's saved probability maps, native 2048, from the
run that scored mask Dice 0.703 and spine Dice 0.635. No refitting; ground truth
only labels the two populations for scoring the discriminator.

Per instance, following the user's formulation: exterior annulus support
mean(P | dilate(M,4) \ M) against inner band mean(P | M \ erode(M,4)), on 56
near-threshold 1:1 masks versus 858 healthy ones.

| feature | AUC | near median | healthy median | best operating point |
|---|---|---|---|---|
| sem_delta | 0.543 | -0.5126 | -0.5264 | precision 0.122, net -37 |
| **sem_exterior** | **0.397** | +0.3220 | +0.3844 | precision 0.039, net -165 |
| sem_ratio | 0.448 | +0.3894 | +0.4110 | precision 0.064, net -41 |
| spine_delta | 0.517 | -0.4516 | -0.4477 | precision 0.122, net -37 |
| spine_exterior | 0.402 | +0.4233 | +0.4853 | precision 0.055, net -162 |

**`sem_exterior` is 0.397, below chance.** Near-threshold masks carry *less*
semantic foreground immediately outside them than healthy masks do - the opposite
of what under-segmentation predicts. The field does not see the missing extent.

That converges with two earlier measurements rather than contradicting them:
exp103 found filaments are not separable from the chromospheric network by any
single-pixel intensity criterion, and exp088 found 284 of 465 misses have zero
overlap with any prediction. The most defensible reading is that a substantial
part of the annotated extent is **not photometrically distinguishable** in H-alpha
at all, so no observable field can trigger its restoration.

Triggers now measured and rejected, all on precision at realistic operating
points rather than on AUC: three geometry marginals, their fixed combination, the
narrow-neck statistic, self-pool boundary disagreement, and five variants of
semantic and spine support. The narrowed hypothesis - identify under-grown
instances from independent foreground evidence - has been tested against every
independent evidence source available here and has not survived.

One item of the user's sequence remains genuinely untested: cross-architecture
disagreement between exp002 and exp097, since the 2026-09-16 local run passed the
same pool twice and measured self-consistency instead. exp097's pool lives in
exp098's kernel output, so that runs as a CPU kernel.

### exp108: the missing area is truncation, not thinness (2026-09-16 11:20 UTC)

The audit the user identified as the most valuable next measurement: of the ~575
missing truth pixels in near-threshold 1:1 masks, where are they? Two skeleton-free
measures - axial fraction (missing pixels projecting outside the mask's own extent
on its principal axis) and distance from each missing pixel to the nearest mask
pixel. Healthy matched pairs are the control.

| population | n | missing px | axial fraction | axial px | dist median | dist p90 | within 3 px | beyond 10 px | half-width |
|---|---|---|---|---|---|---|---|---|---|
| near threshold 0.40-0.50 | 50 | 699 | **0.590** | 376 | **10.9** | 30.7 | 0.263 | 0.514 | 2.2 |
| matched >= 0.50 | 791 | 207 | 0.132 | 26 | 2.0 | 5.0 | 0.757 | 0.000 | 3.0 |

**Truncation dominates.** The two populations are qualitatively different rather
than shifted: healthy masks are marginally thin at the boundary (76% of their small
deficit within 3 px, none beyond 10 px), while near-threshold masks are cut short -
59% of the deficit lies beyond the mask's extent, a median 10.9 px away with a p90
of 30.7 px, on bodies whose median half-width is only 2.2 px.

Two consequences:

1. **Algorithm choice is settled.** Endpoint continuation along a supported path,
   not local boundary refinement. A graph cut in a narrow band around the mask
   cannot reach a stub 10-30 px past the end; a geodesic from the endpoint can.
2. **It explains every failed trigger.** The annulus support, boundary
   disagreement and width statistics all sample a 4 px ring around the mask. The
   defect is 10-30 px beyond it. Those audits were measuring the wrong place, so
   their null results say nothing about evidence further along the axis.

Method note: the first two runs of this audit produced no output because the file
was written without an `if __name__ == "__main__"` block - it imported cleanly and
did nothing. Silence from a script is not a negative result.

Population bookkeeping, raised by the user: exp105b reported 55/845 and exp106
reported 56/858 for supposedly the same strict-1:1 populations. Verified cause -
exp105b emitted at min_area 300 (the baseline rule), exp106 at min_area 100 (the
ensemble's best rule). Reproduced exactly: min_area 300 gives 55/845, min_area 100
gives 56/858. Later audits must pin the emission rule explicitly rather than
inherit a default.

### exp106 corrected: features rescored in both directions (2026-09-16 11:27 UTC)

The user caught a methodological error: an AUC of 0.397 is not absence of signal,
it is a feature ranking the classes backwards, worth 0.603 negated. The original
operating-point search scanned only the hypothesised high side, so features
pointing the other way were rejected without being tested. Rescored over both
directions, emission rule pinned at min_area 100 (56 near / 858 healthy):

| feature | AUC | AUC negated | best over both sides |
|---|---|---|---|
| sem_delta | 0.543 | 0.457 | high q0.98: 5/56 caught, 18 damaged, precision 0.217, net -13 |
| sem_exterior | 0.397 | 0.603 | low q0.98: 3/56, 18 damaged, precision 0.143, net -15 |
| **sem_interior** | **0.365** | **0.635** | low q0.98: 5/56, 18 damaged, precision 0.217, net -13 |
| spine_delta | 0.517 | 0.483 | high q0.98: 5/56, 18 damaged, precision 0.217, net -13 |
| spine_exterior | 0.402 | 0.598 | low q0.98: 3/56, 18 damaged, precision 0.143, net -15 |

The strongest variant was never scored before: **low semantic support inside the
mask**, AUC 0.635 negated. Its interpretation is the user's - these are instances
where the semantic model is itself weak, which argues for an uncertainty *gate*
rather than directional growth. No operating point is net-positive (best -13 at
precision 0.217), so it fails as a standalone trigger.

**Correction to the earlier overclaim.** The previous entry concluded that a
substantial part of the annotated extent is "not photometrically distinguishable
in H-alpha at all". The evidence does not support that. What is established is
narrower: the extent is *not recoverable from the tested photometric and semantic
signals* - this learned field's annulus statistics, simple intensity criteria, and
the geometry summaries. Orientation fields, multiscale ridge structure, texture and
contextual continuation remain untested.

exp108 then explained why all of these read near chance: every one of them samples
a 4 px ring around the mask, while the defect lies a median 10.9 px beyond the
mask's end with a p90 of 30.7 px. The measurements were taken in the wrong place.
The next evidence test must look along the axis, ahead of the endpoint, inside an
angular cone - not around the boundary.

### exp107: cross-architecture disagreement fails too (2026-09-16 11:28 UTC)

The last untested trigger from the user's sequence, run properly this time: exp002
(yolo11m-seg at 1280) and exp097 (YOLOv8l-seg at 2048) as independent second
opinions for each other, both directions, emission pinned at conf 0.35 / erosion 1
/ min_area 100.

| audited pool | second opinion | near/healthy | AUC | best operating point |
|---|---|---|---|---|
| exp002 | exp097 | 56 / 858 | 0.595 | q0.95: caught 8/56, damaged 43, precision 0.157, net -35 |
| exp097 | exp002 | 65 / 795 | 0.574 | q0.98: caught 3/65, damaged 16, precision 0.158, net -13 |

Every operating point is net-negative, and at q0.98 the forward direction catches
**zero** of 56. Genuine cross-architecture disagreement performs the same as the
accidental self-pool measurement (0.597): weak ranking signal, unusable precision.

That closes the trigger search on ring geometry. Seven feature families now
rejected on precision at realistic operating points - three geometry marginals,
their combination, the narrow-neck statistic, five semantic and spine annulus
variants in both directions, and cross-architecture disagreement in both
directions. Every one samples a 4 px ring around the mask.

exp108 explains the common failure: the defect is not at the boundary. It sits a
median 10.9 px beyond the mask's end (p90 30.7 px) on bodies with a 2.2 px median
half-width. Searching a 4 px ring for a stub 10-30 px further along the axis cannot
work, whatever the feature. The next measurement must sample **ahead of the
endpoint, inside an angular cone**, which is what exp109 does.

### exp109: there is nothing ahead of the endpoint to walk along (2026-09-16 11:32 UTC)

exp108 showed the defect sits 10-30 px beyond the mask's end, so this audit samples
there: a 35-degree cone, 4-30 px ahead of each principal-axis endpoint, measuring
the learned foreground field, the spine field and structure-tensor orientation
agreement. Both directions scored, emission pinned at conf 0.35 / erosion 1 /
min_area 100 (56 near, 858 healthy).

| feature | AUC | near median | healthy median | best over both directions |
|---|---|---|---|---|
| truth_ahead (LABEL) | 0.655 | 0.0343 | 0.0000 | net -25, precision 0.295 |
| sem | 0.489 | **0.0005** | 0.0008 | net -34 |
| spine | 0.508 | 0.0017 | 0.0021 | net -34 |
| orientation | 0.532 | 0.7360 | 0.7204 | net -42 |

**The instrument is valid**: `truth_ahead` separates the populations (0.655, near
median 0.0343 against healthy 0.0000), so the cone geometry and endpoint extraction
are right and truncated masks really do have truth ahead of them.

**The evidence is absent.** exp085's foreground probability in that exact cone is
0.0005 - five ten-thousandths - where the annotator drew filament. Spine
probability is 0.0017. Orientation agreement is 0.736 against 0.720, chance.

A geodesic needs a low-cost path to follow. Here the cost field is uniformly
maximal precisely where the missing filament lies, so endpoint continuation has
nothing to walk along. Note also that even the oracle label reaches only precision
0.295: acting on perfect knowledge of "truth is ahead" would still damage more
healthy masks than it repairs at that operating point, because the healthy
population outnumbers the target 15 to 1.

This closes, on measurement rather than opinion, the family of methods that
require forward image evidence: endpoint geodesics, elastica continuation,
vesselness, orientation fields and matched filtering (items 1, 2, 5, 6 and 9 of the
user's list). What remains untouched by this result is the structurally
identifiable merge population, and changing the base segmentation rather than
post-processing it.

### exp110: the best-path test, and one correction to exp109 (2026-09-16 12:02 UTC)

exp109 rejected endpoint continuation using the *mean* learned-foreground
probability in a cone. That statistic is wrong for the question: a geodesic follows
the cheapest path, and a thin supported thread inside a mostly empty cone is exactly
what a shortest-path method exists to find. This audit casts rays at one-degree
steps and reports per-ray statistics instead.

| feature | AUC | near median | healthy median | best operating point |
|---|---|---|---|---|
| truth_best_ray (LABEL) | 0.621 | **0.4074** | 0.0000 | net -27, precision 0.292 |
| best_ray_mean | 0.493 | 0.0082 | 0.0137 | net -38 |
| **best_ray_min** | **0.226** | 0.0000 | 0.0000 | net -34 |
| cone_max | 0.466 | 0.1255 | 0.2353 | net -49 |
| spine_best_ray | 0.479 | 0.0165 | 0.0281 | net -33 |
| **dark_best_ray** | **0.611** | 0.1773 | 0.1455 | net -35, precision 0.157 |

The label validates the instrument: along its best ray a truncated mask has truth
covering 41% of the path, against 0% for healthy masks.

**The conclusion survives, for a sharper reason than exp109 gave.** `best_ray_min`
is 0.0000 for both populations: every ray in the cone crosses at least one pixel of
zero learned support, which is precisely what breaks a geodesic. And the best ray's
mean support is *lower* ahead of truncations (0.0082) than ahead of healthy masks
(0.0137) - the field is quieter exactly where the missing filament is.

**One result reopens a closed door.** `dark_best_ray` - raw absorption against a
sigma-25 background, owing nothing to any network - reaches AUC 0.611 with near
median 0.1773 against healthy 0.1455. It is the only image-derived feature today
above 0.6. So the earlier phrasing that the extent is not recoverable from tested
photometric signals was itself too strong: unlearned absorption along the forward
ray does carry signal. It remains unusable alone (net -35, precision 0.157, because
858 healthy masks swamp 56 targets), but as a gate combined with another signal it
is not dead.

Methodological note for the record: exp109's rejection was made on a statistic that
could not see the hypothesis it was testing. The lesson is the same one that has
recurred all session - match the statistic to the mechanism, not to convenience.

### exp111: the width profile does carry structure that one scalar destroyed (2026-09-16 12:08 UTC)

The user's objection to stopping at width CV and rho was that two profiles can share
a scalar summary while differing obviously in local structure. Measured on
oracle-splittable merges (24) against healthy 1:1 masks (858):

| feature | AUC | merge median | healthy median | best operating point |
|---|---|---|---|---|
| **mean_slope** | **0.809** | 0.1486 | 0.0818 | net -18, precision 0.000 |
| max_slope | 0.765 | 0.4264 | 0.2500 | net -17, precision 0.053 |
| saddle_depth | 0.750 | 0.5186 | 0.1652 | net -16, precision 0.100 |
| peak_ratio | 0.718 | 2.8000 | 2.1858 | net -17, precision 0.095 |
| rho (earlier) | 0.304 | 0.5788 | 0.6923 | net -18 |
| width_cv (earlier) | 0.665 | 0.2611 | 0.2150 | net -18 |
| peak_count | 0.529 | 3.0000 | 3.0000 | net -38 |

The objection was right: mean_slope reaches 0.809 and saddle depth separates
threefold (0.519 against 0.165) where the coefficient of variation managed 0.665.

**But the precision wall is structural, not a feature deficiency.** With 24 targets
against 858 healthy masks, the top 2% of the healthy population alone outnumbers
every target, so even AUC 0.809 yields no net-positive operating point. Any
merge-splitting rule must either accept a much smaller recall at very high
specificity, or be gated by something that first shrinks the candidate set.

### Corrections accepted from the independent assessment (2026-09-16)

1. **The geodesic rejection argument was invalid.** exp110 concluded that a path is
   broken because every ray crosses a pixel of zero learned support. With a finite
   cost `c(x) = alpha*(1-P(x)) + ...` that pixel is expensive, not impassable, and
   Dijkstra crosses it when competing paths cost more. Only infinite cost or
   forbidden traversal blocks a path. The correct quantity is a real minimum path
   cost `C*(p)`, which exp112 now computes.
2. **Status accounting.** Of the sixteen approaches, three were tested directly,
   three partially, and ten were never implemented. Demotion by a shared
   precondition is not refutation, and this log should not read as if it were.
3. Two places where the assessment and this log differ, stated for the record:
   orientation *was* measured in exp109 (structure-tensor ridge direction against
   the mask axis inside the cone, AUC 0.532, near median 0.736 against healthy
   0.720) though never as a gate along a path, which remains open; and the
   width-profile follow-up it lists as queued is the exp111 entry above.

The assessment's recommended direction is the one exp112 takes: build the cost
field from raw absorption and structure-tensor orientation rather than the learned
field, because exp110 showed the network is quieter ahead of truncations (0.0082)
than ahead of healthy masks (0.0137) while raw absorption separates them at 0.611.

### exp112: the geodesic test done correctly, full sample (2026-09-16 12:27 UTC)

Answers the assessment's technical correction. Cost is finite everywhere -
`alpha*(1-dark) + beta*(1-orientation) + step` - so a low-evidence pixel is
expensive rather than impassable, and Dijkstra crosses it when competing paths cost
more. The field is built from the raw photograph (absorption against a sigma-25
background, structure-tensor ridge alignment), not the learned field, because
exp110 showed the network is quieter ahead of truncations than ahead of healthy
masks. Minimum cost of reaching 20 px ahead within a 45-degree cone, full
population: 56 near-threshold, 858 healthy.

| feature | AUC | negated | near median | healthy median | best operating point |
|---|---|---|---|---|---|
| truth (LABEL) | 0.640 | - | 0.2929 | 0.1000 | caught 8, damaged 30, precision 0.210, net -22 |
| cost | 0.441 | **0.559** | 1.1065 | 1.1608 | caught 1, damaged 18, net -17 |
| dark | 0.605 | - | 0.1857 | 0.1366 | caught 2, damaged 18, net -16 |
| orientation | 0.518 | - | 0.8738 | 0.8656 | caught 0, damaged 18, net -18 |

The label separates, so the path search works. Truncated masks do have cheaper
(0.559 negated) and darker (0.605) continuations - the hypothesis's direction - but
at the 98th percentile of the healthy population the cost feature catches 1 of 56
while damaging 18.

**Scope of this result, corrected against the testing manual.** exp112 tested a
*trigger statistic* - the minimum path cost - and nothing else. It never converted
a selected path into mask area, never reran the official scorer, and therefore
never tested the geodesic *operator*. The manual lists that exact mistake as an
invalid conclusion: "evaluating centreline overlap but not reconstructed mask PQ".
So the correct status for method 1 is **partially tested**: the cost statistic is a
weak discriminator, while the repair itself - path plus width reconstruction plus
official rescoring - remains untested, as do its required controls (random-direction
path of equal area, straight tangent extension, matched-area dilation).

Process error worth recording: the ETA given for this run was ~50 minutes; it
finished in six. The per-instance cost was extrapolated from a bounded run whose
early photographs were unrepresentative, and no progress output existed to correct
the estimate mid-run. Long audits should print progress.

**Where the deterministic line stands - corrected statement.** Several
instance-level *triggers* have been audited and are not deployable globally:
width mean_slope 0.809, neck rho 0.712, dark absorption 0.605 and geodesic path
cost 0.559 all carry ranking signal yet produce net-negative operating points,
because 56 targets sit against 858 healthy masks and the top 2% of the healthy
population outnumbers every target.

That is a statement about triggers, not about repairs, and the earlier phrasing
that the post-processing line is "fully audited" or "exhausted" was wrong. This
log's own accounting contradicts it: of the sixteen approaches, three were tested
directly, three partially, and ten were never implemented. A failed trigger does
not refute the repair it was meant to gate.

The correct next step, from the user's testing manual: stop searching for another
universal classifier. For each remaining repair, disable the deployability
question, give it an ORACLE gate on validation, and measure only its maximum
recoverable PQ. Rank the repairs by that ceiling. Kill the ones whose perfect
selector gains nothing; build a deployable gate only for repairs whose oracle gain
is material. Two reservoirs are separate and must be costed separately: detection
headroom (284 of 465 misses have zero overlap with any prediction, unreachable by
mask repair) and existing-instance repair headroom.

### exp113: oracle ceilings for the split and union repair families (2026-09-16 12:59 UTC)

First application of the testing manual's Appendix B step 1: measure a repair
family's maximum recoverable PQ before spending any effort on a trigger for it.
Both operators here READ GROUND TRUTH and are upper bounds only. Neither is a
postprocessor and neither may be submitted.

Baseline recomputed from source artifacts and asserted before anything ran:
PQ 0.4558638687, TP/FP/FN 860/397/465.

| oracle repair | delta PQ | PQ after | TP | FP | FN | crossings up | down |
|---|---|---|---|---|---|---|---|
| **split_by_truth** | **+0.022342** | 0.4782063111 | 900 | 386 | 425 | 40 | 0 |
| union_fragments | +0.011579 | 0.4674425105 | 874 | 354 | 451 | 14 | 0 |

The split operator divided 28 merged predictions into 57 pieces; the union
operator merged 26 fragment groups.

**Both families pass step 1, and the split family is large.** +0.022342 is roughly
twice the total gain from all ensemble and training work this session (+0.012282
for cross-architecture pooling), and it reaches 0.4782 - above the 0.472 that a
printed 0.40 requires.

A predicted failure mode did not occur. Splitting increases the prediction count,
so false positives were expected to rise; instead they fell from 397 to 386,
because each rescued truth converts an FP and an FN into a TP at once. Zero
downward crossings in both repairs, so neither ceiling is bought by damaging
healthy objects.

**What this corrects about earlier reasoning.** exp105c rejected the narrow-neck
splitter on trigger precision (3-7%, net -207 objects) and exp111 rejected the
width-profile gate the same way, both *before* the family's ceiling was known.
That is the sequencing error the manual names: a failed trigger does not refute a
repair. With the ceiling now measured at +0.0223, the gate question is worth real
effort, and the earlier trigger results become evidence about which gate is needed
rather than grounds for abandoning the family.

Still absent: any deployable method. An oracle ceiling is not a submission
candidate, and no slot should be spent on one.

### exp113b: growth ceilings, and a correction to exp108's operational reading (2026-09-16 13:19 UTC)

Appendix B step 1 completed across all four repair families. Every operator reads
ground truth and is an upper bound only. Baseline re-asserted first:
PQ 0.4558638687, TP/FP/FN 860/397/465.

| oracle repair | delta PQ | PQ after | TP | FP | FN | masks touched | pixels added | crossings up/down |
|---|---|---|---|---|---|---|---|---|
| **growth_full** (absolute bound) | **+0.032510** | 0.4883737 | 909 | 348 | 416 | 49 | 56,352 | 49 / 0 |
| split_by_truth | +0.022342 | 0.4782063 | 900 | 386 | 425 | 28 | - | 40 / 0 |
| growth_lateral | +0.018865 | 0.4747293 | 901 | 356 | 424 | 47 | 15,928 | 41 / 0 |
| growth_axial | +0.018103 | 0.4739665 | 897 | 360 | 428 | 41 | 12,499 | 37 / 0 |
| union_fragments | +0.011579 | 0.4674425 | 874 | 354 | 451 | 26 | - | 14 / 0 |

**Correction to exp108's operational reading.** That entry concluded the
axial/lateral decomposition "settles the algorithm choice" in favour of endpoint
continuation over boundary refinement, because 59% of the missing area is axial.
The ceilings contradict it: lateral recovers **+0.018865** against axial's
**+0.018103**, and does so with 47 masks touched against 41. Area share does not
predict recoverable PQ, because a thin lateral deficit needs only a few pixels per
boundary to push an instance across IoU 0.5, while a distal stub 10-30 px long
needs many. Active contours and graph cuts (manual methods 3 and 4) are therefore
not the secondary family this log called them.

**Additivity.** axial + lateral = +0.036968 against the full-growth bound of
+0.032510, so the two populations overlap and no combined growth operator can
exceed +0.0325 on strict 1:1 near-threshold masks.

**What the ranking implies for effort.** Split leads the single families at
+0.022342 and already has candidate gate features measured: width `mean_slope`
AUC 0.809, saddle depth 0.750, neck rho 0.712. Those were previously treated as
rejections; with the ceiling known they are the starting point for a gate. The
growth families are close behind but need a trigger that no audit has yet found,
and their evidence channels (local endpoint ridge at correct scale and polarity,
orientation with coherence weighting, matched filtering) remain untested per the
manual's method sections 5, 6 and 9.

Every number here is an oracle bound. Nothing in this entry is deployable, and no
submission slot should be spent on any of it.

### exp114: real splitters under a perfect gate recover 13% of the ceiling (2026-09-16 14:00 UTC)

Appendix B step 4: rasterise the repair and score it officially. exp113 measured
the split ceiling at +0.022342 by assigning pixels with ground truth. This asks
what implementable operators achieve when told exactly which masks to split.

| configuration | delta PQ | share of ceiling | attempted | split | pieces | crossings up/down |
|---|---|---|---|---|---|---|
| neck_cut, oracle gate | **+0.002846** | +12.7% | 28 | 28 | 56 | 22 / 6 |
| watershed, oracle gate | -0.005072 | -22.7% | 28 | 28 | 80 | 15 / 8 |
| neck_cut, every mask | -0.251834 | -1127% | 1257 | 1213 | 2426 | 33 / 276 |
| watershed, every mask | -0.410318 | -1837% | 1257 | 1248 | 3538 | 22 / 691 |

**Cut placement is now the binding problem, not gating.** Given a perfect gate the
best operator recovers an eighth of the ceiling. Truth assignment turns the same 28
masks into 57 pieces and gains 40 crossings with none lost; the neck cut makes 56
pieces and gains 22 while losing 6. So the operator splits in the wrong place
roughly half the time, and sometimes destroys a healthy match doing it.

Watershed is worse than useless as configured: 80 pieces from 28 masks, about 2.9
each, so a marker limit of 3 with a 0.45 floor over-segments filaments whose median
half-width is 2.2 px. It did not decline on a single mask, which disproves the
concern that the marker constants were too strict.

The ungated columns quantify what an absent gate costs: watershed splits 1,248 of
1,257 predictions and PQ collapses from 0.4559 to 0.0455. Neither operator contains
any judgement about whether a mask should be split; they split whatever is handed
to them.

Per the manual's language rules, these are **tested formulations that are
negative**, not a refuted family. What is ruled out: this marker rule at these
constants, and a cut perpendicular to the principal axis at the narrowest interior
bin. What remains open: fewer markers, orientation-aware cut placement, and cuts
guided by the local interface between two width maxima rather than by the global
axis.

### exp115: tuning the split constants does not fix placement (2026-09-16 14:10 UTC)

Population built once and reused, so each variant costs seconds rather than the
229 s of a rebuild. All configurations use the oracle gate, so only operator
quality is measured. Ceiling +0.022342, 28 oracle merges, 40 available rescues.

**Placement diagnosis, the primary result:**

| operator | oracle rescues available | operator rescues |
|---|---|---|
| neck cut | 40 | **22** |
| watershed, limit 2 | 40 | **18** |

**Constant sweep, secondary:**

| variant | delta PQ | share of ceiling | pieces | up | down |
|---|---|---|---|---|---|
| neck_cut (untuned) | **+0.002846** | +12.7% | 56 | 22 | 6 |
| watershed limit 2, sep 16, floor 0.75 | +0.001938 | +8.7% | 46 | 16 | 5 |
| watershed limit 2, sep 12, floor 0.65 | +0.000407 | +1.8% | 56 | 17 | 7 |
| watershed limit 2, sep 8, floor 0.55 | +0.000374 | +1.7% | 54 | 14 | 4 |
| watershed limit 3, floor 0.45 | -0.005072 | -22.7% | 80 | 15 | 8 |

Tightening the marker rule moved pieces per mask from 2.9 to 1.6 and turned
watershed from clearly harmful to marginally positive, but the rescue count never
rose above 17. **The number of cuts was never the problem; their position is.**
Four settings spanning a wide range of separations and floors all land far below
the untuned neck cut, and every one of them far below the oracle's 40.

Why the geometry is wrong in both operators: the neck cut severs perpendicular to
the mask's global principal axis, which cannot follow the interface of a curved
merge, and watershed floods from distance-transform maxima that sit on the thickest
part of each filament rather than on the boundary between them. Truth assignment
wins its 40 crossings precisely because it cuts along the real interface.

Next operator to test, implied directly by this: a minimum-cost path through the
mask that separates the two width maxima - a curved cut following the local width
minimum - rather than a straight cut across the global axis. Same oracle gate, same
official scoring, so the comparison stays like for like.

### exp116: the interface cut fails, and two of my diagnoses were wrong (2026-09-16 14:19 UTC)

exp115 concluded that split operators fail on cut *placement*, so this operator
carved along the cheapest separating path between the two width maxima - a curved
cut following the local width minimum, which is geometrically what truth assignment
does.

First diagnosis on six real merges: 1 split, 5 declines, with the stages reported
as `no_path` 3 and `carve_no_disconnect` 2. I attributed the first to a
boundary-only target set and the second to 8-connected labelling surviving a 1 px
carve, and fixed both.

**Both attributions were wrong, and the re-diagnosis says so:**

| stage | count |
|---|---|
| no_second_marker | 0 |
| no_path_boundary | 3 |
| no_path_halfplane | **3** |
| carve_4conn_fail | 2 |
| carve_8conn_would_work | **0** |
| reached_pieces | 1 |

Every path that failed against the boundary target also failed against the far
half-plane, so the target set was never the obstacle. And no carve that failed
under 4-connectivity would have succeeded under 8-connectivity, so the connectivity
change addressed nothing. The single path that did run was **6 pixels long** - that
is nicking a corner, not separating two filaments.

The piece cap did work: the one firing now returns 2 pieces rather than 3.

**Scope of the negative.** Three operator geometries have now been implemented and
scored under a perfect gate:

| operator | rescues of available | best delta PQ |
|---|---|---|
| neck cut | 22 of 40 | +0.002846 |
| watershed (best of 4 settings) | 18 of 40 | +0.001938 |
| interface cut | 1 of 6 (sample) | not swept |

This is an operator-placement failure across three distinct formulations, not a
tuning failure and not a gate failure. The split family keeps its +0.022342
ceiling; I have no implementation that approaches it.

**Redirect.** The growth families have ceilings of +0.018865 (lateral) and
+0.018103 (axial) and no operator has been written for either. A first attempt
there is worth more than a fourth splitter, and lateral growth is a narrow-band
operation with no interface to find.

### exp117: growth operators lose to the control that reads no image at all (2026-09-16 17:39 UTC)

First operators written against the growth ceilings, run under the same protocol
the splitters faced: oracle gate, matched control, official metric, harm
accounting. The gate is strict 1:1 masks whose truth IoU lies in [0.4, 0.5).

The ceilings reproduced exactly, which validates the harness:

| ceiling | delta PQ | masks | up | down |
|---|---|---|---|---|
| lateral | +0.018865 | 47 | 41 | 0 |
| axial | +0.018103 | 41 | 37 | 0 |
| full growth | +0.032510 | 49 | 49 | 0 |

**The result contradicts the hypothesis the operators were built on.**

| operator | delta PQ | share of ceiling | up | down |
|---|---|---|---|---|
| **control: dilate 2 iterations** | **+0.006350** | **33.7%** | 15 | 0 |
| control: dilate 1 iteration | +0.004548 | 24.1% | 11 | 0 |
| band grow b4 rel 0.40 | +0.003283 | 17.4% | 8 | 0 |
| band grow b3 rel 0.55 | +0.003218 | 17.1% | 8 | 0 |
| band grow b6 rel 0.55 cap 1.0 | +0.002810 | 14.9% | 7 | 0 |
| band grow b2 rel 0.70 | +0.001980 | 10.5% | 8 | 0 |

Uniform dilation reads no image evidence whatsoever and beats every
absorption-guided variant by roughly a factor of two. The guided operator's
selectivity is not buying accuracy, it is refusing pixels that dilation correctly
adds. Absorption thresholding, relative or otherwise, is a tested negative as a
lateral growth rule. exp103 reached the same verdict for contrast as a proposal
stream, by a different route.

**The axial operator fires now, and still earns nothing.** The first run's direction
test averaged absorption over the whole reach, so a ray that correctly left the
filament end was rejected for being mostly background; it fired on 11-27 of 55
masks. Selecting direction on a short probe fixed that, and the re-run supersedes
those rows:

| ray variant | delta PQ | share of ceiling | fired | up |
|---|---|---|---|---|
| r40 rel 0.35 | +0.001204 | 6.7% | 40/55 | 3 |
| r24 rel 0.50 probe 3 | +0.001202 | 6.6% | 42/55 | 3 |
| r24 rel 0.50 | +0.000809 | 4.5% | 32/55 | 2 |
| r40 rel 0.50 probe 10 | +0.000809 | 4.5% | 28/55 | 2 |
| r40 rel 0.50 cone 25 | +0.000390 | 2.2% | 27/55 | 1 |

Firing more than doubled, from 19/27/11 to 32/40/27, and delta PQ moved by at most
0.0004. Probe length is not a lever either: 3, 6 and 10 all land in the same band.
An operator whose output is flat while its firing rate doubles is not mistuned, so
axial growth joins the splitters as a tested negative at the operator level. That
converges with exp109, exp110 and exp112, which found nothing ahead of the endpoint
to walk along in the first place.

**Determinism check (CLAUDE.md section 5).** Every non-ray row reproduced
byte-identically across the two runs, as did all three ceilings. `band+ray combined`
moved +0.003217 to +0.003224, which is correct - it contains the operator that
changed. Nothing else moved, so the re-run altered only what it was meant to.

**The structural finding, which outlives every variant above.** Compare the same
operator gated and ungated:

| configuration | delta PQ | up | down | masks changed |
|---|---|---|---|---|
| dilate 1, oracle gate | +0.004548 | 11 | 0 | 55 |
| dilate 1, no gate | -0.026434 | 11 | 43 | 953 |
| dilate 2, no gate | -0.080075 | 16 | 117 | 953 |

The crossings *up* are identical, 11 either way. Every gain comes from the 55
near-band masks and every harm from the other 898. The repair's benefit is fixed
and the gate's whole job is damage prevention, so a deployable gate must reach
**TPR/FPR > 43/11 = 3.9** merely to break even.

No measured statistic can be quoted against that bar yet, and the temptation to
quote one must be named. `mean_slope` reaches AUC 0.809 in exp111, but that audit's
positive class is oracle-splittable **merges** against healthy 1:1 masks
(`audit_width_profile.py:162-175`). The growth gate asks a different question -
near-threshold 1:1 masks (IoU 0.4-0.5) against healthy 1:1 masks - and 0.809 says
nothing about it. Carrying the number across would be CLAUDE.md section 10 exactly.
The prize is in any case a fraction of +0.006350.

The ungated -0.026434 independently reproduces exp104's -0.027 through a separate
code path, which is a useful check on both.

**Scope.** This closes absorption-guided lateral growth, not the lateral family:
its +0.018865 ceiling stands, and dilation still recovers a third of it under a
perfect gate. What is now measured twice, across two families, is that
implementable operators recover a small share of their ceiling, and that the
remaining postprocessing headroom is far below the ~+0.016 validation gain needed
to print 0.40 on the public board.

#### exp117 addendum: the growth repair has no trigger, and that was already measured

The +0.006350 above needs a gate reaching TPR/FPR > 3.9. Before building one I
checked what this log already contains, because the gate's population - strict 1:1
masks with IoU in [0.4, 0.5) against healthy 1:1 masks - is the exact population
exp105b through exp107 audited.

Every candidate discriminator has been measured on it and rejected on precision at
realistic operating points, not on AUC:

| trigger family | best AUC | best operating point | source |
|---|---|---|---|
| geometry marginals (elongation, solidity, width CV) | 0.539 | within 0.08 of chance | exp105c |
| fixed geometry combination | 0.548 | within 0.08 of chance | exp105c |
| narrow neck (rho) | 0.712 (merges) | precision 0.062, net -28 | exp105c |
| semantic + spine annulus, 5 variants | 0.543 | `sem_exterior` 0.397, net -165 | exp106 |
| cross-architecture disagreement, both directions | 0.595 | net -35; zero of 56 at q0.98 | exp107 |

The best honest number for this question is around 0.635, from `sem_interior`
negated. Against a break-even bar of 3.9 that yields approximately nothing.

**So the growth family closes at the gate, exactly as the split family closed at
the operator.** The repair is real and its ceiling is real. Selecting the masks it
should touch is what cannot be done, and nine feature families across four
experiments now say so.

**One convergence is worth stating plainly.** exp106 concluded that a substantial
part of the annotated extent is not photometrically distinguishable in H-alpha, so
no observable field can trigger its restoration. That predicts today's operator
result before it was run: if the image does not mark the missing extent, an
absorption-guided operator cannot find it, which is precisely why uniform dilation
beat every guided variant. Two independent routes, one conclusion.

**Consequence for the >0.46 goal.** Postprocessing repair is now measured end to
end across both families that had material ceilings:

| family | oracle ceiling | best real operator (oracle gate) | deployable gate |
|---|---|---|---|
| split | +0.022342 | +0.002846 | not built; 3 geometries failed at placement |
| growth | +0.018865 / +0.018103 | +0.006350 (uniform dilation) | measured unavailable |

Validation stands at 0.4559 and needs about 0.472 to print 0.40 publicly, or about
0.53 for the user's 0.46 target. This class of repair cannot supply that. The
largest known unrealized gain remains the cross-architecture ensemble at 0.468146
validation (+0.012282), which is unsubmitted, with three submission slots left.

#### exp117 addendum 2: where the leverage actually is, costed before any run

CLAUDE.md section 9 says cost the sweep first. Baseline is 860 TP / 397 FP / 465 FN,
PQ 0.4558638687, SQ 0.6843, RQ 0.6662, matched IoU sum 588.52.

Converting misses into matches, PQ after N recoveries:

| N recovered | at IoU 0.60 | at IoU 0.70 | at IoU 0.80 |
|---|---|---|---|
| 25 | 0.4630 | 0.4649 | 0.4668 |
| 50 | 0.4700 | 0.4738 | 0.4776 |
| 100 | 0.4836 | 0.4911 | 0.4985 |
| 200 | 0.5094 | 0.5237 | 0.5381 |
| 284 | 0.5296 | 0.5494 | 0.5692 |

Suppressing pure false positives, PQ after N removals: 50 -> 0.4649, 100 -> 0.4742,
200 -> 0.4941.

**At IoU 0.70 per recovery, 0.472 needs 45 of 465 misses and 0.530 needs 221.**

That reframes the whole effort. Repair polishes IoU on masks that already match, so
it moves the numerator only. Recovering a miss moves the numerator *and* converts an
FN into a TP, which is why 45 recoveries beat every repair result measured today.
For comparison, the two repair families together yield under +0.009 under perfect
gates, and near zero deployed.

**Two cautions against reading this as easy.**

1. The suppression column assumes the removed objects are purely false. exp062
   already swept confidence, erosion and area widely, so the cheap precision is
   spent. Removing further FPs without removing TPs is the same selection problem
   that just closed the growth gate, and it needs its own measured discriminator
   before any of that column is claimed.
2. exp088 found 284 of the 465 misses have zero overlap with any prediction, and
   exp103 found contrast alone does not propose them. So recall is a detection
   problem, not a repair problem, and the 45 recoveries must come from a detector
   or a pipeline stage rather than from post-processing existing masks.

This is consistent with the user's standing direction: build a pipeline to decision
rather than loading one model with the whole task.

### exp118: the discarded candidates hold +0.020822, and the bar is finally close (2026-09-16 18:10 UTC)

`scripts/run_recall_ceiling.py`. exp117 addendum 2 costed recall as worth more than
repair. This measures whether the recoveries already sit in the candidate pool,
discarded by the confidence floor.

The pool holds 1414 candidates over 106 photographs, 813 above the 0.35 floor and
**601 in [0.10, 0.35)**.

**Lowering the floor blind, with no verifier:**

| floor | PQ | TP | FP | FN | added TP | added FP | precision of added |
|---|---|---|---|---|---|---|---|
| 0.35 | 0.455864 | 860 | 397 | 465 | +0 | +0 | - |
| 0.30 | 0.450091 | 892 | 492 | 433 | +32 | +95 | 0.252 |
| 0.25 | 0.442127 | 923 | 602 | 402 | +63 | +205 | 0.235 |
| 0.20 | 0.425956 | 949 | 762 | 376 | +89 | +365 | 0.196 |
| 0.15 | 0.402681 | 966 | 976 | 359 | +106 | +579 | 0.155 |
| 0.10 | 0.396853 | 969 | 1031 | 356 | +109 | +634 | 0.147 |

Every floor loses, which is why exp062 picked 0.35. That is the control.

**ORACLE admit (never deployable): PQ 0.476686, +0.020822.** 132 candidates
admitted, TP 980 / FP 492 / FN 345, recovering **120 misses at mean IoU 0.651**.

Two things make this the best target measured this session.

1. **It is bigger than either repair family's whole ceiling** (+0.022342 split,
   +0.018865 lateral) and unlike them it clears 0.472, the validation level a
   printed 0.40 needs.
2. **The deployable bar is close rather than hopeless.** Break-even is
   k*q > 0.5*PQ*(k+m). At the measured q = 0.651 that is m < 1.86k, so a verifier
   needs about **35% precision**. Blind floor-lowering already achieves 23.5-25.2%.
   The gap is a factor of 1.4. Every repair gate measured this session was wrong by
   a factor of four or more, and nine feature families could not close those. This
   one asks a verifier to sort 24% precision up to 35%.

**Scope, stated so it is not over-read.** The oracle admits only candidates the
detector already proposed at floor 0.25 per view. It is therefore the ceiling for
*re-admitting discarded candidates*, not a ceiling for recall in general. exp088's
284 zero-overlap misses are largely outside its reach and need a proposal stream,
not a verifier. And 120 recovered is oracle selection; a real verifier yields less.

#### exp118 addendum: the precision target is 50%, not 35%

Working the break-even formula into actual operating points corrects the figure I
first wrote. 35% is where the gain reaches **zero**, not where it becomes useful:

| candidates kept | verifier precision | PQ | vs baseline |
|---|---|---|---|
| 120 | 1.00 (oracle) | 0.493442 | +0.037578 |
| 120 | 0.50 | **0.472459** | +0.016596 |
| 120 | 0.40 | 0.462623 | +0.006759 |
| 120 | 0.35 | 0.455845 | **-0.000019** |
| 80 | 0.50 | 0.467250 | +0.011386 |
| 60 | 0.60 | 0.467994 | +0.012130 |
| 40 | 0.70 | 0.465727 | +0.009863 |

So the real requirement is about **50% precision at k around 120**, which is the
only row that clears 0.472. Blind floor-lowering gives 0.235-0.252, so a verifier
must roughly **double** candidate precision, not improve it by a factor of 1.4.

Required precision as a function of recovered-mask quality, for the record:

| mean IoU of recoveries | max false per true | minimum precision |
|---|---|---|
| 0.550 | 1.41 | 0.414 |
| 0.600 | 1.63 | 0.380 |
| 0.651 (measured) | 1.86 | 0.350 |
| 0.700 | 2.07 | 0.326 |

That is a harder target than the first reading suggested, and it is still the
closest deployable bar this session has produced.

### exp120: no single feature verifies a discarded candidate (2026-09-16 18:32 UTC)

`scripts/audit_candidate_verifier.py`. Population: the 526 candidates in
[0.10, 0.35) that survive erosion and the 300 px area floor, of which **132 are
admissible** by exp118's oracle. Base precision **0.2510**, break-even 0.350,
target 0.500.

| feature | AUC | best flipped | best operating point (precision, kept/admitted) |
|---|---|---|---|
| score | 0.641 | 0.641 | q0.90 **0.396** (21/53) |
| limb | 0.398 | **0.602** | q0.50 0.316 (83/263) |
| half_width | 0.541 | 0.541 | q0.70 0.283 (47/166) |
| contrast | 0.503 | 0.503 | q0.90 0.283 (15/53) |
| absorption | 0.523 | 0.523 | q0.30 0.269 (99/368) |
| area | 0.502 | 0.502 | q0.90 0.273 (15/55) |
| extent | 0.520 | 0.520 | q0.30 0.266 (98/368) |
| solidity | 0.518 | 0.518 | q0.50 0.266 (70/263) |
| elongation | 0.486 | 0.514 | q0.70 0.272 (43/158) |

**No feature reaches 0.500 anywhere.** The best point is `score` at 0.396 on 53
candidates, and that row is largely an artefact: the pool is ordered by score and
the 0.35 floor is itself a score threshold, so ranking by score partly re-derives
the cut that already exists. Every genuinely new feature lands between 0.253 and
0.316, against a blind baseline of 0.251 - that is, near enough to admitting at
random.

Only two features sit meaningfully above chance, and they measure unrelated things:
`score` (detector confidence, 0.641) and `limb` **flipped** (position on the disk,
0.602). Candidates nearer disk centre are likelier to be real, which is physically
sensible - foreshortening and limb darkening degrade both annotation and detection
near the edge.

**This is a single-feature screen and not the family's verdict.** Two things remain
untested, and exp121 does both: the `score` x `limb` combination, and - more
importantly - scoring every operating point by the official metric instead of by
precision. Precision is a surrogate. A point with modest precision but favourable
mask quality can still pay, since the arithmetic gives +0.0099 at k=40, p=0.70.

Population note: exp118 counted 601 candidates in the band and exp120 scores 526.
The 75 dropped are below the 300 px area floor after erosion. All 132 admissible
candidates survive in both, so the positive class is identical.

### exp121: the verifier earns +0.000480, and fragmentation explains why (2026-09-16 18:47 UTC)

`scripts/run_verifier_operating_points.py`. Twelve admission rules, each re-emitted
through the frozen path and rescored by the official metric. The `admit nothing`
row reproduced PQ 0.4558638687 with TP/FP 860/397 exactly, so the feature index is
sound and the table is valid.

| rule | admitted | PQ | delta | TP | FP | up | down |
|---|---|---|---|---|---|---|---|
| admit nothing | 0 | 0.455864 | +0.000000 | 860 | 397 | 0 | 0 |
| **score q0.90 AND limb low q0.70** | 35 | 0.456344 | **+0.000480** | 880 | 432 | 20 | 0 |
| score q0.95 | 27 | 0.455800 | -0.000064 | 875 | 426 | 15 | 0 |
| score q0.80 AND limb low q0.50 | 52 | 0.455318 | -0.000546 | 885 | 447 | 25 | 0 |
| score q0.70 AND limb low q0.50 | 77 | 0.455260 | -0.000603 | 899 | 474 | 39 | 0 |
| score q0.90 | 53 | 0.452568 | -0.003296 | 884 | 462 | 24 | 0 |
| score q0.50 | 263 | 0.431710 | -0.024154 | 942 | 707 | 82 | 0 |
| limb low q0.50 | 263 | 0.430471 | -0.025393 | 928 | 690 | 68 | 0 |

The combination does beat either feature alone, and it is the only positive rule,
but +0.000480 is **2.3% of the +0.020822 ceiling**.

**Crossings down are zero in every row.** Admission never destroys an existing
match, because painting gives contested pixels to the higher-scoring mask and every
admitted candidate scores below 0.35. The entire cost is added false positives.

**Effective precision is 0.364 against the 0.558 the oracle achieves**, which is
why the rule earns almost nothing. The best gate admits candidates that produce
+20 TP and +35 FP in pooled record units, so 20 of 55.

CORRECTION (2026-09-16 19:20 UTC, self). I first wrote here that the excess false
positives were *fragmentation* - admitted masks carved into several instances by
the masks above them - and quoted multipliers of 1.49 to 1.63. Then I proposed
*displacement*, that a new mask steals an existing match. **Both were wrong**, and
exp122 plus a direct count refuted them:

- Fragmentation: exp122 measured instances per admitted candidate at 0.90 to 1.00,
  never 1.6. Painting does not split these masks.
- Displacement: impossible in principle. PQ matches at IoU > 0.5, and above 0.5 the
  matching is unique, since two predictions cannot each cover more than half of one
  truth. An added prediction cannot take a match from an existing one.

The real explanation is a **units error of mine**. Instances were counted per
photograph (106) while TP and FP are pooled per record (180), because every
photograph is scored once per annotation pass. 180/106 = 1.698, and counting the
same admissions per record gives exactly 55, matching the observed TP+FP rise.
There was never an extra cost to explain - only two different denominators.

So the gate remains the binding constraint, exactly as first measured, and nothing
here moves the problem to the operator.

### exp122: admission geometry is not a lever (2026-09-16 19:14 UTC)

`scripts/run_fragment_free_admission.py`. Seven gates x three admission geometries,
each rescored officially, with `admit nothing` reproducing the baseline in all
three.

| gate | whole | largest | all_pieces |
|---|---|---|---|
| score q0.90 + limb q0.70 | **+0.000480** | +0.000022 | -0.000151 |
| score q0.95 alone | -0.000064 | -0.000064 | -0.000064 |
| score q0.80 + limb q0.50 | -0.000546 | -0.000981 | -0.001152 |
| score q0.70 + limb q0.50 | -0.000603 | -0.001032 | -0.001200 |
| score q0.80 + limb q0.70 | -0.002345 | -0.002623 | -0.002960 |
| score q0.90 alone | -0.003296 | -0.003748 | -0.003917 |

Every `whole` row reproduces exp121's delta for the same gate to six decimals, on
all six shared gates. The two paths are the same computation: painting gives
contested pixels to the higher-scoring mask, and every admitted candidate scores
below 0.35, so it receives precisely the leftover that carving computes.

Keeping only the largest component is consistently *worse*, by 0.0004 or so,
because discarding the smaller components discards real filament area. The whole
spread across geometries is under 0.0007, against a ceiling of +0.020822.

**Conclusion: the operator is not the problem and geometry is not a lever.** The
gate is. Best deployable remains +0.000480, 2.3% of the ceiling.

### exp119: the ensemble prints 0.39, as the offset predicted (2026-09-16 19:05 UTC)

Submitted the exp102 cross-architecture ensemble: exp002 D4 + exp097 stage2_best
D4, instance-level fusion at sources=2, link 0.5, vote 0.5, emitted at conf 0.35 /
erosion 1 px / area 100.

The kernel replayed the rule on validation before touching a test photograph and
reproduced **0.46814553** against exp102's 0.468146, with TP/FP/FN 852/328/473
identical to the recorded row (CLAUDE.md section 5). CSV: 1153 rows, 177 of 180
photographs contributing, no empty or duplicate rows, SHA256
FB3DCA02B03D5CFC738BDF67A6F9B7C746D7F073FAF78278A84E0DD270488982.

**Public score 0.39.** That ties the previous best rather than beating it. The
+0.012282 validation gain over baseline did not survive two decimals, which is
exactly what this log already warned about: a local spread of 0.013 is invisible on
a board that prints 0.36 for four genuinely different models. Predicted band was
0.391-0.401 and it landed at the bottom of it.

Three test photographs still receive no prediction at all, which exp103 recorded
for the earlier model too. That is pre-existing, not a regression from fusion.

### exp125: a fitted combination triples the gain, and is still 8% of the ceiling (2026-09-16 19:52 UTC)

`scripts/run_learned_verifier.py`. Nine features, 526 discarded candidates, 132
admissible, base rate 0.2510, over 97 photographs that carry any candidate in the
band. Predictions are **out-of-fold** under `GroupKFold` grouped by photograph, so
no candidate is scored by a model that saw another candidate from the same
photograph.

| model | top k | admitted (records) | PQ | delta | up | precision |
|---|---|---|---|---|---|---|
| **logistic** | **20** | 32 | 0.457562 | **+0.001698** | 15 | 0.469 |
| logistic | 40 | 69 | 0.454984 | -0.000880 | 23 | 0.333 |
| logistic | 60 | 98 | 0.453859 | -0.002005 | 30 | 0.306 |
| logistic | 100 | 149 | 0.450803 | -0.005061 | 41 | 0.275 |
| gradient boosting | 20 | 25 | 0.452576 | -0.003288 | 2 | 0.080 |
| gradient boosting | 40 | 51 | 0.450834 | -0.005030 | 7 | 0.137 |
| gradient boosting | 60 | 79 | 0.453589 | -0.002275 | 22 | 0.278 |
| gradient boosting | 100 | 126 | 0.452999 | -0.002865 | 37 | 0.294 |

**Logistic top-20 is +0.001698, three and a half times exp121's hand-drawn pair.**
So the nine features do carry slightly more jointly than `score` and `limb` carry
together. It remains **8.2% of the +0.020822 ceiling**, and its precision of 0.469
is still short of the 0.558 the oracle reaches.

**Gradient boosting fails, and fails informatively.** Every row is negative, and at
top-20 its precision is **0.200 - below the 0.251 base rate**, meaning it is
anti-correlated exactly where it is most confident. With 132 positives over nine
features, 120 depth-2 trees memorise the training folds. This is a property of the
dataset size, not of the hyperparameters.

**Caveat, stated rather than buried.** The out-of-fold predictions are honest, but
k=20 was chosen after seeing the official metric across eight configurations (two
models x four k). Picking the best of eight on the same validation split is
optimistic, so the true held-out gain is below +0.001698. The number is a ceiling
on what this approach delivers, not an unbiased estimate of it.

**Not submission-worthy.** 0.455864 + 0.001698 = 0.457562, which predicts 0.380 to
0.391 publicly against a current best of 0.39. Stacked on the ensemble it would
still most likely print 0.39.

Memory note: the first attempt decoded every candidate of every photograph at once,
about 1414 dense 2048x2048 masks, and the run was killed with 6.1 GiB free. The
rewrite caches RLEs only, builds the claimed-pixel map with `mu.merge`, and decodes
one candidate at a time.

### exp126: isolation features are noise, and I am not counting the gain (2026-09-16 20:06 UTC)

`scripts/run_isolation_verifier.py`. Added two relational features to exp125's
nine: `claimed_fraction` (share of the candidate already emitted) and `nearest_gap`
(distance to the nearest emitted instance).

| feature | AUC | flipped | positive median | negative median |
|---|---|---|---|---|
| claimed_fraction | 0.473 | 0.527 | 0.000 | 0.000 |
| nearest_gap | 0.519 | 0.519 | 112.607 | 107.918 |

Both are chance. `claimed_fraction` is degenerate: its median is 0.000 in **both**
classes, because carving already removes overlapping area and these candidates
barely overlap the emission to begin with.

| feature set | top k | admitted | delta PQ | up | precision |
|---|---|---|---|---|---|
| eleven (+isolation) | 20 | 36 | +0.002772 | 19 | 0.528 |
| nine (exp125) | 20 | 32 | +0.001698 | 15 | 0.469 |

**I do not believe the +0.002772.** Three reasons, and they point the same way:
the two added features are individually at chance; the combined out-of-fold AUC
*fell* (0.647 against 0.652); and the entire difference is **four** extra threshold
crossings, 19 against 15, selected as the best of eight configurations scored on
this same split. That is the shape of noise, not of signal. Recorded so the number
is not quoted later as a gain.

### exp124: cross-architecture confirmation is the best feature found, and my grid missed it (2026-09-16 20:11 UTC)

`scripts/audit_cross_architecture_verifier.py`, run as a Kaggle CPU kernel so
exp097's pool never leaves Kaggle. The question is deliberately not exp107's: not
"do the two architectures disagree about this boundary" but **"does exp097
independently propose a filament here at all"**.

| feature | AUC |
|---|---|
| **confirmed** (exp097 proposes an overlap at IoU >= 0.5) | **0.688** |
| joint (score x confirmation) | 0.679 |
| confirm_iou | 0.651 |
| score | 0.641 |
| partner_score | 0.442 (flipped 0.558) |

**0.688 is the strongest single feature measured on this question** - above `score`,
above every geometry and semantic feature from exp105c to exp107, and above the
fitted nine-feature model's out-of-fold 0.652. The physical argument survives its
test: two architectures proposing the same object is evidence, while two
architectures disagreeing about an edge is not.

Every rule in the run is nonetheless negative, best -0.002558 at precision 0.301.
**That is a fault in my operating-point grid, not in the feature.** The score cuts
were quantiles q0.0, q0.5 and q0.8 over all candidates, so the tightest rule still
admitted 123 records, while exp125 and exp126 showed the paying region is 24 to 36.
The feature was never tested where gains are possible.

Rejecting the evidence because the rule I happened to build failed is precisely the
error corrected earlier in this log. exp127 joins these three features to exp120's
nine and sweeps tight operating points.

Defect worth recording: this kernel decodes every base instance for all 106
photographs at once, about 5.3 GB, the same fault that killed exp125 locally. It
survived on Kaggle's larger worker but ran far longer than the minutes estimated.

### exp129: the error decomposed by stage, and my strategic claim was wrong (2026-09-16 21:02 UTC)

Run at the user's direction, after they objected that a major strategic conclusion
had been produced without establishing its premises. They were right.

For every validation truth, the best IoU any candidate in the D4 pool reaches, with
**no confidence floor, no erosion, no area floor and no painting**. That separates
"never proposed" from "proposed but badly bounded", which final PQ cannot.

| bucket | count | share of 1325 |
|---|---|---|
| >= 0.5, already matchable | 1016 | 76.7% |
| 0.3-0.5, localized but boundary too poor | **194** | 14.6% |
| 0.1-0.3, partially localized | 50 | 3.8% |
| < 0.1, never proposed | **65** | 4.9% |

**The 465 misses break down as: 156 emission failures, 194 boundary failures, 50
weak proposals, 65 localization failures.**

| oracle fix | PQ | delta |
|---|---|---|
| emission only | 0.5057 | +0.0498 |
| boundary only | 0.5149 | +0.0590 |
| both | **0.5566** | +0.1007 |

**Two claims of mine are retracted.**

1. *"The problem is entirely detection."* False. Pure localization failure is the
   **smallest** bucket at 65 truths. Boundary estimation is the largest at 194.
2. *"284 misses have zero overlap with any prediction."* That figure describes the
   **emitted** set at conf 0.35, not the candidate pool. At pool level only 65
   truths are genuinely unproposed. I quoted a number produced downstream of the
   very rule that was discarding the evidence, then reasoned about the detector
   from it.

A third claim is also withdrawn: that our boundary quality already matches the
published method, from SQ 0.684 against their per-instance IoU 0.70. SQ is computed
**only over matched true positives**, so it describes the objects we already found
and says nothing about the 465 we missed, which are plausibly fainter and thinner.
That was a filtered population compared against an unfiltered one.

Mean best-IoU of the 0.3-0.5 group is 0.428, so these proposals sit a long way below
threshold rather than just under it. The band is wide, not marginal.

### exp130: the emission losses are the confidence floor, not our own erosion (2026-09-16 21:14 UTC)

Attribution of the 203 truths that have a proposal above IoU 0.5 yet go unmatched:

| cause | count | share |
|---|---|---|
| **below the 0.35 confidence floor** | **168** | **82.8%** |
| dropped by the 300 px area floor | 13 | 6.4% |
| destroyed by the 1 px erosion | 12 | 5.9% |
| pixels lost to a higher-scoring mask | 10 | 4.9% |

My thin-filament suspicion was specific, testable and **false**. Erosion kills 12,
not the bulk, and the median truth area of these misses is 932 px against a 300 px
floor. A size-conditional emission rule could recover at most 25 truths.

"Below the confidence floor" means the detector found these filaments and scored
them under 0.35, so admitting them is the sub-threshold verifier problem of
exp118-127, whose deployable rules returned about +0.002. **H2 is abandoned.**

Count note: exp129's 156 counts truths whose best **raw** proposal clears 0.5;
exp130's 203 counts those unmatched in the final emission, which also includes
truths lost to another annotation pass's painting. Different measurements.

### exp131: raw proposals are too FAT, and that inverts an earlier finding (2026-09-16 21:22 UTC)

Geometry of the 194 truths whose best raw proposal sits at IoU 0.3-0.5:

| quantity | value |
|---|---|
| mean best IoU | 0.428 |
| median excess / intersection | **0.919** |
| median missing / intersection | 0.316 |
| proposals too large | **58.8%** |

| shape operator | crosses IoU 0.5 | share |
|---|---|---|
| **erode 1** | **56** | **28.9%** |
| dilate 3 | 11 | 5.7% |
| dilate 2 | 10 | 5.2% |
| dilate 1 | 9 | 4.6% |
| close 5x5 | 0 | 0.0% |

**This inverts exp105b.** That audit found near-threshold masks *too small* (missing
0.76, excess 0.18) and motivated the entire growth-repair line, exp113 through
exp117. The populations differ: exp105b measured masks the pipeline had already
eroded 1 px, while these are raw D4 proposals. Raw fusion output is systematically
fat, and the existing erosion is already correcting it.

**Consequence: the 194 overstates the refiner's target.** exp129 bucketed by raw
IoU, but the pipeline works on eroded masks, so a truth whose raw proposal is
0.3-0.5 but clears 0.5 after erosion is a confidence-floor loss, not a boundary
failure. exp132 re-decomposes in the representation the pipeline actually uses,
before any GPU is spent on a boundary model.

### exp132: the corrected decomposition, and the vote knob is dead (2026-09-16 21:34 UTC)

exp129 bucketed truths by their best **raw** proposal, but the pipeline works on
masks eroded 1 px. Re-bucketed in the representation the pipeline actually sees:

| bucket | count | share of 1325 |
|---|---|---|
| matchable with the current rule | 880 | 66.4% |
| lost to the 0.35 confidence floor | **172** | 13.0% |
| **true boundary failure (0.3-0.5 eroded)** | **165** | 12.5% |
| weak proposal 0.1-0.3 | 45 | 3.4% |
| never proposed < 0.1 | 63 | 4.8% |

**Boundary failures are 165, not the 194 exp129 claimed.** Erosion lifts proposal
recall from 76.7% to **79.4%**, confirming exp131's fat-proposal finding at
population level: our own erosion is a boundary correction that already works. The
pipeline then surrenders 13 points to the confidence floor, 79.4% down to 66.4%,
which is 172 truths - a larger bucket than the boundary one, and already abandoned
as H2.

**The pixel-vote knob does not help.**

| pool (link_vote) | raw | eroded | eroded + conf |
|---|---|---|---|
| 0.5_0.5 (current) | 76.7% | **79.4%** | **66.4%** |
| 0.5_0.7 (stricter vote) | 78.4% | 78.0% | 65.2% |
| 0.4_0.5 | 75.5% | 78.0% | 66.9% |

A stricter vote does produce leaner raw masks - raw recall rises to 78.4% - but
after erosion it is *worse* than the current pool. The two corrections are
substitutes, not additive, and the whole spread is under 1.7 points with no variant
dominating. There is no free win here.

**Revised H1 arithmetic.** 165 truths, converting 40% at IoU 0.65, gives about
+0.021 PQ, so validation ~0.477 and public 0.40-0.41. That still clears the
pre-registered continue threshold, but the margin is thinner than the +0.059 the
raw bucketing implied, and exp092 remains the only prior boundary attempt - it
scored +0.001 when measured at native resolution.

### exp134: a CNN on the raw pixels cannot rescore either (2026-09-16 22:05 UTC)

The last untried instrument against the confidence-floor bucket. exp118-127 used
hand-crafted features; this shows a model the actual image.

**Setup.** exp091 had already harvested all 601 training photographs on a T4
(7,442 proposals, parity 1.0), so this cost **zero GPU**. Crops are 128 px, two
channels (photograph and proposed mask), from `exp_025_verifier`'s extractor.

**The labelling bug that killed exp025 is fixed here.** CLAUDE.md section 4: it
labelled a crop positive if *any* annotator drew a filament there, scoring 72% on
that question and 17% on the one PQ asks. Here the unit is
**(photograph, record, proposal)** - positive when the proposal matches *that
record's* truth at IoU >= 0.5. Train 5,281 examples at 19.3% positive, validation
1,016 at 18.1%, which are comparable per-record rates.

**Ceiling, measured not assumed.** 168 of the 172 bucket truths are reachable by a
sub-threshold proposal, and all 168 clear IoU 0.5 on the *raw* mask the crops
carry, so the classifier is not handicapped by the representation. Full recovery
would be PQ 0.5038. exp076 independently found the same 168 on 2026-09-15.

**The model did not learn.** Held AP 0.19-0.25 across 20 epochs against a 17.8%
base rate, with no trend. The loss never left **1.11**, which is exactly where a
random classifier sits at this class balance and positive weight (mean sample
weight 1.60 x ln2). "Best epoch 20 at AP 0.2524" is the maximum of 20 noisy draws.

**Official PQ - every rule that admits anything loses:**

| rule | admitted | PQ | delta | TP | FP | precision |
|---|---|---|---|---|---|---|
| **admit nothing** | 0 | 0.455864 | **+0.000000** | 860 | 397 | - |
| global top 150 | 13 | 0.454144 | -0.001719 | 861 | 409 | 0.077 |
| per photograph <=3 at p>=0.5 | 28 | 0.452631 | -0.003233 | 863 | 422 | 0.107 |
| any rule at p>=0.7 | 0 | 0.455864 | +0.000000 | 860 | 397 | - |

**0.0% of the ceiling.** Break-even needs ~35% precision; the best rule reached
10.7%. Every rule at p >= 0.7 admitted nothing at all - the model never emits a
confident probability on any validation crop.

**The crops were not the problem, which is what makes this conclusive.**

| check | value |
|---|---|
| mask occupancy | 1.3%-29.9%, zero degenerate |
| blank images | 0 |
| darkness contrast (outside minus inside) | AUC 0.5345 |
| mask area | AUC 0.4951 |

Well-formed inputs, and two trivial physical features at chance. A real filament
under a low-confidence proposal does not look different, in a 128 px patch, from a
false one.

**Scope.** Three instruments have now failed on this bucket: geometric features
(AUC <= 0.64), cross-architecture confirmation (0.688, unusable at tight operating
points), and a CNN on pixels (chance). The defensible reading is that the
detector's low confidence here is **correct uncertainty**, not a fixable
miscalibration. The 165-truth boundary bucket remains the only measured family
with a real ceiling that no learned model has properly tested.

Result emailed to the user 2026-09-16. Nothing is running.

### exp135-138: boundary precision is capped by annotators, not by resolution (2026-09-17)

Run to decide between a crop-and-zoom refiner and a `mask_ratio 1` retrain, both aimed at
the 165-truth boundary bucket. All four are free CPU measurements; nothing was trained.

**exp135 - the boundary bucket split.** Of 165 boundary failures (exp002 D4 pool), **98**
already have an emitted mask (score >= 0.35) and **67** sit below the floor. Proposal
median half-width 2.80 px, median long side 90 px. Redrawing an emitted mask turns
FP + FN into one TP with the denominator unchanged, about 1.6x an admission.
CAVEAT: sized on exp002's pool. exp119's pool is different and was never measured.

**exp136 - YOLOv8's prototype grid is not binding.** Ground truth pushed through an
area-downsample to a stride-s grid and back (a PERFECT mask head's ceiling):

| grid | mean IoU | thin (<=3 px) | clearing 0.5 |
|---|---|---|---|
| stride 2 (1024) | 0.976 | 0.971 | 100% |
| **stride 4 (512), YOLOv8 prototypes** | **0.914** | **0.896** | **99.9%** |
| stride 8 (256) | 0.771 | 0.713 | 95.8% |

exp097 reaches SQ 0.6877, so output resolution is not the constraint. This FALSIFIED
the rationale I gave for a crop refiner (that zooming escapes a binding grid).

**exp137 - what a `mask_ratio 4` target throws away.** Self-IoU through the 512 grid:
continuous 0.914, binary-cover 0.818, binary-nearest 0.702. `mask_ratio 1` recovers at
most **+0.096** (thin +0.106) against a mask's OWN ground truth.

Loss code, Ultralytics 8.4.0 (`.venv`): `v8SegmentationLoss.loss` lines 495-497 upsample
the PROTOTYPES bilinearly to the target size, so `mask_ratio 1` really computes the loss
at native resolution; the target downsample is commented out. The `mode="nearest"`
downsample at line 309 is in `BCEDiceLoss`, used only by the semantic branch, which does
not run for instance models. exp097 used **8.4.153**, which was NOT verified.

**exp138 - but annotator disagreement swamps it.** 5,816 filament pairs matched across
annotators on 296 multi-annotated training photographs; annotator A scored against B:

| A's mask | mean IoU vs B | pairs clearing 0.5 |
|---|---|---|
| raw, full resolution | 0.576 | 73.8% |
| continuous 512 (`mask_ratio 1` learnable) | 0.577 | 73.2% |
| binary-cover 512 (`mask_ratio 4` target) | 0.564 | 71.2% |

The quantisation cost that survives annotator noise is **+0.013** (thin +0.016, +2.0%
of pairs crossing 0.5). About 86% of exp137's +0.096 is absorbed by disagreement.

**Conclusions.**
1. `mask_ratio 1` is **not funded.** +0.013 is below the >= 0.02 bar fixed before the
   run. It is a ceiling from perfect encodings; exp068/074's native fine-tunes reached
   only parity. The optimistic case lands at the 0.475 line for printing 0.40.
2. **Full-resolution precision buys nothing against a different annotator** (0.576 raw
   vs 0.577 through a smooth 512 grid). Boundary precision beyond a 512 grid is
   worthless under this label noise, so a zoom refiner's precision rationale is dead.
   A refiner could still address GROSS shape error (bucket mean IoU 0.414, 58.8% of
   raw proposals too fat), but six refiners already failed at that.
3. Consistent with transfer: the 1 px erosion reached public at 88% because it fixes a
   gross systematic bias, not fine precision.
4. Annotators drawing the same filament average IoU 0.576, and 26% of filaments both
   drew do not clear the 0.5 match line. Boundary quality is bounded by annotator
   agreement, and exp097 (SQ 0.6877) already sits near that level on matched pairs.

**Correction to exp138 conclusion 4.** "Bounded by annotator agreement" states a
ceiling. The measurement is agreement between annotators (IoU 0.576), not a bound on
a model. HOST_GUIDANCE point 5 and CLAUDE.md forbid treating it as one.

---

## exp139 - a properly trained Mask2Former (built 2026-09-17, not yet run)

**Why.** The only Mask2Former (exp046, continued as exp058) trained about 4 epochs of a
planned 50 before the session limit cut it (2,309 + ~1,608 steps at batch 1). It shrank
each 2048 photograph to 1024, so its masks were stride 8 against the native frame, a
grid exp136 caps at IoU 0.771. It used swin-tiny and a custom Hungarian loss. Its
validation 0.2468 says nothing about the architecture. Correction to an earlier chat
claim: it used 100 queries (the `from_pretrained` default); 32 was its fixed number of
target slots, and the reduced 32-query config ran only under `--smoke-model`.

**Design** (`experiments/exp_139_mask2former/`):

| | exp139 |
|---|---|
| model | `facebook/mask2former-swin-small-coco-instance`, 100 queries, transformers 4.57.6 loss unchanged |
| input | native pixels only: 1024 crops centred on a random filament, plus 25% of batches as whole 2048 frames at equal pixels per batch |
| targets | pycocotools at native resolution; the HF loss samples them at continuous points, so supervision is sub-pixel |
| augmentation | dihedral group (exact) + gain 0.85-1.15 + gamma 0.8-1.25; no mosaic |
| optimiser | AdamW 1e-4, wd 0.05, backbone x0.1, no decay on norms/embeddings/position tables, clip 0.01 |
| schedule | warmup 500 steps, cosine to zero over the epochs the budget affords (<= 50, >= 25 or refuse) |
| monitoring | validation PQ every 5 epochs on the exp033 path; D4 of the best checkpoint at the end |

**Measured facts behind the design.**
- Split: 974 training records (601 photographs), 180 validation (106). No record is
  empty. At most 23 filaments per training record, 26 per validation record: 100
  queries are never short.
- Filament long side (training): p50 96 px, p90 235, p99 515.5, max 1067; 1.06% exceed
  512 px. A 1024 crop almost always holds its filament whole.
- Both sine position embeddings use `normalize=True` (modeling_mask2former.py lines 1216,
  2036): positions are fractions of the input extent. Crops alone would make 2048
  inference out of distribution; the whole-frame batches fix that.
- Orientation of 4,894 elongated filaments (axis ratio >= 3) versus latitude: |y| < 0.3
  isotropic (50.0% nearer east-west), 0.3-0.6 60.5%, beyond 0.6 88.1% but only 185
  filaments (3.8%). Frames are north-up; 90-degree turns misplace that prior for a few
  percent of filaments, and they keep every D4 test view in distribution.
- bf16: the matcher's cost matrix is promoted to float32 by `torch.minimum` with a
  float32 scalar before SciPy sees it. Checked by running forward, loss and backward
  under bf16 autocast on CPU.

**Verification before any spend** (CODE PATH ONLY: tiny random model, synthetic
stand-in images; none of it says anything about the model's quality):
- Evaluator parity on real data: the exp002 D4 pool re-scores to **0.4558638687**
  (TP 860, FP 397, FN 465), exp062's baseline exactly.
- Memory-probe mode, both extents, fp32 and bf16. A 10-step run with in-run
  evaluation and a background sweep. A stop at step 3 and a resume to step 6 that found
  both reports on disk. The end of a run with Kaggle stubbed: best and latest
  checkpoints plus every report in the first upload, then D4, then a second upload that
  adds the D4 report.
- Prediction geometry with a stub network: an off-centre bar comes back at IoU 0.9954
  single view and identically after D4, so the inverse transforms are right.
- Bugs caught: the `EVAL` event passed `tag` twice (fixed). Persistent DataLoader
  workers never see `set_epoch` on the main process's dataset, so every epoch would
  have repeated the same crops; the epoch now travels inside each sample index.

**Pod.** Code dataset `cosmicwitness/sol-exp139-code`, payload SHA256
`45d59dd1859deff729e73937ff2924d1466240744adb96702455492ba0fa5ddf` (33,601 bytes,
listed intact). Weights go to `cosmicwitness/sol-exp139-mask2former-weights`.
Launch waits for the user's GPU and budget approval.

**Decision rule.** Compare the best validation PQ (single view and D4) with exp119's
0.4681455. Printing 0.40 needs about 0.475. Inspect native-resolution predictions
before any submission.

**exp139 addendum - fitting a 24 GB card (user asked for the full run under $6).**
A 4090 costs $0.34/h; an H100 or A100 costs 2.5-3 times more per unit of work. Memory
of the FULL-SIZE Swin-S Mask2Former (random weights, bf16 autocast, forward + loss +
backward) measured on CPU at 512 and 768 px, in separate processes:

| px | no checkpointing | backbone checkpointing |
|---|---|---|
| 512 | 1.629 GiB | 1.079 GiB |
| 768 | 3.168 GiB | 2.035 GiB |

Linear in pixels. With parameters, Adam moments (0.77 GiB) and three copies of the
float targets: two 1024 crops ~11.5 GiB, four ~21.9; a whole 2048 frame ~22.0 GiB
without checkpointing, ~14.4 with it. Changes:
- The sampler plans optimiser steps (8 crop-equivalents each), then splits each step
  into micro-batches that fit. Checked: micro-batch 8, 4, 2 and 1 give identical step
  plans, identical records and 8 crop-equivalents in every step.
- `--checkpoint whole` recomputes Swin stage activations only for whole-frame
  micro-batches. Checked on CPU: gradients identical to the bit (all 161 parameters,
  drop path on), and the stages really run twice (8 calls instead of 4).
- The pod's probe ladder tries "8 none", "4 none", "4 whole", "2 whole", "2 all",
  "1 all" in that order and logs every rung.
- CPU smoke tests re-run on the new loop (tiny model, synthetic images): probe at micro 2
  with whole-frame checkpointing; a run that sized its schedule, evaluated and ran D4;
  a resume from step 7 to 9 with both stubbed uploads complete.
- Code dataset version 2: payload SHA256
  `20cdf9031ae5e7072bb53c97fb189aca86cb86ac214fd83df1f3256e46aca4de` (35,046 bytes,
  listed intact). Reserve for a $6 cap: $0.50.

**exp139 addendum 2 - H100 plan, speed-ups with identical mathematics (2026-09-17).**
The user chose an 80 GB H100, $6.97 of credit, items 1-3 below, and no concession on
model quality: batch 8 in one forward pass, no accumulation, no checkpointing, all 50
epochs. The 24 GB micro-batch plan is dropped: it is not bit-identical to batch 8,
because HF normalises the loss per forward pass.

Read from transformers 4.57.6 (not yet profiled):
- Pixel-decoder deformable attention is a PyTorch `grid_sample` loop
  (modeling_mask2former.py line 799). HF's own Deformable DETR maps the same op to the
  hub CUDA kernel `kernels-community/deformable-detr` (integrations/hub_kernels.py).
- The Hungarian matcher copies one cost matrix to the CPU per image per decoder layer
  (line 475): 10 layers x 8 images = 80 GPU stalls per step.

Items (`experiments/exp_139_mask2former/src/fast_ops.py`, `data.py`):
1. Deformable attention through the CUDA kernel, hub build or a local build of the MSDA
   source bundled in transformers (kernels/deta), chosen explicitly. The kernel runs on
   float32 inputs, the precision the reference's `grid_sample` uses under autocast.
   The trainer checks it against the PyTorch loop before the first step.
2. Batched matching: every cost matrix is computed, copied to the CPU once, then solved.
   Random draws are made first in HF's recursion order, so the result is bit-identical.
   Fused AdamW (the same update in one kernel; equal to rounding, not to the bit).
3. Loader: photographs decoded once into RAM before workers fork; each instance mask
   decoded once and kept as its bounding-box crop; gain, gamma and normalisation on the
   GPU from uint8 pixels.

CPU equality results (tiny random model; synthetic stand-in pixels, real annotations):
- Batched loss vs HF loss: loss and all 161 gradients identical to the bit, fp32 and bf16.
- Fast loader vs the frozen first loader (`tests/reference_data.py`), 170 keys (150 crops,
  20 whole frames): 0 mask mismatches, 0 metadata mismatches; pixels identical without
  augmentation, max |difference| 6.0e-7 with gain and gamma (float32 pow rounding).
  CPU time per sample 336 -> 116 ms with augmentation, without preloading.

Run policy for the pod: `--epoch-policy fixed` (all 50 epochs, or stop at the speed
check a few minutes in if the budget cannot hold them); in-run evaluation every 10 epochs
with a candidate floor of 0.05, which equals the sweep's lowest threshold and so emits
exactly the same sweep; final D4 on a free Kaggle GPU instead of the pod; scorer parity
in the background on the pod's CPU.

GPU tests pending: free Kaggle T4 kernel `cosmicwitness/sol-exp139-gpu-tests-t4` (both
kernel sources vs the reference, full-model gradients, bit-identity on GPU, memory,
step profile).

**exp139 GPU test run 1 (Kaggle T4, torch 2.8.0+cu128, transformers 4.57.6, fp16 autocast).**
- Loader on the real training images, 230 keys (200 crops, 30 whole frames): 0 mask and 0
  metadata mismatches; pixels identical without augmentation, max |difference| 1.2e-6 with
  it. CPU time per sample 80.1 -> 18.6 ms (augmented), 73.4 -> 3.1 ms (plain). PASS.
- Hub kernel (`kernels-community/deformable-detr` snapshot a4431ff2, build
  torch28-cxx11-cu128) against HF's PyTorch loop, both float32: output and value and
  weight gradients within 2.2e-6 (1024) and 5.7e-6 (2048); location gradient within 2.1e-6
  at 1024 but 1.2e-2 at 2048. Not yet explained. Hypothesis to test, not a finding:
  the location gradient of bilinear sampling jumps at pixel centres, and float32 rounding
  can put a point on either side. Run 2 compares both implementations with a float64 run
  and measures how close each large error sits to a pixel centre.
- Local build of the MSDA source bundled in transformers: compile error on the T4 image.
  Removed.
- Batched loss on GPU: loss identical to the bit (fp32 and fp16) but gradients not. The
  test had no control. `grid_sample`'s CUDA backward is not deterministic, so HF's own
  loss run twice may differ too. Run 2 adds that control.
- Memory with HF's loop, fp16 autocast: one 1024 crop 6.74 GiB, two 13.20 GiB (6.46 GiB
  per crop); one 1536 crop did not fit in 15 GiB. CORRECTION to the CPU estimate: batch 8
  needs about 52 GiB, not 43, before any kernel saving. It still fits 80 GB.
- Step time, batch 2 crops: HF loss 1.339 s, batched loss 1.357 s. The batched matching
  gives no measurable gain on a T4. Top CUDA time: copy_ 20% (dtype casts), grid_sample
  backward 9.5%, mul 8%, cat/stack 7% (mostly the deformable attention's stack).
  CORRECTION: the 1.5-2.5x speed-up guessed before profiling is too high; deformable
  attention is roughly a quarter of the step.

**exp139 GPU test run 2 (Kaggle T4, same stack). All checks pass.**
- Hub kernel against a float64 run of HF's PyTorch loop (1024 px batch 2; 2048 px batch 1):
  output within 4.9e-6 / 8.7e-6, value gradient 2.2e-6 / 4.3e-6, weight gradient
  5.5e-6 / 1.3e-5. Location gradient away from pixel-centre crossings 5.0e-6 / 7.1e-6.
  The large location-gradient errors (max 0.091 / 0.206) are 7 / 39 elements, every one
  within 1.7e-6 / 6.8e-6 px of a pixel-centre crossing. HF's OWN float32 loop scored the
  same way shows the same maxima (0.091 / 0.206) at 12 / 47 elements. Run 1's 1.2% is
  therefore float32 rounding at the kinks of bilinear sampling, present in the reference
  itself. Hypothesis confirmed.
- Batched loss, real pretrained model, one 1024 crop: HF loss twice and batched loss once
  give identical losses to the bit in fp32 and fp16. HF's own two runs already differ in
  gradients (CUDA non-determinism); the batched gradients differ no more. PASS.
- Full model, HF's attention twice vs the kernel. fp32: all three losses identical at print
  precision; median relative gradient difference 2.9e-6 (control 2.5e-7). fp16 autocast:
  loss differs by 8.3e-6 relative, median gradient 5.0e-3 (control 3.8e-4): 16-bit rounding
  amplifies float32-level differences, as any kernel change does under autocast.
- Memory per 1024 crop (fp16): HF 6.64 -> kernel 5.16 GiB (batch 1), 13.04 -> 10.00 GiB
  (batch 2); one 1536 crop now fits (11.11 GiB). Batch 8 on the H100 is about 39 GiB of 80.
- Step time, batch 2 crops: HF 1.433 s; kernel 1.159 s (1.24x); kernel + batched loss
  1.155 s. The batched matching is kept (proven equivalent) but shows no gain on a T4.
- Pod code dataset v3: payload SHA256
  `0d8b0c6ffd96fb1341243b13cd6a945610b3a99145cc672ca4337e4d055c9252` (42,655 bytes,
  listed intact).

H100 estimate (T4 -> H100 factor assumed, NOT measured; the pod's speed check is the real
gate): about 0.32 s per step (range 0.22-0.48), 12,150 steps = 1.1 h (0.7-1.6 h), plus
about 25 min for boot, 10 evaluations and upload: roughly $3.9 total at $2.59/h (range
$3.0-5.3). Credit available: $6.97.

**exp139 launch (2026-09-17).** User approved an H100 NVL at $2.59/h with a $6.50 cap.
Community H100 stock was empty; the user chose to wait for it rather than pay $3.19-3.49
secure. Stock appeared at the third 4-minute check.
- Pod `w9bqhwzk6diukr`, H100 NVL 94 GB, 19 vCPU, 150 GB RAM, CUDA 12.8, created
  13:31:22Z. Settings read back from the pod as intended.
- Attempt 1 failed at `from_pretrained` 53 s into training: the runpod/pytorch image sets
  `HF_HUB_ENABLE_HF_TRANSFER=1` but does not ship `hf_transfer`. Kaggle does not set the
  variable, so the free tests could not see it. Everything before it passed ON THE POD:
  torch 2.8.0+cu128, transformers 4.57.6, kernels 0.12.3; data hash verified; scorer
  parity 0.45586386872921264 reproduced; 707 photographs preloaded and 6,874 instance
  crops built in 47.0 s. The pod stopped itself and uploaded its logs. Cost about $0.10.
- Fix: `HF_HUB_ENABLE_HF_TRANSFER=0` in the pod environment (and in bootstrap.sh for later
  runs). Same pod restarted 13:38:59Z with the markers of attempt 1 cleared once, and
  the cap lowered to $6.40 so the total stays at $6.50.

**exp139 training result (attempt 2, 13:38:59Z-15:27:58Z, pod deleted after verification).**
- All 50 epochs, 12,150 steps at batch 8 in one forward pass, no accumulation, no
  checkpointing. Hub kernel verified on the H100 (same numbers as on the T4). Speed check
  after 45 steps: 0.464 s/step, projected $5.25 incl. reserve against the $6.40 cap: all
  50 epochs committed, no epoch cut. Steady 0.47 s/step throughout; 39.35 GiB of 93 GiB.
- Single-view validation PQ by epoch (exp033 sweep, 106 photographs, 180 records):

  | epoch | 5 | 10 | 15 | 20 | 25 | 30 | 35 | 40 | 45 | 50 |
  |---|---|---|---|---|---|---|---|---|---|---|
  | PQ | 0.4201 | 0.4201 | 0.4414 | 0.4441 | 0.4403 | 0.4565 | 0.4598 | **0.4680** | 0.4668 | 0.4642 |

- Best: epoch 40, PQ 0.46801 at conf 0.85, no erosion, min area 100: SQ 0.6856, RQ 0.6826,
  TP 872, FP 358, FN 453. Epoch 50: 0.4642 (TP 813, FP 269, FN 512). The best epoch is
  selected on the same validation set, so 0.468 is slightly optimistic.
- For scale: the previous Mask2Former (exp046/058) reached 0.2468. exp119, the current
  best submission (two YOLO models, D4, public 0.39), is 0.46815. exp139 matches it with ONE
  model and ONE view. Printing public 0.40 needs about 0.475.
- Weights dataset `cosmicwitness/sol-exp139-mask2former-weights` verified by listing:
  model_e040.pt and model_e050.pt (275,763,231 bytes each), e005-e050 sweep reports,
  events.log, manifest.json. Pod `w9bqhwzk6diukr` deleted; list-pods empty.
- Cost: attempt 1 about 2.2 min, attempt 2 about 109 min at $2.59/h: about $4.8 of the
  $6.97 credit.
- Next (free): Kaggle T4 kernel `cosmicwitness/sol-exp139-submission-t4` replays epoch 40
  in float32 (tolerance 0.005), measures D4, picks the mode, writes submission.csv and a
  review sheet. No submission without the user's approval.

**exp139 deployment check (free Kaggle T4 kernel `cosmicwitness/sol-exp139-submission-t4` v1).**
- Checkpoint model_e040.pt, SHA256 equal to the pod manifest; hub kernel passed on the T4.
- Single view, float32 inference: PQ **0.47052** at conf 0.85, no erosion, min area 100
  (SQ 0.6855, RQ 0.6864, TP 881, FP 361, FN 444). The pod's bf16 inference gave 0.46801:
  float32 inference is +0.0025 (9 more TP). Replay within the 0.005 tolerance.
- D4 (eight views, exp033 vote fusion link 0.5 / vote 0.5): PQ 0.46792 at conf 0.70, min area
  300 (TP 920, FP 463, FN 405). +39 TP but +102 FP: the fusion tuned for YOLO does not help
  this model. Single view chosen.
- Test file written, NOT submitted: 1,192 instances over all 180 test photographs, CSV
  SHA256 `CCA9C06EB7FC70E6FBFF11609554A490F6F0A0BD34A48E88F648F434FC986BFF`.
- Visual review (review_sheet.jpg, 12 native-resolution crops centred on ground truth,
  green truth vs red emission): boundaries hug the dark filament edges, with no fattening
  and no merges; large and medium filaments are found reliably, including a long curved one.
  Misses concentrate on thin, faint filaments, several beside active regions. One
  limb-adjacent emission runs longer than the drawn truth. Crops centred on truth cannot
  show false positives elsewhere; FP count is from the sweep only.
- Transfer estimate (fitted on YOLO submissions; Mask2Former may transfer differently):
  0.823 x 0.4705 + 0.004 = 0.391, which prints 0.39. Printing 0.40 needs about 0.475.

**exp139 PUBLIC SCORE: 0.41** (submission 56309562, 2026-09-17 16:32 UTC). Mask2Former alone,
epoch 40, single view, float32, conf 0.85, no erosion, min area 100, validation 0.47052.
Previous best 0.39 (exp119, validation 0.46815). The YOLO-fitted transfer
(0.823 x val + 0.004) predicted 0.391; the observed 0.41 (true value in [0.405, 0.415))
implies a public/validation ratio of about 0.86-0.88 for this model, against 0.83 for the
YOLO submissions. One observation, so treat the ratio as indicative. This ties the best
honest score on the board. The exp140 combination gate is changed accordingly: it must
beat exp139 alone on validation by at least 0.005.

**exp140 - Mask2Former + YOLO combination (free Kaggle T4 kernel `sol-exp140-ensemble-t4`).**
- Replays exact: exp119 0.46814553 (852/328/473); exp139 0.47052500 (881/361/444).
- Scores mapped to the metric's question per source (isotonic, cross-fitted over two folds of
  photographs). Calibration shows exp139 is over-confident in its mid range: score 0.854 ->
  45% of records match, 0.94 -> 64%, 0.97 -> 88%. YOLO pools are closer to calibrated.
- Fusion variants (exp119's fuse_sources, link 0.5, vote 0.5), validation PQ:

  | variant | PQ | TP/FP/FN | rule |
  |---|---|---|---|
  | raw scores, three models | 0.46677 | 859/342/466 | conf 0.50, area 300 |
  | calibrated, three models | 0.46719 | 877/372/448 | conf 0.35, area 300 |
  | calibrated, three, YOLO masks eroded first | 0.46924 | 832/297/493 | conf 0.40, area 300 |
  | calibrated, exp097 + exp139, YOLO eroded first | **0.47151** | 849/303/476 | conf 0.40, area 100 |

- Decision: best is +0.0010 over exp139 alone (0.47052), under the +0.005 margin. No test file
  written, nothing submitted. Combining with the YOLO models adds nothing measurable to
  Mask2Former under this fusion; exp139 alone (public 0.41) stays the submission.

**Exact gradient accumulation (2026-09-17), the enabler for a cheap full-data run.**
transformers normalises each forward pass by that pass's own mask count (mask and dice) and
its own class-weight sum (cross entropy). Dividing each micro-batch by the number of parts,
the usual accumulation, therefore does NOT reproduce one big batch when the parts hold
different numbers of filaments. `train.step_losses` scales each micro-batch by its exact
share of both quantities. Measured on CPU with the tiny model, stochastic depth silenced and
the loss's point sampling fixed so both runs sample identically (samples of 6, 5, 1 and 1
filaments, split 2 + 2):

| accumulation | loss vs one batch | gradient norm difference |
|---|---|---|
| exact shares | 9.8e-08 | 1.4e-06 |
| 1/parts | 1.5e-03 | 4.1e-01 |

So a 24 GB card at $0.34/h can now train exactly what the 94 GB card trained at $2.59/h.
`load_records(..., "all")` adds the 1,154-record fold for a full-data run.

## exp142 full-data run: blocked by RunPod credit exhaustion (2026-09-17)

Four pods were created for the fold=`all` (1,154 record) Mask2Former run. Every
one was stopped by RunPod within 1 to 6 minutes of reaching RUNNING:

| pod | GPU | price/h | reached | stopped after |
|---|---|---|---|---|
| q101ug5alj8cla | RTX 5090 | 0.69 | dependency stage | 70 s |
| l95e8at39c0cbh | RTX 4090 | 0.34 | dependency stage | 68 s |
| d7og1dmra53n00 | RTX 4090 | 0.34 | dependency stage | 76 s |
| qjis4a8v5yckpy | RTX 3090 | 0.22 | training step 1 | ~5 min |

The third pod proves the bootstrap is sound: it passed the dependency stage,
downloaded the data, verified the deformable-attention kernel against the float64
reference (both checks `passed: true`), built the model, reported
`records: 1154, steps_per_epoch: 288, micro: 2, gpu_gib: 10.07` and completed
training step 1 at 4.587 s/step. The stop came from RunPod, not from the script.

Signature of the cause: `pod-action start` then answers "There are not enough free
GPUs on the host machine", and `create-pod` intermittently answers "no longer any
instances available" while `get-capacity` reports High stock for the same GPU.
September spend stands at $12.16. The credits are gone; the run cannot continue on
RunPod without a top-up.

Measured cost of these attempts: about $0.04.

One planning number worth keeping: on an RTX 3090 at micro-batch 2 the full-data
run costs 288 steps/epoch x 4.6 s = 22 min/epoch, so 30 epochs need about 11
hours ($2.42 at $0.22/h). A 4090 at $0.34/h would need about 4 hours ($1.36).

## exp143 weight soup: submitted, 0.40 public (2026-09-17)

The average of exp139 epoch 40 and epoch 50 weights, emitted with the same fixed rule
(single view, float32, confidence 0.85, no erosion, minimum area 100), wrote 1,215
instances over the 180 test photographs and scored **0.40** public. Epoch 40 alone scores
0.41. The soup does not help this model, so the leaderboard answers exp141's first
question before exp141 finishes.

Kernel: `cosmicwitness/sol-exp143-soup-submission-t4`, built by
`scripts/package_exp143_soup_submission.py --base soup`. It rebuilds the average from the
two checkpoints, checks both against the pod manifest, and writes the CSV only. It runs on
a free T4 and spends no credits.

## exp141 result: neither the soup nor a re-tuned fusion helps (2026-09-18)

Validation PQ on the canonical split, all at the fixed rule confidence 0.85, no erosion,
minimum area 100:

| model | PQ | change |
|---|---|---|
| epoch 40, single view | 0.47052 | baseline |
| epoch 50, single view | 0.46210 | -0.0084 |
| soup of 40 and 50 | 0.46881 | -0.0017 |
| eight views, link 0.60, vote 0.375, exp119 support, conf 0.70 | 0.47002 | -0.0005 |

The soup number predicted the leaderboard: validation -0.0017 became 0.41 -> 0.40 public.

The fusion row is worth keeping for the recall question. Eight-view fusion raises true
positives from 881 to 941 and cuts misses from 444 to 384, but false positives rise from
361 to 486. The extra detections are real filaments about half the time, so the net PQ is
flat. A gate that can tell those two groups apart would convert this into a gain; plain
vote and link thresholds cannot.

Nothing here justifies a submission. Epoch 40 single view stays the best model at 0.41.

## exp145: the host's self-evaluation notebook on exp139 epoch 40 (2026-09-21)

Kernel `cosmicwitness/sol-exp145-host-eval-t4` ran the organisers' notebook
(`azimahmadzadeh/self-evaluation-notebook`, sha256 ae7c9445..., stored at
`experiments/exp_145_host_evaluation/host/`) verbatim on our validation predictions
(106 photographs, 180 records, deployed rule: one view, float32, conf 0.85, area 100).

| | PQ | SQ | RQ | TP | FP | FN |
|---|---|---|---|---|---|---|
| host notebook | 0.4705249989 | 0.6855 | 0.6864 | 881 | 361 | 444 |
| our evaluator | 0.4705249990 | 0.6855 | 0.6864 | 881 | 361 | 444 |

Identical to nine decimals: every validation number in this project measures the
leaderboard's quantity. The notebook's relation counts (any overlap):
- ground truth (1,325): 245 touch no prediction (18.5%), 1,043 exactly one, 37 two or more (2.8%);
- predictions (1,242 record-level): 165 touch nothing (13.3%), 1,037 exactly one, 40 two or more (3.2%).
Fragmentation and merging are rare; the dominant failure is a missed filament.
Recall by GT-area sextile rises from 0.45 (under 471 px) to 0.75 (over 3,183 px).

Inference on a free T4, float32: 2.49 s per photograph (median), 2.7 GiB peak.

Annotator agreement under the same rule, at native resolution
(`src/annotator_agreement.py`, CPU, pycocotools IoU): PQ 0.340 over 1,196 ordered pairs of
the 296 multiply-annotated photographs; 0.347 over the 202 validation pairs. The earlier
1024-raster figure (0.3371) agrees. A reference for label variability, not a ceiling.

Leaderboard for the remaining variants: epoch 50 0.40, eight-view fusion 0.40, calibrated
exp140 ensemble 0.40 (submitted 2026-09-21 after a watcher bug: it matched the shared file
name `submission.csv` against older submissions and skipped it; the watcher now keys on the
message). Epoch 40 alone stays best at 0.41.

**CORRECTION to the exp142 entry above (2026-09-21).** The credit-exhaustion diagnosis was
wrong. The diagnostics the 3090 pod uploaded to `cosmicwitness/sol-exp142-fulldata-weights`
show `train.py exited with 1`, and its `train_stdout.log` ends in `ZeroDivisionError` at
`experiments/exp_139_mask2former/src/train.py` line 453: the budget projection divided by
`--eval-every`, which the full-data run sets to 0. The bootstrap's ERR trap then uploaded
the diagnostics and stopped the pod through the RunPod API, which is why RunPod's system
log showed only "stop container". The two 4090 pods stopped the same way at an earlier
stage whose diagnostics were overwritten and are lost. The division is now guarded
(`evaluations_for` returns 0 when `eval_every` is 0). The same 3090 pod also measured
1.344 s/step at step 25, so 30 full-data epochs cost about $0.71 at $0.22/h, not $2.42.
