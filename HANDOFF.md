# Kaggle filament segmentation handoff

Last updated: 2026-09-15, Africa/Nairobi  
Workspace: `C:\Users\kkaso\Documents\Scripts\Sol`

## Objective and operating constraints

The task is complete only when a Kaggle submission to
`filament-segmentation-2026` has a **verified public score greater than 0.46**.
The current verified best is **0.39**, so the objective is still open.

- The user has already authorized Kaggle compute, training, submission CSV generation, and submissions. Do not stop to ask again.
- Prefer Kaggle Nvidia T4 GPUs. TPU may be used, but it previously remained queued for hours; do not put it on the critical path.
- Be concise and minimize token spend. Use background monitor scripts for long jobs.
- Never use emojis.
- Do not train on public MAGFiLO test annotations. The public release contains the competition test labels.
- Read `CLAUDE.md` and `docs/HOST_GUIDANCE.md` before choosing or promoting an experiment.
- The host says judging also considers individual predictions and algorithmic logic. Review native masks before promotion; do not delete plausible filaments or distort boundaries only to chase a scalar.
- The worktree is intentionally dirty and contains days of uncommitted work. Do not reset, clean, or discard unrelated changes.

## Verified leaderboard state

- Best public submission: ref **56212792**, score **0.39**.
- File: `data/exp062_d4_recall.csv`.
- Local canonical validation for its exact D4 emission: **0.45586386872921264** with TP/FP/FN **860/397/465**.
- Latest submission: exp082, ref **56250449**, score **0.38**. Result: `data/exp082_submission_result.json`.
- exp082 added 77 MORDEN-predecessor semantic-supported masks to exp062. Local PQ rose to 0.45937, but public fell to 0.38. Do not repeat low-confidence proposal addition without materially better evidence.
- Historical local-to-public offset is about **0.067–0.078**. A plausible public 0.46 candidate likely needs canonical validation near **0.53**, not merely 0.46.

## Current authoritative state

### exp085: completed MORDEN-style model

Kaggle kernel `cosmicwitness/sol-exp085-morden-spine-t4` is **COMPLETE** and fully downloaded to `data/exp085_output`.

- 10 epochs, 25.5 training minutes, peak allocation 0.981 GiB.
- Validation mask Dice: **0.7030347**.
- Validation spine Dice: **0.6347279**.
- Standalone connected-component PQ: **0.3226995** (706 TP, 760 FP, 619 FN).
- Conclusion: the solar-specific field learned, but standalone semantic instance decomposition is poor. Do not submit it standalone.
- Checkpoint: `data/exp085_output/training/best.pt`, SHA256 `06045979452475895259ffd139e7fea9377c8bbcdc1f85396eef1dc4e9d0651f`.
- Probability maps: `data/exp085_output/validation/probabilities`.
- Training history: `data/exp085_output/training/status.json`.
- PQ report: `data/exp085_output/validation/report.json`.

The implementation is under `experiments/exp_085_morden_spine`. It adapts the MIT-licensed MORDEN ingredients: 5x5 convolutions, InstanceNorm, multiscale image branches, channel gates, and pyramid context. It adds a spine head and three image channels: raw, CLAHE, and dark local contrast. License is preserved at `third_party/MORDEN_LICENSE`.

### exp086: live local fusion audit

`scripts/audit_morden_fusion.py` is currently running as local PID **27496**. Its monitor is PID **1028**.

- Monitor state: `data/exp086_monitor.json`.
- Stdout/stderr: `data/exp086_morden_fusion.log`, `data/exp086_morden_fusion.err`.
- Terminal report, when done: `data/exp086_morden_fusion/report.json`.
- The worker was authoritatively alive at handoff and CPU-bound. It prints only at completion. Do not restart it just because the log is empty.
- It preserves the exact 0.4558638687 D4 masks as instance anchors and tests exp085 only for pruning, weak-proposal support, and D4-seeded semantic boundaries. Unseeded semantic components cannot enter.

Check it with:

```powershell
Get-Content data\exp086_monitor.json
Get-Process -Id 27496 -ErrorAction SilentlyContinue | Select-Object Id,CPU,WorkingSet64
if (Test-Path data\exp086_morden_fusion\report.json) { Get-Content data\exp086_morden_fusion\report.json -TotalCount 180 }
```

### exp087: LAUNCHED, and the slug in this document was wrong

CORRECTED 2026-09-15 17:30 Nairobi. The previous agent's push DID succeed. Kaggle derives the slug from the title, so it landed on `cosmicwitness/sol-exp087-morden-continuation-t4`, not on `sol-exp087-morden-continue-t4`. Querying only the latter produced a false "not launched" claim, and a duplicate second T4 run was pushed before the error was found. The duplicate `sol-exp087-morden-continue-t4` was cancelled by the user. The authoritative job is the CONTINUATION slug.

Authoritative run: `cosmicwitness/sol-exp087-morden-continuation-t4`, started 2026-09-15 13:49:33 UTC. Watcher PID 2248 downloads to `data/exp087_output`; streamer PID 14196 appends `data/exp087_live.jsonl` (its first lines came from the cancelled duplicate and cover epochs 1-2; ignore them). Progress at 17:30 Nairobi: epoch 14/50, mask Dice 0.70105, spine Dice 0.63514, 0.62 h elapsed, peak 0.981 GiB.

The two kernels differ only in the gzip mtime header of the embedded source archive; the payloads are otherwise byte-identical. The package at `kaggle/exp087_morden_continue_t4` is unchanged except that its title is now slug-identical to its id.

Before pushing anything, run `kaggle kernels list --user cosmicwitness --sort-by dateRun` and confirm nothing equivalent is RUNNING.

Source: `scripts/package_morden_continue.py`. It resumes exp085 epoch 10 for 50 more epochs at LR `8e-5`, then writes TTA validation probability maps without the slow standalone sweep. Expected runtime is about 2.5 hours; exp085 was still improving and had no train/validation gap.

Launch and attach durable monitoring with:

```powershell
python -m py_compile experiments\exp_085_morden_spine\src\train.py experiments\exp_085_morden_spine\src\evaluate.py scripts\package_morden_continue.py
python scripts\package_morden_continue.py
kaggle kernels push -p kaggle\exp087_morden_continue_t4 -t 10800
kaggle kernels status cosmicwitness/sol-exp087-morden-continue-t4
kaggle kernels pull cosmicwitness/sol-exp087-morden-continue-t4 -p data\exp087_remote_v1 -m
python scripts\launch_detached.py --cwd C:\Users\kkaso\Documents\Scripts\Sol --stdout data\exp087_watch.log --stderr data\exp087_watch.err -- python -u -m scripts.watch_kernel cosmicwitness/sol-exp087-morden-continue-t4 data\exp087_output
python scripts\launch_detached.py --cwd C:\Users\kkaso\Documents\Scripts\Sol --stdout data\exp087_stream.log --stderr data\exp087_stream.err -- python -u -m scripts.stream_kernel cosmicwitness/sol-exp087-morden-continue-t4 data\exp087_live.jsonl
```

After pushing, read back `data/exp087_remote_v1/kernel-metadata.json` and verify `machine_shape` is `NvidiaTeslaT4`, GPU is true, and TPU is false. The `stream_kernel` output path is positional; do not pass `--output`.

## Current: exp097 converged; five checkpoints on Kaggle, no pods running

Read `docs/exp097_runpod_training_plan.md` first. It is a YOLOv8l-seg at native
2048, convergence-bound, trained for $2.74 under the user's $9.95 RunPod cap.

State at 2026-09-16 07:16 UTC:
1. No RunPod secret is used. The user decided to pass the Kaggle key directly; it
   goes only into the create-pod call, never into repository files.
2. Code dataset `cosmicwitness/sol-exp097-code` is at v4 and its listing is
   verified (`sol_exp097_code.payload`, 31,069 bytes). SHA256
   `bce589a9632954335e1530390ce3ebc51dbf4313489fa992f3b5bf22be2c1bc9`, which the
   launcher checks before the pod runs anything.
3. Two earlier pods failed and are terminated: `y2cc247f9sq3b5` (stopped at the
   preflight headroom rule) and `qsatuc7fdharyi` (crashed in `on_train_start` on
   Ultralytics 8.4.153's `_ImageCache`). Both are diagnosed in
   docs/ACTIVE_EXPERIMENT.md, with the four defects that second run exposed.
4. Kernel `sol-exp099-exp097-smoke-t4` rehearses the pod code against Ultralytics
   8.4.153 on a free Kaggle T4, including a converged run and an injected
   out-of-memory retry. It passed 23 of 23 checks on payload `bce589a9...`.
   Rebuild it with `scripts/package_exp099_smoke.py` and require
   `SMOKE_RESULT {"passed": true}` before any further pod launch.
5. Pod `hndexs5w71u2wd` ran 2026-09-16 03:18-07:11 UTC on 1x L40 community at
   $0.69/h and **converged at epoch 124**: three decays, best epoch 88, best
   fitness 0.69496, final validation mask mAP50 0.70395 and mAP50-95 0.26242.
   It spent $2.738; with the two failed pods the whole effort cost about $2.96
   of the $9.95. The pod uploaded, verified its listing, stopped itself and was
   terminated. No pods remain.

Five checkpoints are on `cosmicwitness/sol-exp097-yolov8l-2048` with SHA256s in
`manifest.json`: best.pt, last.pt, stage1_best.pt, stage2_best.pt,
stage3_best.pt. Every evaluation and submission verifies those hashes.

Monitoring lesson for the next pod: `stream-pod-logs` returns immediately with
`source: "both"`, a `tail` count and `maxWaitMs` near 3000. It stalls for 300
seconds and fails with `source: "container"` alone, and with `since`. The file
listing of the weights dataset is the reliable secondary signal, since the pod
uploads a full snapshot at every decay, on an error, and at the end.

Evaluation runs per decay, not only at the end. Each decay uploads a full
snapshot, so `kaggle/exp098_exp097_evaluation_t4` (from
`scripts/package_exp098_evaluation.py`) can score the newest checkpoint with
exact PQ against the 0.4558638687 baseline while training continues. It takes
about 20 minutes on a T4 for one checkpoint.

Result for `stage1_best.pt` (epoch 72, the first plateau): D4 PQ 0.453765 at
conf 0.25, no erosion, area 300 — **0.0021 below** the baseline, so no submission
was made. It wins on masks and false positives (SQ 0.6899 against 0.6843, FP 283
against 397) and loses on recall (TP 788 against 860). The sweep now starts at
conf 0.10 in both modes, because the first run put the D4 optimum on the grid's
lower edge.

To submit, run `scripts/package_exp097_submission.py` with exp098's winning
checkpoint, mode and rule plus `--expected-pq`. The kernel replays that rule on
validation, using exp098's own saved D4 pool, and refuses to write a CSV unless
it reproduces the reported PQ within 5e-4.
`tests/test_exp097_emission_parity.py` locks the emitter to `sweep.evaluate`.

The user forbids downloads to the local PC and any unapproved pod start.

## Session of 2026-09-15 evening: what changed

### Submitted

exp090, ref **56257344**, public **0.39**. File `data/exp090_semantic_prune.csv`,
SHA256 `3ECCF19845263090A31B9A3412A687D9BA5315F58C93BB57FF3A86FE1BF3D2EB`,
result `data/exp090_submission_result.json`. It removes 100 of exp062's 1233
instances and changes nothing else: zero additions, zero boundary changes. Local
canonical PQ 0.4558639 to 0.4606350.

Public stayed at 0.39. That is consistent with a real but small gain, because a
+0.0048 local change is far below what two decimals can resolve. The useful
signal is directional: every earlier *addition* experiment lost a point
(exp064, exp065, exp082 all 0.38), and this *removal* held the score.

3 submission slots remained after it.

### The measurement that should drive planning

`data/exp088_headroom/` holds the decomposition. Baseline SQ 0.6843, RQ 0.6662.

- A perfect proposal-quality head over the entire weak D4 pool caps at PQ
  **0.4932**, because the pool's best available mask per truth record averages
  only 0.665 IoU. The strategy this document previously recommended cannot
  reach the goal on its own.
- The best single mask scored against every annotation record of its group
  reaches IoU **0.809** (pixel majority over 221 groups). On those same
  photographs the current emission reaches 0.680. Segmentation quality, not
  discovery, is the larger unclaimed term.
- Two annotators agree with each other at only 0.636, so 0.809 is not a
  contradiction: the metric rewards the mask nearest all annotators at once.
- Uniform morphology cannot collect it. Matched masks have precision 0.817,
  recall 0.826, area ratio 1.041, and a single dilation costs PQ 0.4286. The
  error is local shape.

### exp092: the boundary refiner (main line)

Code is written and compiles: `experiments/exp_092_refiner/` with `crops.py`,
`model.py`, `prepare.py`, `train.py`, `evaluate.py`. Packager
`scripts/package_refiner.py`, kernel `kaggle/exp092_refiner_t4`.

- One training example per (proposal, annotation record) pair, which is the
  question the metric asks. `CLAUDE.md` section 4 explains why the exp025
  "any annotator" labelling was wrong.
- Inputs are the three exp085 solar channels plus the proposal mask, cropped
  around the proposal at 1.8x context and resampled to 256.
- 2.84M parameter U-Net reusing exp085's 5x5 convolutions, instance
  normalisation and channel gates.
- Deployment changes boundaries only. Instance count cannot rise.
- **Verified invariant:** feeding the refiner's own input through `repaint`
  reproduces PQ 0.45586386872921264 exactly, so any measured change is
  attributable to refinement and not to repainting.
- Validation crop cache already built at `data/exp092_val_cache`: 1017 examples,
  mean seed IoU 0.6367.

exp091 finished and exp092 is RUNNING (`cosmicwitness/sol-exp092-refiner-t4`,
watcher PID 4880, streamer PID 17112). When its report lands:

```
python scripts/package_refiner_deploy.py --threshold T --guard G --expected-pq P
kaggle kernels list --user cosmicwitness --sort-by dateRun --page-size 5
kaggle kernels push -p kaggle/exp093_refiner_deploy_t4 -t 10800
```

Take T, G and P from `data/exp092_output/validation/report.json` `best`. Push
only if P exceeds 0.4558638687.

### exp091: train-split D4 proposals

`cosmicwitness/sol-exp091-train-harvest-t4`, RUNNING. The refiner needs
proposals on the train split, which never existed; only validation had them.
This reruns the frozen exp002 detector over the 601 training photographs with
the identical eight dihedral views and fusion constants.

Its parity gate passed perfectly: four shipped validation photographs were
regenerated and matched the stored pool at **mean best IoU 1.0** with identical
instance counts. Watcher PID 21372, streamer PID 24708, output
`data/exp091_output`. Expect about 74 minutes.

When it completes: `python scripts/package_refiner.py` then push
`kaggle/exp092_refiner_t4`.

### exp087 correction

See the exp087 section below. A duplicate was launched from a wrong-slug claim
in this document. Always run
`kaggle kernels list --user cosmicwitness --sort-by dateRun` before pushing.

## Best next strategy

The strongest next idea is a **D4-proposal quality model trained on the exact per-annotation scoring question**, with the pretrained YOLO proposals at the center of the system and exp085 as supporting evidence.

Why this is worth doing:

- The D4 candidate oracle shows **168 of 465** validation misses have a weak proposal above IoU 0.5.
- The old exp025 crop verifier materially improved ranking below the confidence floor, but its labels were wrong: a proposal was positive if **any annotator** drew it. The metric evaluates each annotation record separately.
- Reharvest candidates and create one training example per `(proposal, annotation record)`, or use the fraction of that photo's annotation records that match as a soft target. This learns expected annotation agreement while retaining image/mask morphology.
- Inputs should include the raw crop, proposal mask, exp085 semantic probability, and exp085 spine probability. Candidate confidence and D4 support can enter the final small calibration head.
- Select checkpoints and thresholds by exact canonical pooled PQ, not classifier accuracy or AP alone.
- Keep the deployed rule byte-for-byte identical to validation and assert the unchanged baseline PQ before testing promotions.

Reusable old code:

- `experiments/exp_025_verifier/src/harvest.py`
- `experiments/exp_025_verifier/src/model.py`
- `experiments/exp_025_verifier/src/apply.py`
- `experiments/exp_007_tta/src/dihedral.py`
- `experiments/exp_033_inference/src/sweep.py`
- Exact retained D4 masks with metadata: `data/exp083_d4_consensus/rich_candidates.json`
- Weak/fused D4 validation pool: `data/exp066_output/code/d4_candidates.json`
- Ground-truth and split helpers: `shared/data_split.py`, `experiments/exp_033_inference/src/sweep.py`

Do not reuse exp025's `any annotator` labeling comment or behavior. `CLAUDE.md` section 4 documents why that was the wrong target.

## Completed experiments that should not be repeated

- exp073/074 ROI continuation: D4 PQ 0.45549, below retained baseline.
- exp075 boundary transfer: native/intersection/union all below 0.45586.
- exp076 physical filters: best 0.45463; added FPs outweighed recall.
- exp081 frozen exp004 semantic support: local 0.45937, but exp082 public 0.38.
- exp083 D4 view-count/mean-confidence pruning: no improvement; most accepted masks already have all eight views.
- exp084 area 350 pruning: local 0.45272 and removed visibly real filaments; rejected before submission.
- exp069 reused old verifier: no improvement.
- Naive annotator pixel consensus has already been tested/analyzed and is not enough by itself.

The complete chronology and numeric evidence are in `docs/ACTIVE_EXPERIMENT.md`, `docs/strategy.md`, and `docs/leaderboard-analysis.md`.

## Compute and monitoring

The account UI screenshot `Screenshot 2026-09-15 110713.png` showed **24 h 13 m GPU** and **20 h TPU** remaining at 11:07 Nairobi on 2026-09-15. Several short T4 jobs ran afterward, including exp085 for about 51 total kernel minutes. Recheck the Kaggle UI before relying on an estimate; earlier SDK quota output was misinterpreted.

For every long Kaggle job:

1. Push once.
2. Immediately query status and pull remote metadata to verify the accelerator/configuration.
3. Attach `scripts/watch_kernel.py` for terminal download.
4. Attach `scripts/stream_kernel.py` for durable live logs.
5. Never restart solely because a status call times out or live streaming drops.

For local workers, use `scripts/monitor_local_job.py`; it queries Windows process handles without accidentally killing them.

## Promotion and submission gate

Before spending a submission slot:

1. Verify the unchanged canonical baseline reproduces exactly.
2. Require a material exact-PQ improvement, keeping the historical public offset in mind.
3. Compare TP/FP/FN and instance counts.
4. Render native-resolution largest, median, smallest, and disagreement-heavy changes with `scripts/review_submission_changes.py` or an equivalent deterministic review.
5. Validate schema, test identifiers, RLE decoding, and no-overlap using `shared.utils.check_no_overlap` / `scripts/submit.py`.
6. Save the CSV SHA256, Kaggle submission ref, and result JSON in `data`.

The user has already authorized submission when a candidate passes these gates. Completion still requires a Kaggle public score **strictly greater than 0.46**.
