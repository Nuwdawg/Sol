# Working notes for this repository

## Governing competition guidance (user instruction, 2026-09-13)

Read [the host guidance](docs/HOST_GUIDANCE.md) when choosing experiments or
interpreting scores. The user explicitly requires it to inform our decisions.
The host says judging considers individual predictions and algorithmic logic,
not solely a scalar PQ score. Boundary variability reflects the physical problem;
it is not proof that useful improvement is impossible or that the task is merely
matching an annotation convention.

Continue pursuing the user's >0.46 public-score goal while protecting prediction
quality. Review representative native-resolution predictions and changed masks
alongside metrics before promoting a variant. Look for retained filament detail,
misses, spurious detections, merges/splits and implausible boundary changes.
Do not declare a noise ceiling, suppress plausible objects, or distort boundaries
solely to fit a leaderboard story. Record tradeoffs and reproducible algorithmic
reasons. Keep the official metric and per-annotator evaluation unchanged; the
participant's claim about a "multiple-annotated consensus" is not a metric spec.

Earlier strategy prose about irreducible disagreement and why corrections must
fail contains unproven generalizations. Preserve its measurements, but do not use
those claims as current operating rules.

Written after a two-day effort that moved the leaderboard score not at all. The
experiments are in `docs/leaderboard-analysis.md` and the reasoning in
`docs/strategy.md`. This file is the other half: the mistakes, so they are not
repeated. Every entry below actually happened here.

---

## 1. Never assert a configuration; read it back

The single most expensive habit. Four instances:

- **Claimed `mask_ratio=1` was active** because the validation loop ran 45
  iterations, which implied batch 2. Batch size says nothing about mask ratio.
  `runs/*/args.yaml` said `mask_ratio: 2`. Twelve GPU hours measured a
  configuration whose central change was inert.
- **Claimed a run was healthy at "20 minutes in"** from a status check that was
  already stale. It had died at 5 minutes.
- **Claimed freezing the backbone would free the memory.** It did not — the run
  still sat at 13.2 GB and still failed. The cost was one allocation in the mask
  loss, not backbone activations.
- **Claimed "batch size is not a lever"** from two runs whose memory was
  polluted by a leak, then built a whole ladder on it.

**Rule:** after any run starts, read `args.yaml` (or the equivalent resolved
config) and confirm the thing you intended is what is running. Before reporting
a state, re-query it. An inference about a setting is not a reading of it.

## 2. Ultralytics silently overrides what you pass

- `optimizer="auto"` logs `ignoring 'lr0=...'` and picks its own optimiser and
  learning rate. Name the optimiser explicitly or your `lr0` is decoration.
- `resume=True` (bare) routes to `get_latest_run()`, which searches Ultralytics'
  settings directory, not yours. Pass the checkpoint path.
- Removing `time=` also removes Ultralytics' rescaling of the LR schedule to fit
  the budget. `epochs=400` with a 12-hour cap reached epoch 62 with the cosine
  still at peak. **Size the schedule to the session, not to an aspiration.** A
  short schedule that anneals beats a long one that is cut off.
- The OOM fallback ladder must fail loudly. A silent fallback to a weaker
  configuration is worse than a crash, because it produces a plausible result
  for the wrong experiment.

## 3. Memory experiments need process isolation

Testing several memory configurations in one process gave two rungs
**byte-identical** OOM errors while the batch between them halved — the second
never ran, because `torch.cuda.empty_cache()` cannot reclaim what the previous
trainer still references. Every subsequent rung was then measured against a
polluted heap.

Run each attempt as a subprocess. Also: allocator settings
(`PYTORCH_CUDA_ALLOC_CONF` / `PYTORCH_ALLOC_CONF`) must be set **before anything
imports torch**; set after `torch.cuda.is_available()` they do nothing.

## 4. Train the model on the question the metric asks

The verifier was labelled positive if **any** annotator drew a filament at that
location, reasoning that labelling against one would teach it to call another's
filament a false positive. But PQ scores each record against **one** annotator.
The model reached 72% accuracy at "would somebody draw this" and 17% at "did
this person draw it". The 55-point gap was the labelling decision, not the model.

**Rule:** write down the metric's exact question before choosing labels, and
check the training objective is that question and not a nearby one.

## 5. Validate and deploy the identical rule

A fusion rule was tuned with agreement computed against a candidate pool at
confidence floor 0.05, then deployed running the second model at 0.30. Far fewer
confirmations, 312 vetoes instead of a handful, 979 instances against a baseline
of 1238. Caught only by comparing row counts.

**Rule:** the deployment path must reproduce the validation path exactly. Assert
a cheap invariant (row count, instance count, a known PQ) that fails loudly when
it does not.

## 6. A synthetic test that encodes your assumption proves nothing

Intensity-snapping mask boundaries scored IoU **0.9995** on a synthetic filament
and **destroyed** real masks (SQ 0.678 to 0.654, monotonically worse). The
synthetic filament had a hard intensity edge — which was precisely the
assumption under test. Real filaments fade over several pixels, which is why
annotators disagree about them.

**Rule:** a sanity check confirms the code runs. Only real data tests the idea.
State which one you are doing.

## 7. Watchers whose exit condition is "I did the thing"

A background loop polled for GPU quota and pushed the training kernel when it
appeared. The quota reset, a scheduled reminder fired, and the kernel was
launched by hand — leaving the poller still armed. Its next attempt, two minutes
away, would have pushed a duplicate version and restarted the run.

**Rule:** a watcher that performs an action must check whether the action is
already done, not just whether it is possible.

## 8. Test the lifecycle that will actually occur

A probe verified that a kernel can read its own previous output — for a kernel
that **completed**. The training kernel ends by hitting the 12-hour cap, which
is **cancelled**, and a cancelled kernel's output does not mount. The whole
multi-session resume plan rested on the untested case.

Kaggle also strips the top directory when unpacking an uploaded archive:
`runs/polygon/weights` arrives as `polygon/weights`.

## 9. Cost the sweep before running it

- Caching dense 2048x2048 masks per erosion level is ~9 GB each; three levels
  was SIGKILL. Keep masks as RLE between uses — a couple of kilobytes each —
  and decode only what is painted.
- A 120-point grid took hours where coarse-then-refine takes minutes and finds
  the same optimum.
- `pycocotools.mask.iou` works on RLE directly. Do not decode to dense to
  compare masks.

## 10. Do not generalise a finding past its evidence

Measured that false positives occur at the human disagreement rate, then
extended it to the misses — where it was false, since only 2.1% of truth is
actually invisible. Then over-corrected the other way after the verifier failed.
Two reversals on one question in a day.

**Rule:** a result about one side of a confusion matrix is a result about that
side. Say which measurement supports which claim.

## 11. Numbers you already have should change your actions

Validation said the fusion rule removed 9 instances out of 1238. It was
submitted anyway, and proved 99.3% identical to a previous submission — a wasted
submission that my own measurement had already predicted.

**Rule:** when a measurement says a change is negligible, say so plainly and
push back before spending a limited resource on it.

## 12. Read the primary source, not the abstract

"1,593 observations" in the MAGFiLO abstract counts annotation passes. The
release holds **958 unique observations**, of which the competition already uses
887. An apparent doubling of available training data was actually ~71
photographs.

## 13. Communication

The user asked repeatedly for shorter answers. Long replies with tables and
caveats were not wanted while waiting for a number. Lead with the number, state
the decision, stop.

---

## Project-specific facts worth not rediscovering

- The competition metric pools TP/FP/FN across all images and divides once;
  `shared.utils.aggregate_pq` matches the organisers' implementation and there
  is a differential test in `tests/test_pq_matches_official.py`.
- Validation predicts the leaderboard with an offset of 0.067-0.078, verified
  across a 0.04 range including a deliberately degraded submission. Judge ideas
  on validation before spending a submission.
- Over eleven submissions with public scores from 0.20 to 0.41, public is
  about 0.84 x per-record validation PQ (ten inside the rounding interval;
  `experiments/exp_200_test_truth/README.md`). Pooled truth is refuted (section 17).
- The public leaderboard shows two decimals. A local spread of 0.013 is
  invisible there — four genuinely different models all printed 0.36.
- CPU inference is bit-for-bit identical to GPU inference here (same CSV MD5).
  GPU hours belong to training.
- The public MAGFiLO release contains the test annotations. Do not train on it.

## 14. Enumerate what is running before you launch

The handoff stated that exp087 "was prepared but not launched" and that Kaggle
reported the slug did not exist. Both halves were checked against the slug
`sol-exp087-morden-continue-t4`. The previous agent's push had actually
succeeded under `sol-exp087-morden-continuation-t4`, the slug Kaggle derives
from the *title* "Sol exp087 MORDEN continuation T4". Trusting the handoff, a
second identical continuation was pushed and ran on a second T4 for about ten
minutes before the duplicate was found.

Two separate faults combined:

- A status query for one guessed slug is not a check for running work.
  `kaggle kernels list --user <user> --sort-by dateRun` lists what actually ran.
- Kaggle derives the slug from the title, not from the `id` field. When the two
  disagree the push either 409s or silently lands on a different slug than the
  one every later status call will use. Keep `title` and `id` slug-identical.

**Rule:** before pushing any kernel, list the account's kernels by run date and
confirm nothing equivalent is already RUNNING. Never infer "not launched" from a
single slug lookup, and never inherit that inference from a handoff document.

## 15. Kaggle datasets rewrite what you upload

Found while shipping exp097's code to a RunPod pod through a private Kaggle
dataset:

- **Kaggle auto-extracts archives.** An uploaded `sol_exp097_code.tar.gz` was
  listed afterwards as its extracted files. The single tarball whose SHA256 the pod
  was going to verify no longer existed. Ship integrity-checked bundles under a
  non-archive name (`.payload`) and check the listing, not the upload message.
- **Kaggle CLI 2.2.4 on Windows breaks on `-p` paths with forward slashes.** It
  flattens the path into a temporary filename by replacing only `os.sep`, so
  `-p kaggle/exp097_code_dataset` made a nested temporary path and failed with
  `[Errno 2]` before uploading anything. Run the command from inside the folder
  with `-p .`.
- Parallel Bash tool calls share one working directory. A `cd` in one call can
  move the other, so use absolute paths when calls run together.

**Rule:** after any Kaggle dataset upload, list its files and sizes and compare
them with what was intended before anything depends on the dataset.

## 16. A pinned library is a source to read, not an API to assume

exp097's pod reached training and then died inside its own `on_train_start`, on
`len()` of `dataset.ims`. In Ultralytics 8.4.153 a complete RAM cache is an
`_ImageCache` object, not a list. The callbacks had unit tests, but every test
supplied a list, which was exactly the assumption under test (section 6). Reading
the pinned version's source afterwards found three more defects in the same file,
among them `on_fit_epoch_end` being called a second time by `final_eval`, which
would have written a phantom epoch into the convergence history.

**Rule:** when driver code touches a library's internals, read those internals in
the exact pinned version, then run the real code against that version before it
costs money. The rehearsal belongs on free compute: exp099 runs the identical
payload on a Kaggle T4 in about ten minutes.

## 17. A hypothesis that fits the level must also fit the differences

exp200 explained the validation-to-public gap with "pooled" truth: every annotator's
filaments in one image, matched one-to-one. It put seven past submissions on their public
scores with no fitted constant, so new files were built and a GPU retrain was planned on
it. Those seven barely differ in what they emit, and a constant offset fitted them just as
well. The exp_028 recall probe, submitted in September to test exactly this, was never
checked. Rebuilt from its own weights (its test file matched byte for byte), it missed the
pooled prediction by 0.03, and exp097 stage1 against stage2 missed by 0.012. Per-record
scoring predicted both.

**Rule:** test a scoring hypothesis on the past submissions that differ most in what they
emit (recall probes, degraded models), before building on it. A fit to the level is one
number; the differences between submissions are the evidence. Before overriding a finding
recorded here, rerun the experiment that established it.
