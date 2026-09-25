# exp097: YOLOv8l-seg at native 2048 on RunPod, trained until converged

Status: code written and tested locally; **nothing launched and nothing billed**.
Owner decisions still required are marked **DECISION**.

## 1. Objective

Produce the strongest pretrained detector that the 0.39 pipeline can use unchanged:
- D4 dihedral TTA;
- cluster fusion at IoU 0.5 with a 0.5 pixel vote;
- erosion;
- confidence and area emission rules.

Train it until validation stops improving, not until a clock runs out. Train as fast
as the hardware allows; above 70% GPU utilization is the aim, not a condition.
Spend at most the user's **$9.95**, and ideally much less.

Success on RunPod means three things:
- the weights exist in the private Kaggle dataset;
- their SHA256 matches the pod's manifest;
- `summary.json` records `stop_reason: converged`.

Success for the project is decided later on Kaggle. It requires exact canonical PQ
above the 0.4558639 baseline, and then public score.

## 2. Why this architecture

| evidence | what it shows |
|---|---|
| exp095, exp002 weights under the public 0.55 team's exact inference recipe: PQ 0.41114 (ours: 0.45586) | the 0.55 team's advantage is in its weights, not its postprocessing |
| that team's published model file: `magfilo_260825_yolov8l_seg_imgsz2048_nc1.pt` | large YOLOv8 segmentation model, trained at 2048 |
| `data/checkpoints/exp002/outputs/args.yaml`: yolo11m-seg, imgsz 1280, optimizer auto, time 8.5 h | our 0.39 model is medium-sized and trained at 1280 |
| exp_010, 2048-trained, 62 of 400 epochs, rate never annealed: validation 0.4274 but public 0.36 = exp002's | a 2048-trained model transfers better even when badly undertrained |
| exp068/073, 2048 fine-tunes of 8 and 4 epochs: D4 0.4555 | short continuations are not a test of native training |

No model has been trained at 2048 from COCO-pretrained weights with a schedule that
finishes. This run does that with the larger model. The training data is the
competition's own. The public MAGFiLO release contains the test labels and is not used.

## 3. What "trained extensively" means here

There is no `time=` limit and no epoch schedule. `convergence.py` decides the length:

| setting | value | reason |
|---|---|---|
| learning rate after warmup | constant `lr0` = 1e-3 (`lrf=1.0`, no cosine) | a schedule tied to an epoch count forces a guess at the length |
| plateau test | smoothed validation fitness has not risen by more than 0.002 for 12 epochs | validation has multiple annotators per photograph, so single epochs are noisy |
| smoothing | mean of the last 3 epochs | one lucky or unlucky epoch cannot reset or trigger a plateau |
| on plateau | multiply the rate by 0.2 and restart the 12-epoch window | every lower rate gets a full chance |
| stop | a plateau persists after 3 decays (rates 1e-3, 2e-4, 4e-5, 8e-6) | "converged": no rate tried still helps validation |
| epoch ceiling | 1000 | Ultralytics requires an integer; the run cannot reach it |

The decay is applied to the LambdaLR scheduler's `base_lrs` and to every parameter
group. The rate actually in use is read back and asserted at the start of every epoch.

The only stops that are not convergence:
1. **Budget:** spend plus the next epoch plus a $0.50 upload reserve would exceed $9.95. This is the user's money cap.
2. **Error:** the traceback is saved, the last weights are uploaded and the pod stops.

GPU utilization never stops training. It is recorded every epoch for diagnosis.

`summary.json` names the reason.

## 4. Hardware and cost

Live RunPod read on 2026-09-15, single-GPU pods:

| GPU | VRAM | community $/h | secure $/h | stock |
|---|---|---|---|---|
| **L40S (planned)** | 48 GB | **0.79** | 1.09 | MEDIUM |
| RTX PRO 6000 Blackwell SE | 96 GB | 1.69 | 2.09 | HIGH |
| A100 80GB | 80 GB | 1.39 | 1.59 | LOW |
| H100 80GB HBM3 | 80 GB | 2.69 | 3.49 | LOW |

Why the L40S:
- **Price:** it gives the most compute per dollar of the cards with enough memory for a large model at 2048.
- **Memory:** 48 GB allows several images per batch, which steadies BatchNorm. It also means full-disk 2048 frames do not need tiling.

At $0.79/h, $9.95 buys at most 12.6 hours. Real speed is **measured by the preflight,
not assumed**; `preflight.json` prints seconds and dollars per epoch before training
starts. If the L40S has no community stock at launch, the fallback is the RTX PRO
6000, which has high stock. **DECISION** at launch.

Pod filters: at least 8 vCPUs and 40 GB RAM per GPU, host CUDA ≥ 12.8.
Image: `runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404`, the plugin's default PyTorch image.

## 5. Training speed

The aim is the fastest possible training; GPU utilization above 70% is the
expected sign of that, not a requirement. Idle GPU time at 2048 px comes from
three sources, and each has a counter-measure.

| cause | counter-measure | how it is verified |
|---|---|---|
| JPEG decode of 4-megapixel images | `cache='ram'`, about 14.5 GB for 974 train and 180 val samples | `on_train_start` asserts every sample is cached; Ultralytics otherwise falls back to disk with only a warning |
| augmentation on too few CPU workers | workers = vCPUs − 2 (at most 16); pod filter ≥ 8 vCPUs; hue and saturation jitter off (grayscale images) | per-epoch **data-wait fraction**: loop time blocked on the dataloader, with a CUDA sync every 25 batches |
| batch too small for the GPU | preflight measures batches 2, 4, 6, 8, 12, 16, 24 on the **most instance-dense** images (worst-case mask memory) and picks the fastest batch within 85% of memory; a larger batch wins only a tie within 1% | peak reserved memory per candidate, in a fresh process each time |

Also:
- cuDNN autotuning is on, and every image is exactly 2048 × 2048;
- AMP is on;
- deterministic mode is off.

Measurement: `nvidia-smi` samples utilization, memory and power every 2 seconds into
`gpu_samples.csv`. The power fraction is recorded because utilization can read high
while kernels are light.

No utilization gate: training always proceeds at the fastest measured setting. The
preflight fails only if no batch size fits in memory. Its report says what limits
speed: the input pipeline when the data-wait fraction exceeds 20% (a pod with more
vCPUs would train faster), otherwise the GPU itself.

## 6. Data flow and storage

```
local repo --(package_exp097_code.py, 29 KB .payload)--> Kaggle dataset cosmicwitness/sol-exp097-code
Kaggle competition data --(pod's own network, 0.7 GB)--> pod
pod --(kaggle datasets create/version)--> Kaggle dataset cosmicwitness/sol-exp097-yolov8l-2048 (private)
Kaggle kernels (dataset_sources) --> evaluation, then test submission
```

**During the run, nothing is downloaded to the local PC.** After the run, the output
dataset is also copied to `D:\Sol\exp097_outputs` for redundancy (user permission,
2026-09-15). `scripts/fetch_exp097_outputs.py` checks every file against the pod's
SHA256 manifest. Live status comes from two sources:
- **Pod events:** the RunPod MCP `stream-pod-logs` tool reads the one-line JSON events from the pod log.
- **Dataset listings:** Kaggle API listings of the weights dataset, a few KB each.

Pod layout. `/workspace` is a 60 GB persistent volume that survives container
restarts; the container disk is 30 GB.

| path | contents |
|---|---|
| `/workspace/code` | verified code tarball, extracted |
| `/workspace/exp097/download` | competition zip |
| `/workspace/exp097/competition` | extracted competition data |
| `/workspace/exp097/yolo_ds` | YOLO dataset: symlinked images, labels with the 0.5 px polygon offset, canonical photograph-grouped split, val fraction 0.15 |
| `/workspace/exp097/preflight_base`, `preflight/` | densest 64 samples; per-batch subsets and logs |
| `/workspace/exp097/weights/yolov8l-seg.pt` | COCO-pretrained starting weights (SHA256 logged) |
| `/workspace/exp097/runs/exp097_yolov8l_2048/weights/{best,last}.pt` | Ultralytics checkpoints |
| `/workspace/exp097/artifacts/` | everything that is uploaded (section 9) |

Transfer to Kaggle:
- **When:** at every learning-rate decay (at most 3 times) and at the end.
- **What:** each upload is a complete snapshot with a `manifest.json` of SHA256s. Mid-run snapshots have optimizer state stripped, about 90 MB each.
- **Verification:** after the final upload, the pod polls the dataset listing until every file is present with the right size. The evaluation kernel re-checks each checkpoint's SHA256 before using it.

## 7. Code map

| file | role |
|---|---|
| `experiments/exp_097_runpod_yolo/pod/launcher.sh` | pod start command: installs Kaggle CLI 2.2.4, downloads the code dataset, **refuses to run unless the payload SHA256 equals `SOL_CODE_SHA256`**, starts bootstrap |
| `.../pod/bootstrap.sh` | stages: restart check → environment and secret check → budget watchdog → dependencies (Ultralytics 8.4.153) → weights → data → preflight → training → upload → self-stop |
| `.../src/fetch_data.py` | competition download, hash and count checks (707/180), shapely check, YOLO dataset build, densest subset |
| `.../src/preflight.py` | batch and throughput benchmark in subprocesses, fastest-setting selection, cost projection |
| `.../src/train.py` | Ultralytics driver with callbacks: cache, optimizer and rate assertions, convergence control, telemetry, GPU summaries, budget guard, uploads, summary |
| `.../src/convergence.py` | plateau controller (pure logic) |
| `.../src/telemetry.py` | data-wait versus step-time accounting (pure logic) |
| `.../src/gpu_monitor.py` | nvidia-smi sampler |
| `.../src/kaggle_sync.py` | stage, manifest, private create or version, verify |
| `.../src/events.py` | one-line JSON events |
| `.../src/evaluate_checkpoints.py` | Kaggle-side scoring with the 0.39 pipeline |
| `scripts/package_exp097_code.py` | builds the code dataset and prints `SOL_CODE_SHA256` |
| `scripts/package_exp097_pod_request.py` | writes the exact create-pod body for review |
| `scripts/package_exp098_evaluation.py` | Kaggle T4 evaluation kernel |
| `tests/test_exp097_*.py` | unit tests of controller, telemetry, monitor parsing, preflight decision, Kaggle transfer |

## 8. Training configuration

| parameter | value | reason |
|---|---|---|
| model | `yolov8l-seg.pt`, COCO-pretrained | section 2 |
| imgsz, rect, multi_scale | 2048, off, off | native frame; every submission infers at 2048 |
| batch | chosen by preflight | largest that fits with headroom at peak throughput |
| nbs | 64 | gradient accumulation keeps the effective batch at 64 |
| optimizer | AdamW, lr0 1e-3, betas (0.937, 0.999), weight decay 5e-4 | named explicitly: `auto` overrides lr0 |
| warmup | 3 epochs, warmup_bias_lr 0 | the default bias warmup of 0.1 suits SGD and would spike biases under AdamW |
| schedule | constant, then decay on plateau (section 3) | convergence-bound |
| augmentation | degrees 15, translate 0.1, scale 0.5, flipud 0.5, fliplr 0.5, mosaic 0, mixup 0, copy_paste 0, hsv_v 0.4, hsv_h 0, hsv_s 0 | exp002's geometry; dihedral TTA relies on flip and rotation invariance; grayscale input makes hue and saturation jitter meaningless |
| masks | mask_ratio 4, overlap_mask on | stock supervision, as exp002 |
| validation | every epoch | the convergence decision needs it; checkpoint choice uses exact PQ on Kaggle later |
| cache, AMP, cuDNN | RAM, on, benchmark | throughput (section 5) |
| seed | 2026 | |
| labels | prepare_yolo with polygon offset 0.5 | reconciles Ultralytics' polygon rasterization with the scorer's (training-target IoU 0.898 → 0.959) |

## 9. Documentation the run produces

`epochs.jsonl` holds one record per epoch:
- epoch, fitness and every validation metric;
- losses and learning rates;
- the controller decision: action, reason, smoothed fitness, best epoch, epochs since best, decays;
- throughput: images per second, data-wait fraction, train and validation seconds;
- GPU usage during training and during validation: mean and 10th-percentile utilization, power fraction, peak memory;
- dollars spent;
- the stop flag and its reason.

Also uploaded:

| file | contents |
|---|---|
| `summary.json` | stop reason, epochs, controller state, final validation, spend |
| `preflight.json` | every candidate batch and the gate decision |
| `data_manifest.json` | data hashes, counts, dataset statistics |
| `environment.json` | GPU, driver, vCPUs, RAM, disk, pod id, price |
| `resolved_args.json`, `args.yaml` | settings Ultralytics actually used |
| `results.csv` | Ultralytics' per-epoch metrics |
| `gpu_samples.csv` | all 2-second GPU samples |
| `events.log`, `train.log` | full logs |
| weights | `stage<k>_best.pt`, `best.pt`, `last.pt` |

After the run, `docs/ACTIVE_EXPERIMENT.md` gets the result tables.

## 10. Safety and failure handling

| risk | handling |
|---|---|
| Kaggle key handling | user decision (2026-09-15): passed directly in the create-pod environment, not through a RunPod secret; never written to the repository or printed in logs; a missing or placeholder key fails the launcher's first Kaggle download within a minute |
| wrong code on the pod | launcher verifies the tarball SHA256 against the value set at creation |
| container restart re-running paid stages | marker files: restart during training uploads and stops; restart after completion stops |
| spend beyond $9.95 | train.py stops gracefully with a reserve; a shell watchdog stops the pod at the cap from any stage |
| out of memory on an unseen crowded image | preflight uses the densest samples plus 15% headroom; an error uploads the last weights and stops |
| silent Ultralytics fallbacks | RAM cache, optimizer, initial rate, batch and per-epoch rate are asserted at runtime |
| upload failure | 4 attempts with backoff; the pod *stops* rather than terminates, so the volume keeps the weights |
| self-stop failure | REST v1 stop, then GraphQL `podStop`, using the injected `RUNPOD_POD_ID` and pod-scoped `RUNPOD_API_KEY`; if both fail the log says so and the MCP `pod-action` tool stops it |
| community host reliability | on-demand community pods are not spot instances; persistent volume; snapshots uploaded at every decay |

## 11. Launch procedure (not started)

1. The Kaggle key is inserted directly into the create-pod environment at launch (user decision); it is not stored in the repository.
2. **Done 2026-09-15.** The code payload is uploaded and its listing verified (section 13). To change code: `python scripts/package_exp097_code.py`, then from inside `kaggle/exp097_code_dataset` run `kaggle datasets version -p . -m "..." -r skip -t`, check the listing, and regenerate the pod request with the new hash. The payload is a gzip-compressed tar named `.payload` because Kaggle auto-extracts `.tar.gz` uploads.
3. Read live price and stock with `list-gpu-types`.
4. `python scripts/package_exp097_pod_request.py --code-sha256 <sha> --price-per-hour <live>`.
5. **DECISION:** approve the request, GPU and price.
6. Create the pod with `create-pod` and the reviewed body.
7. Watch `stream-pod-logs` for `launcher`, `code_verified`, `environment`, `VERSIONS`, `DATA`, and `PREFLIGHT_CANDIDATE`/`PREFLIGHT`.
   - If no batch size fits in memory, the pod has already uploaded the diagnosis and stopped.
8. During training, check the `EPOCH` lines sparingly: convergence decisions, utilization, spend.
9. At `SUMMARY`, confirm the Kaggle dataset listing. Run `python scripts/fetch_exp097_outputs.py` to copy and hash-check the outputs on the D drive, then terminate the pod.

## 12. After training, on Kaggle

1. `python scripts/package_exp098_evaluation.py`, then push the T4 kernel.
2. The kernel reproduces the baseline 0.4558639 from the stored exp002 D4 pool, or stops.
3. It verifies checkpoint hashes, then runs a single-view sweep on every checkpoint.
4. It runs the D4 pipeline on the best 2 checkpoints and sweeps again. The sweeps cover confidence 0.20–0.50 or 0.25–0.50, erosion none or 1 px, and minimum area 100 or 300, all scored with exact PQ.
5. Acceptance: the best D4 rule must beat 0.4558639 by a margin the leaderboard can show. Then a test-submission kernel replays that exact rule on validation and asserts its PQ before emitting the test CSV. Native mask review comes before any submission.

## 13. Known unknowns

These cannot be verified before the pod runs:
- whether the start command string is parsed as a shell command (the launcher logs within seconds);
- whether the pod-scoped API key may stop its own pod (logged; manual fallback);
- real throughput and memory at 2048 (measured by the preflight before training);
- Ultralytics 8.4.153 attribute names used by the callbacks (asserted; failures stop early and cheaply).

Verified locally:
- the unit tests pass;
- all modules compile;
- both shell scripts pass `bash -n`;
- Kaggle CLI 2.2.4 syntax was read from its help text;
- the competition archive layout (707 train and 180 test photographs, annotation path) was read from the local copy;
- the private dataset `cosmicwitness/sol-exp097-code` lists exactly `sol_exp097_code.payload` (29,279 bytes, SHA256 `a56bbf593f012afa563c89b4fdcb27163186aa88dd4e96f1c6d8d710f0d13509`, reproducible build) and `manifest.json`;
- `pod_request.json` carries that hash, launches with the `.payload` name, and holds only a Kaggle key placeholder.
