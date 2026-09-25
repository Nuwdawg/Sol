#!/usr/bin/env bash
# exp139 pod orchestration: one properly trained Mask2Former on an 80 GB H100.
#
#   environment -> money watchdog -> dependencies -> data -> scorer parity (background)
#   -> training (kernel check, speed check, validation PQ inside) -> upload -> self-stop.
#
# Carried over from exp128's bootstrap, which ran: the environment checks, the
# REST-then-GraphQL self-stop, the diagnostics upload on any failure, and the
# budget watchdog that stops the pod at the cap whatever the trainer is doing.
#
# The user's rules for this run, and how the script keeps them:
#   - No concession on model quality. Batch 8 crop-equivalents per step in ONE
#     forward pass, no accumulation, no checkpointing, all epochs. There is no memory
#     ladder: on a T4 one 1024 crop measured 6.46 GiB with HF's PyTorch attention, so
#     batch 8 needs about 52 GiB of the 80, and a smaller configuration is never tried.
#     An out-of-memory error stops the run at its first step instead of silently
#     training something else.
#   - No stop part-way through. The trainer measures its speed in the first minutes
#     (`--epoch-policy fixed`). If all epochs cannot fit the budget it stops THEN, a
#     few minutes in, never hours in.
#   - Speed-ups are proven equal first. The deformable attention kernel is checked
#     against a float64 run of HF's PyTorch loop before the first step. If it fails,
#     training uses that PyTorch loop (KERNEL_FALLBACK): same mathematics, slower.
#
# A container restart RESUMES from last.pt: the trainer saves every epoch and the batch
# plan is a pure function of (seed, epoch), so a resume repeats at most one epoch.
set -Eeuo pipefail

WORK=/workspace/exp139
CODE=/workspace/code
ART="$WORK/artifacts"
LOG="$ART/events.log"
: "${SOL_BILLING_OFFSET_SECONDS:=300}"
: "${SOL_EPOCHS:=50}"
: "${SOL_RESERVE_USD:=0.5}"
: "${SOL_EVAL_EVERY:=5}"
: "${SOL_MSDA:=hub}"
: "${SOL_MSDA_REVISION:=}"
# The H100 run used the defaults below. A 24 GB card sets SOL_MICRO_BATCH=2 and
# SOL_CHECKPOINT=whole; a full-data run sets SOL_FOLD=all, SOL_EVAL_EVERY=0 (no held-out
# set to score), SOL_SAVE_EPOCHS and SOL_SANITY.
: "${SOL_FOLD:=train}"
: "${SOL_MICRO_BATCH:=8}"
: "${SOL_CHECKPOINT:=none}"
: "${SOL_SAVE_EPOCHS:=}"
: "${SOL_SANITY:=0}"
: "${SOL_EMIT_EPOCHS:=}"
: "${SOL_SUBMIT:=0}"
: "${SOL_RULE_CONF:=0.85}"
: "${SOL_RULE_GROW:=0}"
: "${SOL_RULE_AREA:=100}"

mkdir -p "$ART" "$WORK/hf"
cd "$WORK"
export PYTHONPATH="$CODE" PYTHONUNBUFFERED=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_HOME="$WORK/hf" TOKENIZERS_PARALLELISM=false
# The runpod/pytorch image sets HF_HUB_ENABLE_HF_TRANSFER=1 without installing
# hf_transfer, so every Hub download fails (pod w9bqhwzk6diukr, 2026-09-17).
export HF_HUB_ENABLE_HF_TRANSFER=0

now_iso() { date -u +%Y-%m-%dT%H:%M:%SZ; }
log() { printf '%s %s\n' "$1" "$2" | tee -a "$LOG"; }
stage() { log STAGE "{\"name\": \"$1\", \"time\": \"$(now_iso)\"}"; }
error() { log ERROR "{\"stage\": \"$1\", \"message\": \"$2\", \"time\": \"$(now_iso)\"}"; }

stop_pod() {
    stage self_stop
    touch "$ART/STOP_REQUESTED"
    if [[ -n "${RUNPOD_POD_ID:-}" && -n "${RUNPOD_API_KEY:-}" ]]; then
        local code
        code=$(curl -s -o "$ART/stop_rest.json" -w '%{http_code}' -X POST \
            "https://rest.runpod.io/v1/pods/${RUNPOD_POD_ID}/stop" \
            -H "Authorization: Bearer ${RUNPOD_API_KEY}" || echo 000)
        log STAGE "{\"name\": \"self_stop_rest\", \"http\": \"$code\"}"
        [[ "$code" == 2* ]] && sleep 900
        code=$(curl -s -o "$ART/stop_graphql.json" -w '%{http_code}' -X POST \
            "https://api.runpod.io/graphql" -H "Content-Type: application/json" \
            -H "Authorization: Bearer ${RUNPOD_API_KEY}" \
            --data "{\"query\": \"mutation { podStop(input: {podId: \\\"${RUNPOD_POD_ID}\\\"}) { id desiredStatus } }\"}" \
            || echo 000)
        log STAGE "{\"name\": \"self_stop_graphql\", \"http\": \"$code\"}"
        [[ "$code" == 2* ]] && sleep 900
    fi
    error self_stop "pod could not stop itself; stop it with the MCP pod-action tool"
    sleep infinity
}

upload_diagnostics() {
    python -m experiments.exp_097_runpod_yolo.src.kaggle_sync \
        --dataset "$SOL_UPLOAD_DATASET" --title "sol exp139 mask2former weights" \
        --message "$1" --staging "$WORK/staging_diagnostics" --verify-seconds 120 \
        --log "$LOG" --file events.log="$LOG" \
        --file environment.json="$ART/environment.json" \
        --file train_stdout.log="$ART/train_stdout.log" \
        || error upload "diagnostic upload failed"
}

trap 'error bootstrap "command failed at line $LINENO"; upload_diagnostics "bootstrap failure line $LINENO"; stop_pod' ERR

if [[ -f "$ART/RUN_COMPLETE" ]]; then
    stage restart_after_completion
    stop_pod
fi

# ------------------------------------------------------------- environment ---
stage environment
for name in KAGGLE_USERNAME KAGGLE_KEY SOL_CODE_SHA256 SOL_PRICE_PER_HOUR SOL_BUDGET_USD \
            SOL_UPLOAD_DATASET SOL_POD_START_EPOCH; do
    if [[ -z "${!name:-}" ]]; then
        error environment "missing $name"
        stop_pod
    fi
done
if [[ "$KAGGLE_KEY" == *"RUNPOD_SECRET"* || "$KAGGLE_KEY" == *"{{"* || ${#KAGGLE_KEY} -lt 20 ]]; then
    error environment "KAGGLE_KEY did not resolve"
    stop_pod
fi
python - > "$ART/environment.json" <<'PY'
import json, os, shutil, subprocess
def run(command):
    try:
        return subprocess.run(command, capture_output=True, text=True, timeout=30).stdout.strip()
    except Exception as error:
        return f"error: {error}"
print(json.dumps(dict(
    gpu=run(["nvidia-smi", "--query-gpu=name,memory.total,driver_version",
             "--format=csv,noheader"]),
    cuda_version=os.environ.get("CUDA_VERSION"), vcpus=os.cpu_count(),
    ram_gib=round(os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 2**30, 1),
    workspace_free_gib=round(shutil.disk_usage("/workspace").free / 2**30, 1),
    pod_id=os.environ.get("RUNPOD_POD_ID"), datacenter=os.environ.get("RUNPOD_DC_ID"),
    price_per_hour=float(os.environ["SOL_PRICE_PER_HOUR"]),
    budget_usd=float(os.environ["SOL_BUDGET_USD"])), indent=2))
PY
log ENVIRONMENT "$(python -c "import json; print(json.dumps(json.load(open('$ART/environment.json'))))")"

# ------------------------------------------------------ money cap watchdog ---
watchdog() {
    while true; do
        sleep 60
        if ! python - "$SOL_POD_START_EPOCH" "$SOL_BILLING_OFFSET_SECONDS" \
                "$SOL_PRICE_PER_HOUR" "$SOL_BUDGET_USD" <<'PY'
import sys, time
start, offset, price, budget = map(float, sys.argv[1:])
sys.exit(1 if (time.time() - start + offset) / 3600 * price >= budget else 0)
PY
        then
            error budget_watchdog "spend reached the budget cap"
            upload_diagnostics "budget cap reached"
            stop_pod
        fi
    done
}
watchdog &
WATCHDOG_PID=$!

# ------------------------------------------------------------ dependencies ---
stage dependencies
apt-get update -qq >/dev/null && apt-get install -y -qq libgl1 libglib2.0-0 >/dev/null
pip install -q --no-cache-dir "transformers==4.57.6" pycocotools pandas scipy \
    opencv-python-headless kernels ninja
# Facts, not assertions: a stopped pod cannot be asked what it saw, and training itself
# fails loudly on anything that actually matters (CUDA, versions, memory).
python - 2>&1 <<'PY' | tee -a "$LOG"
import json, importlib.metadata as metadata
import cv2, pandas, pycocotools, scipy, torch, transformers
print("VERSIONS " + json.dumps(dict(torch=torch.__version__, cuda=torch.version.cuda,
      cuda_available=torch.cuda.is_available(), bf16=torch.cuda.is_bf16_supported(),
      capability=list(torch.cuda.get_device_capability(0)) if torch.cuda.is_available() else None,
      transformers=transformers.__version__, kernels=metadata.version("kernels"),
      scipy=scipy.__version__, opencv=cv2.__version__, gpu=torch.cuda.get_device_name(0),
      memory_gib=round(torch.cuda.get_device_properties(0).total_memory / 2**30, 1))))
PY

# -------------------------------------------------------------------- data ---
stage data
python - "$WORK" "$ART/data_manifest.json" <<'PY' | tee -a "$LOG"
import json, sys
from pathlib import Path
from experiments.exp_097_runpod_yolo.src.fetch_data import download, extract, locate, verify
work, manifest = Path(sys.argv[1]), Path(sys.argv[2])
archive = download("filament-segmentation-2026", work / "download")
root = extract(archive, work / "competition")
annotations, train, test = locate(root)
report = dict(archive=archive.name, **verify(annotations, train, test, 707, 180),
              paths=dict(annotations=str(annotations), train_images=str(train),
                         test_images=str(test)))
manifest.write_text(json.dumps(report, indent=2))
print("DATA " + json.dumps(report))
PY
ANNOTATIONS=$(python -c "import json; print(json.load(open('$ART/data_manifest.json'))['paths']['annotations'])")
IMAGES=$(python -c "import json; print(json.load(open('$ART/data_manifest.json'))['paths']['train_images'])")
TEST_IMAGES=$(python -c "import json; print(json.load(open('$ART/data_manifest.json'))['paths']['test_images'])")

# ----------------------------------------------- scorer parity, in the background ---
# CPU only, so it runs beside training instead of holding the GPU idle. A failure does
# not touch the model; it only means the in-run PQ must be re-scored elsewhere.
(
    if [[ ! -f "$WORK/pools/exp002_val_pool.json" ]]; then
        kaggle datasets download cosmicwitness/sol-exp002-pools -f exp002_val_pool.json \
            -p "$WORK/pools" -q
        if [[ -f "$WORK/pools/exp002_val_pool.json.zip" ]]; then
            python -c "import zipfile; zipfile.ZipFile('$WORK/pools/exp002_val_pool.json.zip').extractall('$WORK/pools')"
        fi
    fi
    python -m experiments.exp_139_mask2former.src.evaluate --annotations "$ANNOTATIONS" \
        --output "$ART" --tag parity --parity-pool "$WORK/pools/exp002_val_pool.json" --log "$LOG" \
        || error parity "the scorer did not reproduce exp062; re-score the reports before trusting them"
) > "$ART/parity_stdout.log" 2>&1 &

# ------------------------------------------------------------------ workers ---
CPUS=$(python - <<'PY'
import os
cpus = len(os.sched_getaffinity(0))
try:
    quota, period = open("/sys/fs/cgroup/cpu.max").read().split()
    if quota != "max":
        cpus = min(cpus, max(1, int(quota) // int(period)))
except (OSError, ValueError):
    pass
print(cpus)
PY
)
WORKERS=$(( CPUS > 6 ? (CPUS - 4 < 12 ? CPUS - 4 : 12) : 2 ))
log STAGE "{\"name\": \"workers\", \"cpus\": $CPUS, \"workers\": $WORKERS}"

# ---------------------------------------------------------------- training ---
RESUME=""
if [[ -f "$ART/TRAINING_STARTED" ]]; then
    RESUME="--resume"
    stage resume_after_restart
fi
REVISION=()
if [[ -n "$SOL_MSDA_REVISION" ]]; then
    REVISION=(--msda-revision "$SOL_MSDA_REVISION")
fi
stage training
touch "$ART/TRAINING_STARTED"
trap - ERR
set +e
python -u -m experiments.exp_139_mask2former.src.train --annotations "$ANNOTATIONS" \
    --images "$IMAGES" --output "$ART/run" --batch 8 --micro-batch "$SOL_MICRO_BATCH" \
    --checkpoint "$SOL_CHECKPOINT" --fold "$SOL_FOLD" --sanity-photographs "$SOL_SANITY" \
    --save-epochs $SOL_SAVE_EPOCHS \
    --bf16 --workers "$WORKERS" --epochs "$SOL_EPOCHS" --epoch-policy fixed \
    --msda "$SOL_MSDA" "${REVISION[@]}" --eval-every "$SOL_EVAL_EVERY" --no-final-d4 \
    --price-per-hour "$SOL_PRICE_PER_HOUR" --budget-usd "$SOL_BUDGET_USD" \
    --reserve-usd "$SOL_RESERVE_USD" \
    --billing-start "$(( SOL_POD_START_EPOCH - SOL_BILLING_OFFSET_SECONDS ))" \
    --upload-dataset "$SOL_UPLOAD_DATASET" $RESUME \
    2>&1 | tee -a "$ART/train_stdout.log" \
    | grep --line-buffered -E '^(MODEL|LOADER|KERNEL|KERNEL_FALLBACK|DATA|RESUME|SWEEP_RELAUNCHED|SCHEDULE|STEP|EPOCH|CHECKPOINT|EVAL_STARTED|EVAL|SANITY|BEST|UPLOAD|UPLOAD_ERROR|ERROR|SUMMARY) '
TRAIN_RC=${PIPESTATUS[0]}
set -e
trap 'error bootstrap "command failed at line $LINENO"; upload_diagnostics "bootstrap failure line $LINENO"; stop_pod' ERR

if [[ $TRAIN_RC -eq 0 ]]; then
    # The pod writes and submits the test file itself: no further GPU session is needed and
    # no measurement stands between the trained weights and the leaderboard.
    if [[ -n "$SOL_EMIT_EPOCHS" ]]; then
        stage emit_test
        for EPOCH in $SOL_EMIT_EPOCHS; do
            CHECKPOINT="$ART/run/model_e$(printf '%03d' "$EPOCH").pt"
            if [[ ! -f "$CHECKPOINT" ]]; then
                error emit_test "missing $CHECKPOINT"
                continue
            fi
            SUBMIT=()
            [[ "$SOL_SUBMIT" == "1" ]] && SUBMIT=(--submit)
            python -u -m experiments.exp_139_mask2former.src.emit_test --checkpoint "$CHECKPOINT" \
                --test-images "$TEST_IMAGES" --output "$ART/run" --conf "$SOL_RULE_CONF" \
                --grow "$SOL_RULE_GROW" --min-area "$SOL_RULE_AREA" --msda "$SOL_MSDA" \
                --log "$LOG" "${SUBMIT[@]}" \
                --message "exp142 full-data Mask2Former epoch $EPOCH, single view fp32, conf $SOL_RULE_CONF grow $SOL_RULE_GROW area $SOL_RULE_AREA" \
                2>&1 | grep -E '^(EMIT|EMIT_DONE|SUBMITTED|KERNEL_FALLBACK|ERROR) ' | tee -a "$LOG" || \
                error emit_test "epoch $EPOCH failed"
        done
        python -m experiments.exp_097_runpod_yolo.src.kaggle_sync \
            --dataset "$SOL_UPLOAD_DATASET" --title "sol exp139 mask2former weights" \
            --message "submissions written" --staging "$WORK/staging_submissions" \
            --verify-seconds 120 --log "$LOG" --file events.log="$LOG" \
            $(for f in "$ART"/run/submission_*.csv; do [[ -f "$f" ]] && echo --file "$(basename "$f")=$f"; done) \
            || error upload "submission upload failed"
    fi
    touch "$ART/RUN_COMPLETE"
    stage complete
else
    error training "train.py exited with $TRAIN_RC"
    upload_diagnostics "training exit $TRAIN_RC"
    touch "$ART/RUN_COMPLETE"
fi
kill "$WATCHDOG_PID" 2>/dev/null || true
stop_pod
