#!/usr/bin/env bash
# exp097 pod orchestration:
#   environment -> dependencies -> weights -> data -> preflight gate -> training
#   -> upload to Kaggle -> self-stop.
# Started by pod/launcher.sh only after the code tarball's SHA256 matched the
# value set at pod creation. Design and rationale: docs/exp097_runpod_training_plan.md
#
# Container stdout is the only live channel (the RunPod MCP tools read it and
# cannot open a shell), so it carries one-line `TAG {json}` events only. Full
# Ultralytics output goes to $ART/train.log and is uploaded with the weights.
set -Eeuo pipefail

WORK=/workspace/exp097
CODE=/workspace/code
ART="$WORK/artifacts"
LOG="$ART/events.log"
MODEL_NAME=yolov8l-seg.pt
RUN_NAME=exp097_yolov8l_2048
TITLE="sol exp097 yolov8l 2048 weights"
: "${SOL_BILLING_OFFSET_SECONDS:=300}"   # image pull and boot are billed before this script starts

mkdir -p "$ART" "$WORK/weights"
cd "$WORK"
export PYTHONPATH="$CODE" PYTHONUNBUFFERED=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export YOLO_CONFIG_DIR="$WORK/ultralytics_config"

now_iso() { date -u +%Y-%m-%dT%H:%M:%SZ; }
log() { printf '%s %s\n' "$1" "$2" | tee -a "$LOG"; }
stage() { log STAGE "{\"name\": \"$1\", \"time\": \"$(now_iso)\"}"; }
error() { log ERROR "{\"stage\": \"$1\", \"message\": \"$2\", \"time\": \"$(now_iso)\"}"; }

stop_pod() {
    # Stops (not terminates) the pod: GPU billing ends, the /workspace volume and
    # every artifact on it survive for recovery. Terminate only after the Kaggle
    # dataset has been confirmed from outside.
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
    # Best effort and never fatal: a failed diagnostic upload must not block stopping.
    python -m experiments.exp_097_runpod_yolo.src.kaggle_sync \
        --dataset "$SOL_UPLOAD_DATASET" --title "$TITLE" --message "$1" \
        --staging "$WORK/staging_diagnostics" --verify-seconds 120 --log "$LOG" \
        --file events.log="$LOG" --file environment.json="$ART/environment.json" \
        --file data_manifest.json="$ART/data_manifest.json" \
        --file preflight.json="$ART/preflight.json" --file train.log="$ART/train.log" \
        --file error.json="$ART/error.json" --file summary.json="$ART/summary.json" \
        || error upload "diagnostic upload failed"
}

trap 'error bootstrap "command failed at line $LINENO"; upload_diagnostics "bootstrap failure line $LINENO"; stop_pod' ERR

# ---------------------------------------------------------------- restarts ---
# A container restart must never silently re-run a paid stage.
if [[ -f "$ART/RUN_COMPLETE" ]]; then
    stage restart_after_completion
    stop_pod
fi
if [[ -f "$ART/TRAINING_STARTED" ]]; then
    error restart "container restarted during training; not resuming automatically"
    upload_diagnostics "restart detected during training"
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
    error environment "KAGGLE_KEY did not resolve from the RunPod secret"
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
    gpu=run(["nvidia-smi", "--query-gpu=name,memory.total,driver_version,power.limit",
             "--format=csv,noheader"]),
    cuda_version=os.environ.get("CUDA_VERSION"), pytorch_image=os.environ.get("PYTORCH_VERSION"),
    vcpus=os.cpu_count(),
    ram_gib=round(os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 2**30, 1),
    workspace_free_gib=round(shutil.disk_usage("/workspace").free / 2**30, 1),
    pod_id=os.environ.get("RUNPOD_POD_ID"), datacenter=os.environ.get("RUNPOD_DC_ID"),
    price_per_hour=float(os.environ["SOL_PRICE_PER_HOUR"]),
    budget_usd=float(os.environ["SOL_BUDGET_USD"]),
    code_sha256=os.environ["SOL_CODE_SHA256"]), indent=2))
PY
log ENVIRONMENT "$(python -c "import json; print(json.dumps(json.load(open('$ART/environment.json'))))")"

# ------------------------------------------------------ money cap watchdog ---
# Not a training schedule: the user's hard spend ceiling. train.py stops gracefully
# before this (it keeps a reserve for the final upload); this catches hangs in any
# stage, including ones where Python never starts.
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
pip install -q --no-cache-dir "ultralytics==8.4.153" shapely pycocotools
python - <<'PY' | tee -a "$LOG"
import json
import cv2, shapely, torch, ultralytics
assert torch.cuda.is_available(), "CUDA is not available"
print("VERSIONS " + json.dumps(dict(torch=torch.__version__, cuda=torch.version.cuda,
      ultralytics=ultralytics.__version__, shapely=shapely.__version__,
      opencv=cv2.__version__, gpu=torch.cuda.get_device_name(0))))
PY

# ----------------------------------------------------------------- weights ---
stage weights
if [[ ! -f "$WORK/weights/$MODEL_NAME" ]]; then
    (cd "$WORK/weights" && python -c "from ultralytics import YOLO; YOLO('$MODEL_NAME')")
fi
test -f "$WORK/weights/$MODEL_NAME"
log WEIGHTS "{\"file\": \"$MODEL_NAME\", \"sha256\": \"$(sha256sum "$WORK/weights/$MODEL_NAME" | cut -d' ' -f1)\"}"

# -------------------------------------------------------------------- data ---
stage data
python -m experiments.exp_097_runpod_yolo.src.fetch_data --workdir "$WORK" \
    --manifest "$ART/data_manifest.json" --log "$LOG"
TRAIN_IMAGES=$(python -c "import json; print(json.load(open('$ART/data_manifest.json'))['yolo']['train_images'])")

# ----------------------------------------------------------------- preflight ---
stage preflight
# os.cpu_count() reports the host (96 cores on the first pod), not the pod's
# allocation (9 vCPUs); read the container's CPU quota instead.
WORKERS=$(python - <<'PY'
import os
cpus = len(os.sched_getaffinity(0))
try:
    quota, period = open("/sys/fs/cgroup/cpu.max").read().split()
    if quota != "max":
        cpus = min(cpus, max(1, int(quota) // int(period)))
except (OSError, ValueError):
    pass
print(max(2, min(cpus - 1, 16)))
PY
)
log STAGE "{\"name\": \"workers\", \"workers\": $WORKERS}"
# `set +e` does not silence the ERR trap; an || list does, so an expected non-zero
# exit reaches the diagnosis below instead of the generic trap.
PREFLIGHT_RC=0
python -m experiments.exp_097_runpod_yolo.src.preflight \
    --model "$WORK/weights/$MODEL_NAME" --base "$WORK/preflight_base" \
    --workdir "$WORK/preflight" --workers "$WORKERS" --train-images "$TRAIN_IMAGES" \
    --price-per-hour "$SOL_PRICE_PER_HOUR" --headroom 0.92 \
    --output "$ART/preflight.json" --log "$LOG" || PREFLIGHT_RC=$?
if [[ $PREFLIGHT_RC -ne 0 ]]; then
    error preflight "no batch size fit in memory or the preflight crashed, exit $PREFLIGHT_RC; no training was started"
    upload_diagnostics "preflight exit $PREFLIGHT_RC"
    stop_pod
fi
BATCH=$(python -c "import json; print(json.load(open('$ART/preflight.json'))['decision']['choice']['batch'])")
log STAGE "{\"name\": \"preflight_passed\", \"batch\": $BATCH, \"workers\": $WORKERS}"

# ---------------------------------------------------------------- training ---
stage training
touch "$ART/TRAINING_STARTED"
trap - ERR
set +e
python -u -m experiments.exp_097_runpod_yolo.src.train \
    --data "$WORK/yolo_ds/data.yaml" --model "$WORK/weights/$MODEL_NAME" \
    --batch "$BATCH" --workers "$WORKERS" --project "$WORK/runs" --name "$RUN_NAME" \
    --artifacts "$ART" --budget-usd "$SOL_BUDGET_USD" --price-per-hour "$SOL_PRICE_PER_HOUR" \
    --pod-start-epoch "$(( SOL_POD_START_EPOCH - SOL_BILLING_OFFSET_SECONDS ))" \
    --upload-dataset "$SOL_UPLOAD_DATASET" --upload-title "$TITLE" --staging "$WORK/staging" \
    2>&1 | tee -a "$ART/train.log" \
    | grep --line-buffered -E '^(STAGE|EPOCH|UPLOAD|UPLOAD_ERROR|ERROR|SUMMARY) '
TRAIN_RC=${PIPESTATUS[0]}
set -e

if [[ $TRAIN_RC -eq 0 ]]; then
    touch "$ART/RUN_COMPLETE"
    stage complete
else
    error training "train.py exited with $TRAIN_RC"
    [[ -f "$ART/final_upload.json" ]] || upload_diagnostics "training exit $TRAIN_RC"
fi
kill "$WATCHDOG_PID" 2>/dev/null || true
stop_pod
