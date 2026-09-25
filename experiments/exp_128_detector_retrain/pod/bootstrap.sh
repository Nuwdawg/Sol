#!/usr/bin/env bash
# exp128 pod orchestration: three ablation arms in one paid session.
#
#   environment -> dependencies -> weights -> [per arm: data -> train -> upload]
#   -> self-stop.
#
# Arms, fixed in docs/PREREGISTRATION.md and never stacked:
#   A  raw images, every annotator pass          (exp097's path, the control)
#   B  + preprocessing                           (limb-flattened, histogram-matched)
#   C  + one annotator pass per photograph       (dedup median)
#
# Differences from exp097's bootstrap, and why each is needed:
#   - Sentinels are PER ARM. exp097 used one RUN_COMPLETE and one TRAINING_STARTED
#     to make a container restart a safe no-op; reusing them here would make arm B
#     look like a restart of arm A and stop the pod.
#   - Each arm builds into its own yolo_ds_$ARM, because build_yolo returns early
#     when data.yaml already exists.
#   - Preflight runs once. Image size and instance density are the same across
#     arms, so the batch decision carries; re-running it would burn paid minutes
#     to re-derive the same number.
#   - Measurement is NOT done here. Checkpoints upload to Kaggle and a free CPU
#     kernel runs evaluate_checkpoints + evaluate_recall afterwards, the same
#     separation exp098 used. The pod does GPU work only.
set -Eeuo pipefail

WORK=/workspace/exp128
CODE=/workspace/code
ART="$WORK/artifacts"
LOG="$ART/events.log"
MODEL_NAME=yolov8l-seg.pt
: "${SOL_BILLING_OFFSET_SECONDS:=300}"
: "${SOL_ARMS:=A B C}"

mkdir -p "$ART" "$WORK/weights"
cd "$WORK"
export PYTHONPATH="$CODE" PYTHONUNBUFFERED=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export YOLO_CONFIG_DIR="$WORK/ultralytics_config"

now_iso() { date -u +%Y-%m-%dT%H:%M:%SZ; }
log() { printf '%s %s\n' "$1" "$2" | tee -a "$LOG"; }
stage() { log STAGE "{\"name\": \"$1\", \"time\": \"$(now_iso)\"}"; }
error() { log ERROR "{\"stage\": \"$1\", \"message\": \"$2\", \"time\": \"$(now_iso)\"}"; }

arm_flags() {
    case "$1" in
        A) echo "--dedup none" ;;
        B) echo "--dedup none --preprocess" ;;
        C) echo "--dedup median --preprocess" ;;
        *) echo "" ;;
    esac
}

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
        --dataset "$SOL_UPLOAD_DATASET" --title "sol exp128 detector retrain" \
        --message "$1" --staging "$WORK/staging_diagnostics" --verify-seconds 120 \
        --log "$LOG" --file events.log="$LOG" \
        --file environment.json="$ART/environment.json" \
        --file preflight.json="$ART/preflight.json" \
        || error upload "diagnostic upload failed"
}

trap 'error bootstrap "command failed at line $LINENO"; upload_diagnostics "bootstrap failure line $LINENO"; stop_pod' ERR

if [[ -f "$ART/ALL_ARMS_COMPLETE" ]]; then
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
    arms=os.environ.get("SOL_ARMS"),
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

# -------------------------------------------------------------- arms loop ---
PREFLIGHT_DONE=0
BATCH=0
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

for ARM in $SOL_ARMS; do
    ARM_ART="$ART/$ARM"
    mkdir -p "$ARM_ART"
    if [[ -f "$ARM_ART/RUN_COMPLETE" ]]; then
        stage "arm_${ARM}_already_complete"
        continue
    fi
    if [[ -f "$ARM_ART/TRAINING_STARTED" ]]; then
        error "arm_$ARM" "container restarted during this arm; not resuming automatically"
        continue
    fi
    FLAGS=$(arm_flags "$ARM")
    stage "arm_${ARM}_data"
    log ARM "{\"arm\": \"$ARM\", \"flags\": \"$FLAGS\"}"
    python -m experiments.exp_097_runpod_yolo.src.fetch_data --workdir "$WORK" \
        --manifest "$ARM_ART/data_manifest.json" --log "$LOG" $FLAGS
    # build_yolo returns early when data.yaml exists, so give each arm its own tree.
    if [[ ! -d "$WORK/yolo_ds_$ARM" ]]; then
        mv "$WORK/yolo_ds" "$WORK/yolo_ds_$ARM"
    fi
    TRAIN_IMAGES=$(python -c "import json; print(json.load(open('$ARM_ART/data_manifest.json'))['yolo']['train_images'])")
    log ARM_DATA "{\"arm\": \"$ARM\", \"train_images\": $TRAIN_IMAGES}"

    if [[ $PREFLIGHT_DONE -eq 0 ]]; then
        stage preflight
        PREFLIGHT_RC=0
        python -m experiments.exp_097_runpod_yolo.src.preflight \
            --model "$WORK/weights/$MODEL_NAME" --base "$WORK/preflight_base" \
            --workdir "$WORK/preflight" --workers "$WORKERS" --train-images "$TRAIN_IMAGES" \
            --price-per-hour "$SOL_PRICE_PER_HOUR" --headroom 0.92 \
            --output "$ART/preflight.json" --log "$LOG" || PREFLIGHT_RC=$?
        if [[ $PREFLIGHT_RC -ne 0 ]]; then
            error preflight "no batch size fit, exit $PREFLIGHT_RC; no training started"
            upload_diagnostics "preflight exit $PREFLIGHT_RC"
            stop_pod
        fi
        BATCH=$(python -c "import json; print(json.load(open('$ART/preflight.json'))['decision']['choice']['batch'])")
        PREFLIGHT_DONE=1
        log STAGE "{\"name\": \"preflight_passed\", \"batch\": $BATCH, \"workers\": $WORKERS}"
    fi

    stage "arm_${ARM}_training"
    touch "$ARM_ART/TRAINING_STARTED"
    trap - ERR
    set +e
    python -u -m experiments.exp_097_runpod_yolo.src.train \
        --data "$WORK/yolo_ds_$ARM/data.yaml" --model "$WORK/weights/$MODEL_NAME" \
        --batch "$BATCH" --workers "$WORKERS" --project "$WORK/runs" \
        --name "exp128_$ARM" --artifacts "$ARM_ART" \
        --budget-usd "$SOL_BUDGET_USD" --price-per-hour "$SOL_PRICE_PER_HOUR" \
        --pod-start-epoch "$(( SOL_POD_START_EPOCH - SOL_BILLING_OFFSET_SECONDS ))" \
        --upload-dataset "$SOL_UPLOAD_DATASET" \
        --upload-title "sol exp128 arm $ARM weights" --staging "$WORK/staging_$ARM" \
        2>&1 | tee -a "$ARM_ART/train.log" \
        | grep --line-buffered -E '^(STAGE|EPOCH|UPLOAD|UPLOAD_ERROR|ERROR|SUMMARY) '
    TRAIN_RC=${PIPESTATUS[0]}
    set -e
    trap 'error bootstrap "command failed at line $LINENO"; upload_diagnostics "bootstrap failure line $LINENO"; stop_pod' ERR

    if [[ $TRAIN_RC -eq 0 ]]; then
        touch "$ARM_ART/RUN_COMPLETE"
        log ARM_COMPLETE "{\"arm\": \"$ARM\"}"
    else
        error "arm_$ARM" "train.py exited with $TRAIN_RC"
        upload_diagnostics "arm $ARM training exit $TRAIN_RC"
    fi
done

touch "$ART/ALL_ARMS_COMPLETE"
stage complete
kill "$WATCHDOG_PID" 2>/dev/null || true
stop_pod
