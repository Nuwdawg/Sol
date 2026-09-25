#!/usr/bin/env bash
# Poll the pending Kaggle kernels, submit each CSV as it lands, push the queued kernel
# when a GPU slot frees. Safe to stop and restart: every step records a marker file and
# checks the competition's own submission list before spending a slot.
#
#   bash scripts/watch_and_submit.sh            # poll every 5 minutes until all are done
#   bash scripts/watch_and_submit.sh --once     # one pass, then exit
#
# Kaggle allows two GPU sessions at a time, so the third kernel is pushed by this script
# rather than queued by Kaggle.
set -u

COMPETITION=filament-segmentation-2026
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WORK="$ROOT/kaggle/_watch"
INTERVAL=${INTERVAL:-300}
ONCE=0
[[ "${1:-}" == "--once" ]] && ONCE=1
mkdir -p "$WORK"

# slug | csv name written by the kernel | submission message
JOBS=(
"cosmicwitness/sol-exp143-soup-submission-t4|submission_model_e050.csv|exp143 exp139 epoch 50 alone, single view fp32, conf0.85 grow0 area100, validation PQ 0.46210"
"cosmicwitness/sol-exp140-ensemble-t4|submission.csv|exp140 three-way ensemble exp002+exp097+exp139, isotonic calibration, validation PQ 0.47151"
"cosmicwitness/sol-exp144-fusion-submission-t4|submission_fusion_model_e040.csv|exp144 exp139 epoch 40 eight-view fusion, link0.60 vote0.375 exp119 support, conf0.70 grow0 area100, validation PQ 0.47002"
)
QUEUED_SLUG="cosmicwitness/sol-exp144-fusion-submission-t4"
QUEUED_DIR="$ROOT/kaggle/exp144_fusion_submission_t4"

status_of() { kaggle kernels status "$1" 2>&1 | tail -1; }

already_submitted() {
    # The competition's own list is the truth. A marker file alone could lie after a
    # failed upload, and a duplicate submission wastes one of five daily slots.
    kaggle competitions submissions "$COMPETITION" --csv 2>/dev/null | grep -qF "$1"
}

push_queued() {
    [[ -f "$WORK/pushed_queued" ]] && return 0
    local output
    output=$(cd "$QUEUED_DIR" && kaggle kernels push -p . 2>&1 | tail -1)
    if [[ "$output" == *"successfully pushed"* ]]; then
        touch "$WORK/pushed_queued"
        echo "PUSHED $QUEUED_SLUG"
    else
        echo "WAITING to push $QUEUED_SLUG: $output"
    fi
}

collect_and_submit() {
    local slug=$1 csv=$2 message=$3
    local name="${slug##*/}" dir="$WORK/${slug##*/}"
    [[ -f "$WORK/done_$name" ]] && return 0
    local state
    state=$(status_of "$slug")
    case "$state" in
        *RUNNING*|*QUEUED*) echo "RUNNING  $name"; return 0 ;;
        # A kernel this script has not pushed yet has no session, so Kaggle answers 404.
        # That is "not started", not "finished", and must not close the job.
        *404*|*"Not Found"*) echo "PENDING  $name (not pushed yet)"; return 0 ;;
        *COMPLETE*) ;;
        *) echo "STOPPED  $name: $state"; touch "$WORK/done_$name"; return 0 ;;
    esac
    mkdir -p "$dir"
    # The Kaggle CLI on Windows prints a charmap error while still writing the files,
    # so the file listing decides success, not the exit code.
    (cd "$dir" && kaggle kernels output "$slug" -p . >/dev/null 2>&1)
    local path="$dir/$csv"
    if [[ ! -f "$path" ]]; then
        path=$(ls -1 "$dir"/*.csv 2>/dev/null | head -1)
    fi
    if [[ -z "${path:-}" || ! -f "$path" ]]; then
        echo "NO CSV   $name (kernel complete, no csv in output)"
        touch "$WORK/done_$name"
        return 0
    fi
    local rows
    rows=$(( $(wc -l < "$path") - 1 ))
    if (( rows < 100 )); then
        echo "BAD CSV  $name has $rows rows; not submitting"
        touch "$WORK/done_$name"
        return 0
    fi
    # Key on the message, never the file name: several kernels write "submission.csv",
    # and matching that name skipped exp140 on 2026-09-18 because older, different
    # submissions carried the same file name.
    if already_submitted "$message"; then
        echo "SKIP     $name already on the submission list"
        touch "$WORK/done_$name"
        return 0
    fi
    echo "SUBMIT   $name $(basename "$path") rows=$rows"
    kaggle competitions submit "$COMPETITION" -f "$path" -m "$message" 2>&1 | tail -2
    touch "$WORK/done_$name"
}

while true; do
    push_queued
    remaining=0
    for job in "${JOBS[@]}"; do
        IFS='|' read -r slug csv message <<< "$job"
        collect_and_submit "$slug" "$csv" "$message"
        [[ -f "$WORK/done_${slug##*/}" ]] || remaining=$((remaining + 1))
    done
    if (( remaining == 0 )); then
        echo "ALL DONE"
        kaggle competitions submissions "$COMPETITION" --csv 2>/dev/null | head -5
        break
    fi
    (( ONCE == 1 )) && break
    sleep "$INTERVAL"
done
