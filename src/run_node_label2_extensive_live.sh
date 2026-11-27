#!/bin/bash

# LENS Label 2 — native IB contention plus live model watcher.

set -u

ACTIVATE="${LENS_ACTIVATE:-/fs/atipa/app/rl9.x/python3/3.11.7/bin/activate}"
source "$ACTIVATE"

POC_DIR="${LENS_POC_DIR:-$HOME/lens_poc}"

OUTPUT_DIR="$1"
MASTER_ADDR="$2"
MASTER_PORT="$3"

RANK="${SLURM_PROCID:-0}"
HOST="$(hostname)"

IB_DEVICE="${IB_DEVICE:-mlx4_0}"
IB_PORT="${IB_PORT:-1}"

STRESS_ENABLE="${STRESS_ENABLE:-1}"
STRESS_DELAY_S="${STRESS_DELAY_S:-30}"
STRESS_DURATION_S="${STRESS_DURATION_S:-90}"

DDP_PARAMETER_MB="${DDP_PARAMETER_MB:-64}"
DDP_LOG_EVERY="${DDP_LOG_EVERY:-1}"

LIVE_ENABLE="${LIVE_ENABLE:-1}"
LIVE_BATCH_SECONDS="${LIVE_BATCH_SECONDS:-5}"
LIVE_HISTORY_SECONDS="${LIVE_HISTORY_SECONDS:-30}"

MODEL_DIR="${HEALTH_MODEL_DIR:-$HOME/shai-model-test}"
MODEL_ACTIVATE="${HEALTH_MODEL_ACTIVATE:-$HOME/venvs/shai-model-py311/bin/activate}"
WATCHER="$MODEL_DIR/live_watch.py"

export NCCL_IB_TIMEOUT="${NCCL_IB_TIMEOUT:-24}"
export NCCL_IB_RETRY_CNT="${NCCL_IB_RETRY_CNT:-7}"
export NCCL_DEBUG="${NCCL_DEBUG:-INFO}"
export NCCL_DEBUG_SUBSYS="${NCCL_DEBUG_SUBSYS:-INIT,NET}"

unset LD_PRELOAD IB_DELAY_US IB_DELAY_BLOCK_S IB_DELAY_WINDOW_S
unset IB_DELAY_PERIOD_S IB_DELAY_LOG

POLLER="$POC_DIR/telemetry_poller.py"
DDP="$POC_DIR/ddp_comm_workload_instrumented.py"

STRESS_LOG="$OUTPUT_DIR/rdma_stress_${HOST}.log"
STRESS_CSV="$OUTPUT_DIR/stress_active_${HOST}.csv"
LIVE_LOG="$OUTPUT_DIR/live_agent_${HOST}.log"

mkdir -p "$OUTPUT_DIR"
mkdir -p "$OUTPUT_DIR/live"

printf 'timestamp,hostname,stress_active,tool\n' > "$STRESS_CSV"

mark_stress() {
    printf '%s,%s,%s,ib_write_bw\n' \
        "$(date +%s.%N)" "$HOST" "$1" >> "$STRESS_CSV"
}

run_stress_server() {
    sleep "$STRESS_DELAY_S"
    mark_stress 1

    echo "[$HOST] RDMA server starting" >> "$STRESS_LOG"

    ib_write_bw \
        -d "$IB_DEVICE" \
        -i "$IB_PORT" \
        -b \
        -D "$STRESS_DURATION_S" \
        >> "$STRESS_LOG" 2>&1

    rc=$?

    mark_stress 0
    echo "[$HOST] RDMA server exited rc=$rc" >> "$STRESS_LOG"
}

run_stress_client() {
    sleep "$((STRESS_DELAY_S + 2))"
    mark_stress 1

    echo "[$HOST] RDMA client connecting to $MASTER_ADDR" \
        >> "$STRESS_LOG"

    ib_write_bw "$MASTER_ADDR" \
        -d "$IB_DEVICE" \
        -i "$IB_PORT" \
        -b \
        -D "$STRESS_DURATION_S" \
        >> "$STRESS_LOG" 2>&1

    rc=$?

    mark_stress 0
    echo "[$HOST] RDMA client exited rc=$rc" >> "$STRESS_LOG"
}

STRESS_PID=""
LIVE_PID=""
POLLER_PID=""

if [[ "$STRESS_ENABLE" == "1" ]]; then
    if [[ "$RANK" == "0" ]]; then
        run_stress_server &
    else
        run_stress_client &
    fi

    STRESS_PID=$!
fi

start_live_watcher() {
    if [[ "$LIVE_ENABLE" != "1" ]]; then
        echo "[$HOST] Live watcher disabled" >> "$LIVE_LOG"
        return 0
    fi

    if [[ "$RANK" != "0" ]]; then
        return 0
    fi

    if [[ ! -f "$MODEL_ACTIVATE" ]]; then
        echo "[$HOST] Model environment not found: $MODEL_ACTIVATE" \
            >> "$LIVE_LOG"
        return 0
    fi

    if [[ ! -f "$WATCHER" ]]; then
        echo "[$HOST] Live watcher not found: $WATCHER" \
            >> "$LIVE_LOG"
        return 0
    fi

    (
        source "$MODEL_ACTIVATE"
        cd "$MODEL_DIR" || exit 1

        exec python3 -u "$WATCHER" \
            --run-dir "$OUTPUT_DIR" \
            --output-dir "$OUTPUT_DIR/live" \
            --batch-seconds "$LIVE_BATCH_SECONDS" \
            --history-seconds "$LIVE_HISTORY_SECONDS"
    ) > "$LIVE_LOG" 2>&1 &

    LIVE_PID=$!

    echo "[$HOST] Live watcher started PID=$LIVE_PID" \
        >> "$LIVE_LOG"
}

cleanup() {
    if [[ -n "${LIVE_PID:-}" ]]; then
        kill "$LIVE_PID" 2>/dev/null || true
        wait "$LIVE_PID" 2>/dev/null || true
    fi

    if [[ -n "${STRESS_PID:-}" ]]; then
        kill "$STRESS_PID" 2>/dev/null || true
        wait "$STRESS_PID" 2>/dev/null || true
    fi

    if [[ -n "${POLLER_PID:-}" ]]; then
        kill "$POLLER_PID" 2>/dev/null || true
        wait "$POLLER_PID" 2>/dev/null || true
    fi
}

trap cleanup EXIT INT TERM

echo "[$HOST] extensive Label 2 rank=$RANK stress=$STRESS_ENABLE"
echo "[$HOST] stress delay=${STRESS_DELAY_S}s duration=${STRESS_DURATION_S}s"
echo "[$HOST] DDP parameter=${DDP_PARAMETER_MB}MB"
echo "[$HOST] IB device=${IB_DEVICE}:${IB_PORT}"
echo "[$HOST] Live watcher=$LIVE_ENABLE"

python3 "$POLLER" \
    --output-dir "$OUTPUT_DIR" \
    --interval 0.5 \
    --duration 1700 \
    --ib-device "$IB_DEVICE" \
    --ib-port "$IB_PORT" &

POLLER_PID=$!

start_live_watcher

python3 -u "$DDP" \
    --epochs 99999 \
    --parameter-mb "$DDP_PARAMETER_MB" \
    --log-every "$DDP_LOG_EVERY" \
    --master-addr "$MASTER_ADDR" \
    --master-port "$MASTER_PORT" \
    2>&1

DDP_EXIT=$?

exit "$DDP_EXIT"
