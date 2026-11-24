#!/usr/bin/env bash
set -euo pipefail

ACTIVATE="${LENS_ACTIVATE:-/fs/atipa/app/rl9.x/python3/3.11.7/bin/activate}"

if [[ ! -f "$ACTIVATE" ]]; then
  echo "ERROR: PyTorch environment not found at $ACTIVATE" >&2
  exit 1
fi

source "$ACTIVATE"

OUTPUT_DIR=$1
MASTER_ADDR=$2
MASTER_PORT=$3

POC_DIR=${POC_DIR:-${SLURM_SUBMIT_DIR:-$HOME/lens_poc}}

RUN_SECONDS=${RUN_SECONDS:-300}
PAUSE_RANK=${PAUSE_RANK:-1}
PAUSE_EVERY=${PAUSE_EVERY:-200}
PAUSE_SECONDS=${PAUSE_SECONDS:-0.25}
DDP_PARAMETER_MB=${DDP_PARAMETER_MB:-64}
IB_DEVICE=${IB_DEVICE:-mlx4_0}
IB_PORT=${IB_PORT:-1}

mkdir -p "$OUTPUT_DIR"

echo "[$(hostname)] Label 1 rank=${SLURM_PROCID:-0}"
echo "[$(hostname)] pause_rank=${PAUSE_RANK}"
echo "[$(hostname)] run=${RUN_SECONDS}s"
echo "[$(hostname)] pause_every=${PAUSE_EVERY}"
echo "[$(hostname)] pause_seconds=${PAUSE_SECONDS}"

python3 "$POC_DIR/telemetry_poller.py" \
  --output-dir "$OUTPUT_DIR" \
  --interval 0.5 \
  --duration "$((RUN_SECONDS + 30))" \
  --ib-device "$IB_DEVICE" \
  --ib-port "$IB_PORT" &

POLLER_PID=$!

cleanup() {
  kill "$POLLER_PID" 2>/dev/null || true
  wait "$POLLER_PID" 2>/dev/null || true
}

trap cleanup EXIT

python3 "$POC_DIR/ddp_label1_workload.py" \
  --output-dir "$OUTPUT_DIR" \
  --master-addr "$MASTER_ADDR" \
  --master-port "$MASTER_PORT" \
  --run-seconds "$RUN_SECONDS" \
  --pause-rank "$PAUSE_RANK" \
  --pause-every "$PAUSE_EVERY" \
  --pause-seconds "$PAUSE_SECONDS" \
  --parameter-mb "$DDP_PARAMETER_MB"
