#!/bin/bash
# ─────────────────────────────────────────────────────────────────
# LENS — Network Stall Data Collection (Label 2)
# Application-layer stall injection via sleep after AllReduce.
# Stall rank 1 sleeps every 5 steps for 4 seconds.
# All nodes stall waiting — produces Label 2 telemetry fingerprint.
# ─────────────────────────────────────────────────────────────────

ACTIVATE="/fs/atipa/app/rl9.x/python3/3.11.7/bin/activate"
source $ACTIVATE

POC_DIR="$HOME/lens_poc"
OUTPUT_DIR="$1"
MASTER_ADDR="$2"
MASTER_PORT="$3"

# Stall parameters
STALL_RANK=1        # rank 1 (g11) injects the stall
STALL_EVERY=5       # stall every 5 steps
STALL_DURATION=4.0  # sleep 4 seconds per stall

echo "[$(hostname)] Starting poller for network stall collection..."

# Start poller in background
python3 $POC_DIR/telemetry_poller.py \
    --output-dir $OUTPUT_DIR \
    --interval 0.5 \
    --duration 1700 \
    --ib-device mlx4_0 \
    --ib-port 1 &
POLLER_PID=$!

echo "[$(hostname)] Poller started with PID $POLLER_PID"

# Run stall workload
echo "[$(hostname)] Starting DDP with stall injection — master=$MASTER_ADDR:$MASTER_PORT"
echo "[$(hostname)] Stall config: rank=$STALL_RANK every=$STALL_EVERY duration=${STALL_DURATION}s"

python3 $POC_DIR/ddp_workload_stall.py \
    --epochs 99999 \
    --batch-size 16 \
    --master-addr $MASTER_ADDR \
    --master-port $MASTER_PORT \
    --stall-rank $STALL_RANK \
    --stall-every $STALL_EVERY \
    --stall-duration $STALL_DURATION 2>&1

DDP_EXIT=$?
echo "[$(hostname)] DDP stall workload exited with code $DDP_EXIT"

# Stop poller
kill $POLLER_PID 2>/dev/null || true
echo "[$(hostname)] Done."
