#!/bin/bash
ACTIVATE="/fs/atipa/app/rl9.x/python3/3.11.7/bin/activate"
source $ACTIVATE

POC_DIR="$HOME/lens_poc"
OUTPUT_DIR="$1"
MASTER_ADDR="$2"
MASTER_PORT="$3"

# Start poller in background on this node
python3 $POC_DIR/telemetry_poller.py \
    --output-dir $OUTPUT_DIR \
    --interval 0.5 \
    --duration 1800 \
    --ib-device mlx4_0 \
    --ib-port 1 &
POLLER_PID=$!

echo "[$(hostname)] Poller started with PID $POLLER_PID"

# Run DDP workload
python3 $POC_DIR/ddp_workload.py \
    --epochs 99999 \
    --batch-size 16 \
    --master-addr $MASTER_ADDR \
    --master-port $MASTER_PORT

# Stop poller when DDP finishes
kill $POLLER_PID 2>/dev/null || true
echo "[$(hostname)] Done."
