#!/bin/bash
# ─────────────────────────────────────────────────────────────────
# LENS — Idle Data Collection (Label 5)
# Runs ONLY the telemetry poller — no DDP workload.
# GPU stays idle, IB shows only background traffic.
# ─────────────────────────────────────────────────────────────────

ACTIVATE="/fs/atipa/app/rl9.x/python3/3.11.7/bin/activate"
source $ACTIVATE

POC_DIR="$HOME/lens_poc"
OUTPUT_DIR="$1"
DURATION="${2:-800}"   # default 800 seconds (~13 minutes)

echo "[$(hostname)] Starting idle poller — no DDP workload"
echo "[$(hostname)] Duration: ${DURATION}s"

# Run poller only — label=idle
python3 $POC_DIR/telemetry_poller.py \
    --output-dir $OUTPUT_DIR \
    --interval 0.5 \
    --duration $DURATION \
    --ib-device mlx4_0 \
    --ib-port 1 \

echo "[$(hostname)] Idle collection done."
