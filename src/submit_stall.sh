#!/bin/bash
#SBATCH --job-name=lens_stall
#SBATCH --nodes=3
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00
#SBATCH --mem=32G
#SBATCH --nodelist=g4,g5,g11
#SBATCH --output=lens_stall_%j.log
#SBATCH --error=lens_stall_%j.err
#SBATCH --partition=gpuqs

# ─────────────────────────────────────────────────────────────────
# LENS — Network Stall Data Collection (Label 2)
# Runs DDP with application-layer stall injection on rank 1.
# Produces telemetry fingerprint identical to real network delay:
#   - gpu_util drops to 0% on ALL nodes during stall
#   - xmit_wait_delta spikes on stall node
#   - xmit_data_delta drops across all nodes
# ─────────────────────────────────────────────────────────────────

POC_DIR="$HOME/lens_poc"
OUTPUT_DIR="$POC_DIR/results/stall_${SLURM_JOB_ID}"
mkdir -p "$OUTPUT_DIR"

MASTER_NODE=$(scontrol show hostnames "$SLURM_NODELIST" | head -n 1)
export MASTER_ADDR="$MASTER_NODE"
export MASTER_PORT=29500

echo "============================================"
echo " LENS STALL Collection — Job $SLURM_JOB_ID"
echo " Nodes  : $SLURM_NODELIST"
echo " Master : $MASTER_ADDR"
echo " Output : $OUTPUT_DIR"
echo " Stall  : rank=1, every=5 steps, duration=4s"
echo "============================================"

echo "[$(date)] Launching stall workload on all nodes..."

srun --ntasks=3 \
     --output="$OUTPUT_DIR/node_%n_%t.log" \
     bash ~/lens_poc/run_node_stall.sh \
         "$OUTPUT_DIR" "$MASTER_ADDR" "$MASTER_PORT"

echo "[$(date)] Stall collection complete."
echo ""
echo "============================================"
echo " Results in: $OUTPUT_DIR"
echo " CSV files:"
ls "$OUTPUT_DIR"/*.csv 2>/dev/null || echo "  (no CSVs — check logs)"
echo ""
echo " Quick validation:"
echo "   wc -l $OUTPUT_DIR/*.csv"
echo "   awk -F',' 'NR>1 {print \$6, \$30}' $OUTPUT_DIR/telemetry_*.csv | tail -20"
echo "============================================"
