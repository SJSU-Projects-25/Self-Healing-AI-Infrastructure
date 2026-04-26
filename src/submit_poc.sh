#!/bin/bash
#SBATCH --job-name=lens_poc
#SBATCH --nodes=3
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00
#SBATCH --mem=32G
#SBATCH --nodelist=g8,g10,g11
#SBATCH --output=lens_poc_%j.log
#SBATCH --error=lens_poc_%j.err
#SBATCH --partition=gpuqs

POC_DIR="$HOME/lens_poc"
OUTPUT_DIR="$POC_DIR/results/${SLURM_JOB_ID}"
mkdir -p "$OUTPUT_DIR"

MASTER_NODE=$(scontrol show hostnames "$SLURM_NODELIST" | head -n 1)
export MASTER_ADDR="$MASTER_NODE"
export MASTER_PORT=29500

echo "============================================"
echo " LENS POC — Job $SLURM_JOB_ID"
echo " Nodes : $SLURM_NODELIST"
echo " Master: $MASTER_ADDR"
echo " Output: $OUTPUT_DIR"
echo "============================================"

echo "[$(date)] Launching poller + DDP on all nodes..."

# Single srun — wrapper runs poller in background + DDP in foreground on each node
srun --ntasks=3 \
     --output="$OUTPUT_DIR/node_%n_%t.log" \
     bash ~/lens_poc/run_node.sh "$OUTPUT_DIR" "$MASTER_ADDR" "$MASTER_PORT"

echo "[$(date)] All nodes finished."
echo ""
echo "============================================"
echo " POC complete. Results in: $OUTPUT_DIR"
echo " CSV files:"
ls "$OUTPUT_DIR"/*.csv 2>/dev/null || echo "  (no CSVs yet — check logs)"
echo "============================================"
