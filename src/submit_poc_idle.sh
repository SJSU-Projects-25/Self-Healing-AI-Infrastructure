#!/bin/bash
#SBATCH --job-name=lens_idle
#SBATCH --nodes=3
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --time=00:20:00
#SBATCH --mem=16G
#SBATCH --nodelist=g2,g5,g7
#SBATCH --output=lens_idle_%j.log
#SBATCH --error=lens_idle_%j.err
#SBATCH --partition=gpuqs

# ─────────────────────────────────────────────────────────────────
# LENS — Idle Data Collection Job (Label 5)
# Collects telemetry with GPU completely idle — no training job.
# Produces baseline noise floor data for ML classifier.
# ─────────────────────────────────────────────────────────────────

POC_DIR="$HOME/lens_poc"
OUTPUT_DIR="$POC_DIR/results/idle_${SLURM_JOB_ID}"
mkdir -p "$OUTPUT_DIR"

DURATION=800   # 800 seconds = ~13 minutes of idle data

echo "============================================"
echo " LENS IDLE Collection — Job $SLURM_JOB_ID"
echo " Nodes  : $SLURM_NODELIST"
echo " Output : $OUTPUT_DIR"
echo " Duration: ${DURATION}s per node"
echo "============================================"

echo "[$(date)] Starting idle telemetry collection..."

srun --ntasks=3 \
     --output="$OUTPUT_DIR/node_%n_%t.log" \
     bash ~/lens_poc/run_node_idle.sh "$OUTPUT_DIR" "$DURATION"

echo "[$(date)] Idle collection complete."
echo ""
echo "============================================"
echo " Results in: $OUTPUT_DIR"
echo " CSV files:"
ls "$OUTPUT_DIR"/*.csv 2>/dev/null || echo "  (no CSVs — check logs)"
echo ""
echo " Quick validation:"
echo "   wc -l $OUTPUT_DIR/*.csv"
echo "   awk -F',' 'NR>1 {print \$6}' $OUTPUT_DIR/telemetry_*.csv | sort | uniq -c"
echo "============================================"
