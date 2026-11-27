#!/bin/bash

#SBATCH --job-name=lens_l2ext
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --time=00:08:00
#SBATCH --mem=32G
#SBATCH --nodelist=g8,g11
#SBATCH --partition=gpuqs

set -u

POC_DIR="${LENS_POC_DIR:-$HOME/lens_poc}"

OUTPUT_DIR="$POC_DIR/results/label2_extensive_${SLURM_JOB_ID}"

mkdir -p "$OUTPUT_DIR"

MASTER_ADDR="$(scontrol show hostnames "$SLURM_NODELIST" | head -n 1)"
MASTER_PORT=$((29500 + SLURM_JOB_ID % 1000))

export MASTER_ADDR
export MASTER_PORT

export IB_DEVICE="${IB_DEVICE:-mlx4_0}"
export IB_PORT="${IB_PORT:-1}"

export STRESS_ENABLE="${STRESS_ENABLE:-1}"
export STRESS_DELAY_S="${STRESS_DELAY_S:-30}"
export STRESS_DURATION_S="${STRESS_DURATION_S:-90}"

export DDP_PARAMETER_MB="${DDP_PARAMETER_MB:-64}"
export DDP_LOG_EVERY="${DDP_LOG_EVERY:-1}"

export LIVE_ENABLE="${LIVE_ENABLE:-1}"
export LIVE_BATCH_SECONDS="${LIVE_BATCH_SECONDS:-5}"
export LIVE_HISTORY_SECONDS="${LIVE_HISTORY_SECONDS:-30}"

export HEALTH_MODEL_DIR="${HEALTH_MODEL_DIR:-$HOME/shai-model-test}"
export HEALTH_MODEL_ACTIVATE="${HEALTH_MODEL_ACTIVATE:-$HOME/venvs/shai-model-py311/bin/activate}"

RUNNER="$POC_DIR/run_node_label2_extensive.sh"

echo "============================================"
echo "LENS Label 2 — extensive IB contention run"
echo "Job       : $SLURM_JOB_ID"
echo "Nodes     : $SLURM_NODELIST"
echo "Output    : $OUTPUT_DIR"
echo "Stress    : enable=$STRESS_ENABLE delay=${STRESS_DELAY_S}s duration=${STRESS_DURATION_S}s"
echo "DDP       : parameter=${DDP_PARAMETER_MB}MB log_every=${DDP_LOG_EVERY}"
echo "IB        : ${IB_DEVICE}:${IB_PORT}"
echo "Live      : enable=$LIVE_ENABLE batch=${LIVE_BATCH_SECONDS}s history=${LIVE_HISTORY_SECONDS}s"
echo "Model     : $HEALTH_MODEL_DIR"
echo "============================================"

timeout --signal=TERM 360s \
    srun \
    --ntasks=2 \
    --output="$OUTPUT_DIR/node_%n_%t.log" \
    bash "$RUNNER" \
    "$OUTPUT_DIR" \
    "$MASTER_ADDR" \
    "$MASTER_PORT"

echo "Runner exit code: $?"

ls -lh "$OUTPUT_DIR"
