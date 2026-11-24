#!/usr/bin/env bash

#SBATCH --job-name=lens_l1
#SBATCH --partition=gpuqs
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --time=00:10:00
#SBATCH --mem=32G
#SBATCH --output=lens_label1_%j.log
#SBATCH --error=lens_label1_%j.err

set -euo pipefail

POC_DIR=${SLURM_SUBMIT_DIR:-$HOME/lens_poc}

OUTPUT_DIR="$POC_DIR/results/label1_${SLURM_JOB_ID}"

MASTER_ADDR=$(
  scontrol show hostnames "$SLURM_NODELIST" | head -n 1
)

MASTER_PORT=${MASTER_PORT:-$((29500 + SLURM_JOB_ID % 1000))}

export NCCL_DEBUG=${NCCL_DEBUG:-INFO}
export NCCL_DEBUG_SUBSYS=${NCCL_DEBUG_SUBSYS:-NET}

export RUN_SECONDS=${RUN_SECONDS:-300}
export PAUSE_RANK=${PAUSE_RANK:-1}
export PAUSE_EVERY=${PAUSE_EVERY:-200}
export PAUSE_SECONDS=${PAUSE_SECONDS:-0.25}
export DDP_PARAMETER_MB=${DDP_PARAMETER_MB:-64}

export IB_DEVICE=${IB_DEVICE:-mlx4_0}
export IB_PORT=${IB_PORT:-1}

mkdir -p "$OUTPUT_DIR"

echo "Label 1 job=${SLURM_JOB_ID}"
echo "Nodes=${SLURM_NODELIST}"
echo "Master=${MASTER_ADDR}:${MASTER_PORT}"
echo "Output=${OUTPUT_DIR}"
echo "Run seconds=${RUN_SECONDS}"
echo "Pause rank=${PAUSE_RANK}"
echo "Pause every=${PAUSE_EVERY}"
echo "Pause seconds=${PAUSE_SECONDS}"

srun --ntasks=2 \
  --output="$OUTPUT_DIR/node_%n_%t.log" \
  bash "$POC_DIR/run_node_label1.sh" \
  "$OUTPUT_DIR" \
  "$MASTER_ADDR" \
  "$MASTER_PORT"
