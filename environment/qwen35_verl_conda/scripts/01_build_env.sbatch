#!/bin/bash
#SBATCH --job-name=q35-conda-build
#SBATCH --account=general_perflab
#SBATCH --partition=polar,polar3,polar4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=30
#SBATCH --mem=220G
#SBATCH --time=04:00:00
#SBATCH --output=/lustre/fsw/portfolios/general/users/jiaychen/logs/%x-%j.log
#SBATCH --error=/lustre/fsw/portfolios/general/users/jiaychen/logs/%x-%j.log

set -euo pipefail
WORK=${WORK:-/lustre/fsw/portfolios/general/users/jiaychen}
WORLDMODEL=${WORLDMODEL:-$WORK/WorldModel}
SCRIPTS_DIR=${QWEN35_SCRIPTS_DIR:-$WORLDMODEL/scripts/qwen35_conda}
mkdir -p "$WORK/logs"

echo "timestamp=$(date -Is) host=$(hostname) job=$SLURM_JOB_ID"
nvidia-smi --query-gpu=name,driver_version,memory.total,compute_cap --format=csv,noheader
WORK="$WORK" WORLDMODEL="$WORLDMODEL" QWEN35_SCRIPTS_DIR="$SCRIPTS_DIR" \
  bash "$SCRIPTS_DIR/build_env.sh"
