#!/bin/bash
# One 8-GPU allocation for every post-build acceptance gate.
#SBATCH --job-name=q35-conda-accept
#SBATCH --account=general_perflab
#SBATCH --partition=polar,polar3,polar4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=8
#SBATCH --cpus-per-task=96
#SBATCH --mem=0
#SBATCH --time=04:00:00
#SBATCH --output=/lustre/fsw/portfolios/general/users/jiaychen/logs/%x-%j.log
#SBATCH --error=/lustre/fsw/portfolios/general/users/jiaychen/logs/%x-%j.log

set -euo pipefail
umask 007

WORK=${WORK:-/lustre/fsw/portfolios/general/users/jiaychen}
WORLDMODEL=${WORLDMODEL:-$WORK/WorldModel}
SCRIPTS_DIR=${QWEN35_SCRIPTS_DIR:-$WORLDMODEL/scripts/qwen35_conda}
ORIGINAL_ENV=${TRAIN_ENV_OVERRIDE:-$WORK/conda-envs/qwen35-verl-vllm018}
RECREATED_ENV=${RECREATED_ENV_OVERRIDE:-$WORK/conda-envs/qwen35-verl-vllm018-recreated-$SLURM_JOB_ID}
QWEN35_MODEL=${QWEN35_MODEL_PATH_OVERRIDE:-$WORK/checkpoints/Qwen3.5-4B-base}
QWEN3VL_MODEL=${QWEN3VL_MODEL_PATH_OVERRIDE:-$WORK/checkpoints/Qwen3VL_4B_Final_swift}

ORIGINAL_Q35_RUN="$SLURM_JOB_ID-original-qwen35"
RECREATE_RUN="$SLURM_JOB_ID-recreate"
FRESH_Q35_RUN="$SLURM_JOB_ID-recreated-qwen35"
FRESH_Q3VL_RUN="$SLURM_JOB_ID-recreated-qwen3vl"

stop_ray() {
  local env_prefix=$1
  if [[ -x "$env_prefix/bin/ray" ]]; then
    "$env_prefix/bin/ray" stop --force >/dev/null 2>&1 || true
  fi
}
trap 'stop_ray "$ORIGINAL_ENV"; stop_ray "$RECREATED_ENV"' EXIT

echo "ACCEPTANCE_START job=$SLURM_JOB_ID host=$(hostname) recreated_env=$RECREATED_ENV"

# Maintainer-only recovery path for an environment created before the pinned
# Qwen3.5/FA2 guard was added to build_env.sh. A normal clean build/recreate
# already applies the patch and should leave PREPARE_EXISTING_ENV unset.
if [[ "${PREPARE_EXISTING_ENV:-0}" == 1 ]]; then
  echo "PREPARE_EXISTING_ENV_START env=$ORIGINAL_ENV"
  "$ORIGINAL_ENV/bin/python" "$SCRIPTS_DIR/patch_transformers_qwen35_fa.py"
  CUDA_LAUNCH_BLOCKING=1 \
    "$ORIGINAL_ENV/bin/python" "$SCRIPTS_DIR/check_stack.py" --run-kernel
  WORK="$WORK" ENV_PREFIX="$ORIGINAL_ENV" WORLDMODEL="$WORLDMODEL" \
    QWEN35_SCRIPTS_DIR="$SCRIPTS_DIR" \
    bash "$SCRIPTS_DIR/refresh_artifact_scripts.sh"
  echo "PREPARE_EXISTING_ENV_SUCCESS env=$ORIGINAL_ENV"
fi

SMOKE_RUN_ID="$ORIGINAL_Q35_RUN" \
  RAY_TMPDIR_OVERRIDE="/tmp/q35-$SLURM_JOB_ID-o" \
  TRAIN_ENV_OVERRIDE="$ORIGINAL_ENV" \
  MODEL_FAMILY=qwen35 MODEL_PATH_OVERRIDE="$QWEN35_MODEL" \
  EXPERIMENT_PREFIX=qwen35-4b-original-smoke \
  bash "$SCRIPTS_DIR/02_rl_smoke.sbatch"
stop_ray "$ORIGINAL_ENV"

RECREATE_RUN_ID="$RECREATE_RUN" RECREATED_ENV_OVERRIDE="$RECREATED_ENV" \
  bash "$SCRIPTS_DIR/03_recreate_verify.sbatch"

SMOKE_RUN_ID="$FRESH_Q35_RUN" \
  RAY_TMPDIR_OVERRIDE="/tmp/q35-$SLURM_JOB_ID-f" \
  TRAIN_ENV_OVERRIDE="$RECREATED_ENV" \
  MODEL_FAMILY=qwen35 MODEL_PATH_OVERRIDE="$QWEN35_MODEL" \
  EXPERIMENT_PREFIX=qwen35-4b-recreated-smoke \
  bash "$SCRIPTS_DIR/02_rl_smoke.sbatch"
stop_ray "$RECREATED_ENV"

SMOKE_RUN_ID="$FRESH_Q3VL_RUN" \
  RAY_TMPDIR_OVERRIDE="/tmp/q35-$SLURM_JOB_ID-v" \
  TRAIN_ENV_OVERRIDE="$RECREATED_ENV" \
  MODEL_FAMILY=qwen3vl MODEL_PATH_OVERRIDE="$QWEN3VL_MODEL" \
  EXPERIMENT_PREFIX=qwen3vl-4b-recreated-smoke \
  bash "$SCRIPTS_DIR/02_rl_smoke.sbatch"
stop_ray "$RECREATED_ENV"

ORIGINAL_Q35_JOB="$ORIGINAL_Q35_RUN" \
  RECREATE_JOB="$RECREATE_RUN" \
  FRESH_Q35_JOB="$FRESH_Q35_RUN" \
  FRESH_Q3VL_JOB="$FRESH_Q3VL_RUN" \
  bash "$SCRIPTS_DIR/04_finalize_release.sbatch"

echo "FULL_ACCEPTANCE_SUCCESS job=$SLURM_JOB_ID"
