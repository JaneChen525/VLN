#!/bin/bash
#SBATCH --job-name=q35-conda-recreate
#SBATCH --account=general_perflab
#SBATCH --partition=polar,polar3,polar4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=30
#SBATCH --mem=128G
#SBATCH --time=04:00:00
#SBATCH --output=/lustre/fsw/portfolios/general/users/jiaychen/logs/%x-%j.log
#SBATCH --error=/lustre/fsw/portfolios/general/users/jiaychen/logs/%x-%j.log

set -euo pipefail
WORK=${WORK:-/lustre/fsw/portfolios/general/users/jiaychen}
WORLDMODEL=${WORLDMODEL:-$WORK/WorldModel}
ARTIFACT_DIR=${MANIFEST_DIR_OVERRIDE:-$WORLDMODEL/artifacts/qwen35-conda-env}
RECREATED_ENV=${RECREATED_ENV_OVERRIDE:-$WORK/conda-envs/qwen35-verl-vllm018-recreated}
CONDA_ROOT=${CONDA_ROOT_OVERRIDE:-$WORK/miniconda3}
VERL_SRC=${VERL_SRC_OVERRIDE:-$WORLDMODEL/vln/reinforcement_learning}
MODEL=${MODEL_PATH_OVERRIDE:-$WORK/checkpoints/Qwen3.5-4B-base}
QWEN3VL_MODEL=${QWEN3VL_MODEL_PATH_OVERRIDE:-$WORK/checkpoints/Qwen3VL_4B_Final_swift}
RUN_ID=${RECREATE_RUN_ID:-$SLURM_JOB_ID}

bash "$ARTIFACT_DIR/recreate_env.sh" \
  "$CONDA_ROOT" "$RECREATED_ENV" "$ARTIFACT_DIR" "$WORLDMODEL" "$VERL_SRC"
"$RECREATED_ENV/bin/python" "$ARTIFACT_DIR/scripts/check_stack.py" \
  --model "$MODEL" --expect-model-type qwen3_5 --run-kernel
"$RECREATED_ENV/bin/python" "$ARTIFACT_DIR/scripts/check_stack.py" \
  --model "$QWEN3VL_MODEL" --expect-model-type qwen3_vl

RESULT_DIR="$WORLDMODEL/results/qwen-conda-smoke/recreate/$RUN_ID"
mkdir -p "$RESULT_DIR"
{
  echo "timestamp=$(date -Is)"
  echo "job_id=$SLURM_JOB_ID"
  echo "run_id=$RUN_ID"
  echo "env=$RECREATED_ENV"
  echo 'status=RECREATE_KERNEL_AND_DUAL_CONFIG_CHECK_SUCCESS'
} > "$RESULT_DIR/RECREATE_SUCCESS"
echo "RECREATE_KERNEL_AND_DUAL_CONFIG_CHECK_SUCCESS env=$RECREATED_ENV result=$RESULT_DIR"
