#!/usr/bin/env bash
# Submit serial 4h jobs for the four-endpoint resumable standard evaluation.

set -euo pipefail
umask 007

export WORK=${WORK:-/lustre/fsw/portfolios/general/users/jiaychen}
export MODEL_PATH=${MODEL_PATH:-$WORK/checkpoints/Qwen3VL_4B_Final_swift}
export RESULT_NAME=${RESULT_NAME:-final-swift-val-unseen-pass4-fast4}
export SEED_RESULT=${SEED_RESULT:-$WORK/results/final-swift-val-unseen-pass4/result.json}
export PASS_K=4
export VLLM_GPU_IDS=${VLLM_GPU_IDS:-0,1,2,3}
export RENDER_GPU_IDS=${RENDER_GPU_IDS:-4,5,6,7}
export WORKERS_PER_SHARD=${WORKERS_PER_SHARD:-12}
export VLLM_MAX_NUM_SEQS=${VLLM_MAX_NUM_SEQS:-16}
export VLLM_MM_CACHE_GB=${VLLM_MM_CACHE_GB:-8}

SBATCH_SCRIPT=${SBATCH_SCRIPT:-$WORK/WorldModel/scripts/env4_standard_eval_4way.sbatch}
EVAL_JOBS=${EVAL_JOBS:-4}
SBATCH_TIME=${SBATCH_TIME:-04:00:00}
INITIAL_DEPENDENCY=${INITIAL_DEPENDENCY:-}
JOB_NAME=${JOB_NAME:-std-eval-4way-${RESULT_NAME}}
CHAIN_FILE=${CHAIN_FILE:-$WORK/logs/${RESULT_NAME}-job-chain.txt}
DONE=$WORK/results/$RESULT_NAME/EVAL_COMPLETE

if [[ -f "$DONE" ]]; then
  echo "Evaluation already complete: $DONE"
  exit 0
fi

bash -n "$SBATCH_SCRIPT"
mkdir -p "$WORK/logs"
: > "$CHAIN_FILE"

previous_job=$INITIAL_DEPENDENCY
for _ in $(seq 1 "$EVAL_JOBS"); do
  dependency_args=()
  if [[ -n "$previous_job" ]]; then
    dependency_args=(--dependency="afterany:$previous_job")
  fi
  job_id=$(sbatch --parsable --job-name="$JOB_NAME" --time="$SBATCH_TIME" \
    --export=ALL "${dependency_args[@]}" "$SBATCH_SCRIPT")
  echo "$job_id" >> "$CHAIN_FILE"
  previous_job=$job_id
done

paste -sd '>' "$CHAIN_FILE"
echo "Submitted $EVAL_JOBS serial four-endpoint jobs for $RESULT_NAME"
