#!/usr/bin/env bash
# Submit serial 4h standard-eval jobs. Each segment resumes from RESULT_DIR/result.json.

set -euo pipefail
umask 007

export WORK=${WORK:-/lustre/fsw/portfolios/general/users/jiaychen}
export MODEL_PATH=${MODEL_PATH:-$WORK/checkpoints/Qwen3VL_4B_Final_swift}
export RESULT_NAME=${RESULT_NAME:-final-swift-val-unseen-pass4}
export PASS_K=${PASS_K:-4}
export SPLIT_NUM=${SPLIT_NUM:-24}
export VLLM_GPU_IDS=${VLLM_GPU_IDS:-0}
export VLLM_DP=${VLLM_DP:-1}
export VLLM_TP=${VLLM_TP:-1}
export RENDER_GPU=${RENDER_GPU:-1}

SBATCH_SCRIPT=${SBATCH_SCRIPT:-$WORK/WorldModel/scripts/env4_standard_eval.sbatch}
EVAL_JOBS=${EVAL_JOBS:-8}
SBATCH_TIME=${SBATCH_TIME:-04:00:00}
INITIAL_DEPENDENCY=${INITIAL_DEPENDENCY:-}
JOB_NAME=${JOB_NAME:-std-eval-${RESULT_NAME}}
CHAIN_FILE=${CHAIN_FILE:-$WORK/logs/${RESULT_NAME}-job-chain.txt}
DONE=$WORK/results/$RESULT_NAME/EVAL_COMPLETE

if [[ "$PASS_K" != 4 ]]; then
  echo "Standard eval requires PASS_K=4, got $PASS_K" >&2
  exit 2
fi
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
echo "Submitted $EVAL_JOBS serial standard-eval jobs for $RESULT_NAME"
