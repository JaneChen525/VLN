#!/usr/bin/env bash
# Submit the env4 GRPO chain with spare jobs for automatic recovery.

set -euo pipefail
umask 007

WORK=${WORK:-/lustre/fsw/portfolios/general/users/jiaychen}
SBATCH_SCRIPT=${SBATCH_SCRIPT:-$WORK/WorldModel/scripts/env4_grpo_4step.sbatch}
EXP=${EXP:-p13-env4-final-mrope-traj-grpo-32x8}
TOTAL_STEPS=${TOTAL_STEPS:-30}
PLANNED_JOBS=${PLANNED_JOBS:-8}
BUFFER_JOBS=${BUFFER_JOBS:-2}
JOB_NAME=${JOB_NAME:-${EXP}-4step}
CHAIN_FILE=${CHAIN_FILE:-$WORK/logs/${EXP}-job-chain.txt}
TRACKER=$WORK/WorldModel/checkpoints/vln-grpo/$EXP/latest_checkpointed_iteration.txt
SBATCH_EXPORT="ALL,EXP=$EXP,TOTAL_STEPS=$TOTAL_STEPS"

bash -n "$SBATCH_SCRIPT"
mkdir -p "$WORK/logs"

current_step=0
if [[ -s "$TRACKER" ]]; then
  current_step=$(tr -d '[:space:]' < "$TRACKER")
fi
if ! [[ "$current_step" =~ ^[0-9]+$ ]]; then
  echo "Invalid checkpoint tracker: $current_step" >&2
  exit 1
fi
if (( current_step >= TOTAL_STEPS )); then
  echo "Training already complete: step=$current_step"
  exit 0
fi

job_count=$((PLANNED_JOBS + BUFFER_JOBS))
job_ids=()
previous_job=""
for _ in $(seq 1 "$job_count"); do
  if [[ -z "$previous_job" ]]; then
    job_id=$(sbatch --parsable --job-name="$JOB_NAME" --export="$SBATCH_EXPORT" "$SBATCH_SCRIPT")
  else
    job_id=$(sbatch --parsable --job-name="$JOB_NAME" --export="$SBATCH_EXPORT" \
      --dependency="afterany:$previous_job" "$SBATCH_SCRIPT")
  fi
  job_ids+=("$job_id")
  previous_job=$job_id
done

printf '%s\n' "${job_ids[*]}" | tee "$CHAIN_FILE"
echo "Submitted $PLANNED_JOBS planned + $BUFFER_JOBS buffer jobs."
echo "CHAIN=$(IFS='>'; echo "${job_ids[*]}")"
