#!/bin/bash
#SBATCH --job-name=resume-smoke
#SBATCH --account=general_perflab
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --partition=polar,polar3,polar4
#SBATCH --time=00:30:00
#SBATCH --gpus=8
#SBATCH --output=/lustre/fsw/portfolios/general/users/jiaychen/logs/%x-%j.log
#SBATCH --error=/lustre/fsw/portfolios/general/users/jiaychen/logs/%x-%j.log

set -euo pipefail

WORK=/lustre/fsw/portfolios/general/users/jiaychen
WORLDMODEL=$WORK/WorldModel
EXP=resume-auto-smoke-env4-20260714
TOTAL_STEPS=2
STEPS_PER_JOB=1
SAVE_FREQ=1
TRACKER=$WORLDMODEL/checkpoints/vln-grpo/$EXP/latest_checkpointed_iteration.txt
ENV_LOG=$WORK/logs/env-server-${SLURM_JOB_ID}.log

mkdir -p "$WORK/logs" "$WORLDMODEL/wandb"

CURRENT_STEP=0
if [[ -s "$TRACKER" ]]; then
  CURRENT_STEP=$(tr -d '[:space:]' < "$TRACKER")
fi
if ! [[ "$CURRENT_STEP" =~ ^[0-9]+$ ]]; then
  echo "Invalid checkpoint tracker: $CURRENT_STEP"
  exit 1
fi
if (( CURRENT_STEP >= TOTAL_STEPS )); then
  echo "Resume smoke already complete: step=$CURRENT_STEP"
  exit 0
fi
TARGET_STEP=$((CURRENT_STEP + STEPS_PER_JOB))
if (( TARGET_STEP > TOTAL_STEPS )); then
  TARGET_STEP=$TOTAL_STEPS
fi
echo "Job segment: step $CURRENT_STEP -> $TARGET_STEP"

cd "$WORLDMODEL"
PYTHONPATH="$WORLDMODEL/vln/reinforcement_learning:$WORLDMODEL:$WORLDMODEL/vln:${PYTHONPATH:-}" \
  "$WORK/miniconda3/envs/vln/bin/python" \
  -m recipe.vln_navida.env_server.launch \
  --exp-config config/vln_r2r.yaml \
  --port 8002 \
  --pool-size 4 \
  --gpu-ids 0 \
  --split train \
  > "$ENV_LOG" 2>&1 &
ENV_PID=$!
TRAIN_PID=""

cleanup() {
  if [[ -n "$TRAIN_PID" ]]; then
    kill "$TRAIN_PID" 2>/dev/null || true
  fi
  kill "$ENV_PID" 2>/dev/null || true
  wait "$ENV_PID" 2>/dev/null || true
}
trap cleanup EXIT

READY=0
for _ in $(seq 1 120); do
  if ! kill -0 "$ENV_PID" 2>/dev/null; then
    echo "env_server exited unexpectedly"
    tail -n 100 "$ENV_LOG"
    exit 1
  fi

  STATS=$(curl -fsS http://127.0.0.1:8002/stats 2>/dev/null || true)
  if [[ "$STATS" == *'"free_workers":4'* ]]; then
    READY=1
    echo "env_server ready: $STATS"
    break
  fi
  sleep 5
done

if (( READY == 0 )); then
  echo "env_server startup timeout"
  tail -n 100 "$ENV_LOG"
  exit 1
fi

srun --overlap \
  --container-image=verlai/verl:vllm011.latest \
  --container-mounts=$WORK:$WORK \
  bash -c "
export WORLDMODEL=$WORLDMODEL
export AGENT_CFG=$WORLDMODEL/vln/reinforcement_learning/recipe/vln_navida/config/agent_loop.yaml
export PYTHONPATH=$WORLDMODEL/vln/reinforcement_learning:$WORLDMODEL:$WORLDMODEL/vln:\$PYTHONPATH
export HOME=$WORK/.wandb-home
export WANDB_MODE=online
export WANDB_DIR=$WORLDMODEL/wandb
export WANDB_RUN_ID=$EXP
export WANDB_RESUME=allow
export VLLM_MM_INPUT_CACHE_GIB=2
export ROLLOUT_SCHEDULER=sliding
export ROLLOUT_WINDOW=4

cd $WORLDMODEL && \
MODEL_PATH=$WORK/checkpoints/Qwen3VL_4B_Final_swift \
TRAIN_FILE=$WORK/data/manifests/vln_r2r_train_filtered_966.parquet \
VAL_FILE=$WORK/data/manifests/vln_r2r_val_unseen_1839.parquet \
TOTAL_STEPS=$TOTAL_STEPS SAVE_FREQ=$SAVE_FREQ \
TRAIN_BATCH_SIZE=4 ROLLOUT_N=2 NUM_WORKERS=4 \
ROLLOUT_WINDOW=4 ROLLOUT_SCHEDULER=sliding \
PPO_MINI_BATCH_SIZE=4 PPO_MICRO_BATCH_SIZE=1 LOG_PROB_MICRO=1 \
ROLLOUT_TP=4 GPU_MEM_UTIL=0.4 \
ADV_ESTIMATOR=grpo_trajectory \
EXPERIMENT=$EXP \
bash vln/reinforcement_learning/recipe/vln_navida/run_grpo.sh \
trainer.resume_mode=auto
" &
TRAIN_PID=$!

while kill -0 "$TRAIN_PID" 2>/dev/null; do
  CHECKPOINT_STEP=0
  if [[ -s "$TRACKER" ]]; then
    CHECKPOINT_STEP=$(tr -d '[:space:]' < "$TRACKER")
  fi

  if [[ "$CHECKPOINT_STEP" =~ ^[0-9]+$ ]] && (( CHECKPOINT_STEP >= TARGET_STEP )); then
    echo "Completed job segment at checkpoint step $CHECKPOINT_STEP"
    if (( CHECKPOINT_STEP < TOTAL_STEPS )); then
      echo "Auto-cancelling Slurm job $SLURM_JOB_ID"
      scancel "$SLURM_JOB_ID"
    fi
    break
  fi
  sleep 2
done

wait "$TRAIN_PID"
