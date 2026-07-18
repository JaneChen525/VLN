#!/bin/bash
#SBATCH --job-name=grpo-4step
#SBATCH --account=general_perflab
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --partition=polar,polar3,polar4
#SBATCH --time=04:00:00
#SBATCH --gpus=8
#SBATCH --output=/lustre/fsw/portfolios/general/users/jiaychen/logs/%x-%j.log
#SBATCH --error=/lustre/fsw/portfolios/general/users/jiaychen/logs/%x-%j.log

set -euo pipefail
umask 007

WORK=/lustre/fsw/portfolios/general/users/jiaychen
WORLDMODEL=$WORK/WorldModel
EXP=${EXP:-p13-env4-final-mrope-traj-grpo-32x8}
TOTAL_STEPS=${TOTAL_STEPS:-30}
STEPS_PER_JOB=${STEPS_PER_JOB:-4}
SAVE_FREQ=${SAVE_FREQ:-4}

CKPT_ROOT=$WORLDMODEL/checkpoints/vln-grpo/$EXP
TRACKER=$CKPT_ROOT/latest_checkpointed_iteration.txt
ENV_LOG=$WORK/logs/env-server-${SLURM_JOB_ID}.log

mkdir -p "$WORK/logs" "$WORLDMODEL/wandb"

sync_wandb_only() {
  srun --overlap \
    --container-image=verlai/verl:vllm011.latest \
    --container-mounts=$WORK:$WORK \
    bash -c "
export HOME=$WORK/.wandb-home
export WANDB_DIR=$WORLDMODEL/wandb
bash $WORLDMODEL/scripts/sync_wandb_run.sh \"\$WANDB_DIR\" $EXP
"
}

CURRENT_STEP=0
if [[ -s "$TRACKER" ]]; then
  CURRENT_STEP=$(tr -d '[:space:]' < "$TRACKER")
fi
if ! [[ "$CURRENT_STEP" =~ ^[0-9]+$ ]]; then
  echo "Invalid checkpoint tracker: $CURRENT_STEP"
  exit 1
fi
if (( CURRENT_STEP >= TOTAL_STEPS )); then
  echo "Training already complete: step=$CURRENT_STEP"
  sync_wandb_only
  exit 0
fi

TARGET_STEP=$((CURRENT_STEP + STEPS_PER_JOB))
if (( TARGET_STEP > TOTAL_STEPS )); then
  TARGET_STEP=$TOTAL_STEPS
fi
echo "Job segment: step $CURRENT_STEP -> $TARGET_STEP"

# P13/P14 use a constant LR. Ending this invocation at TARGET_STEP lets verl
# save, close W&B, and exit normally without changing the effective LR schedule.
cd "$WORLDMODEL"
PYTHONPATH="$WORLDMODEL/vln/reinforcement_learning:$WORLDMODEL:$WORLDMODEL/vln:${PYTHONPATH:-}" \
  "$WORK/miniconda3/envs/vln/bin/python" \
  -m recipe.vln_navida.env_server.launch \
  --exp-config config/vln_r2r.yaml \
  --port 8002 --pool-size 32 \
  --gpu-ids 0,1,2,3,4,5,6,7 --split train \
  > "$ENV_LOG" 2>&1 &
ENV_PID=$!

cleanup() {
  kill "$ENV_PID" 2>/dev/null || true
  wait "$ENV_PID" 2>/dev/null || true
}
trap cleanup EXIT

READY=0
for _ in $(seq 1 180); do
  if ! kill -0 "$ENV_PID" 2>/dev/null; then
    tail -n 100 "$ENV_LOG"
    exit 1
  fi
  STATS=$(curl -fsS http://127.0.0.1:8002/stats 2>/dev/null || true)
  if [[ "$STATS" == *'"free_workers":32'* ]]; then
    READY=1
    echo "env_server ready: $STATS"
    break
  fi
  sleep 5
done
if (( READY == 0 )); then
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
export VLLM_MM_INPUT_CACHE_GIB=32
export ROLLOUT_SCHEDULER=sliding
export ROLLOUT_WINDOW=32

TRAIN_RC=0
cd $WORLDMODEL && \
MODEL_PATH=$WORK/checkpoints/Qwen3VL_4B_Final_swift \
TRAIN_FILE=$WORK/data/manifests/vln_r2r_train_filtered_966.parquet \
VAL_FILE=$WORK/data/manifests/vln_r2r_val_unseen_1839.parquet \
TOTAL_STEPS=$TARGET_STEP SAVE_FREQ=$SAVE_FREQ \
TRAIN_BATCH_SIZE=32 ROLLOUT_N=8 NUM_WORKERS=32 \
ROLLOUT_WINDOW=32 ROLLOUT_SCHEDULER=sliding \
PPO_MINI_BATCH_SIZE=32 PPO_MICRO_BATCH_SIZE=4 LOG_PROB_MICRO=4 \
ROLLOUT_TP=4 GPU_MEM_UTIL=0.5 \
ADV_ESTIMATOR=grpo_trajectory \
EXPERIMENT=$EXP \
bash vln/reinforcement_learning/recipe/vln_navida/run_grpo.sh \
trainer.resume_mode=auto || TRAIN_RC=\$?

SYNC_RC=0
bash scripts/sync_wandb_run.sh \"\$WANDB_DIR\" \"\$WANDB_RUN_ID\" || SYNC_RC=\$?
if (( TRAIN_RC != 0 )); then
  exit \$TRAIN_RC
fi
exit \$SYNC_RC
"

CHECKPOINT_STEP=$(tr -d '[:space:]' < "$TRACKER")
if ! [[ "$CHECKPOINT_STEP" =~ ^[0-9]+$ ]] || (( CHECKPOINT_STEP < TARGET_STEP )); then
  echo "Segment ended without target checkpoint: expected=$TARGET_STEP actual=$CHECKPOINT_STEP"
  exit 1
fi
echo "Completed job segment at checkpoint step $CHECKPOINT_STEP"
