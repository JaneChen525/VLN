#!/bin/bash
#SBATCH --job-name=q35-conda-rl-smoke
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
TRAIN_ENV=${TRAIN_ENV_OVERRIDE:-$WORK/conda-envs/qwen35-verl-vllm018}
HABITAT_PY=${HABITAT_PY_OVERRIDE:-$WORK/miniconda3/envs/vln/bin/python}
CONDA_EXE_PATH=${CONDA_EXE_OVERRIDE:-$WORK/miniconda3/bin/conda}
SCRIPTS_DIR=${QWEN35_SCRIPTS_DIR:-$WORLDMODEL/scripts/qwen35_conda}
MODEL_FAMILY=${MODEL_FAMILY:-qwen35}
MODEL=${MODEL_PATH_OVERRIDE:-$WORK/checkpoints/Qwen3.5-4B-base}
HABITAT_CONFIG=${HABITAT_CONFIG_OVERRIDE:-$SCRIPTS_DIR/vln_r2r_smoke.yaml}
HABITAT_SCENES_DIR=${HABITAT_SCENES_DIR_OVERRIDE:-$WORK/data/scene_datasets}
HABITAT_R2R_DATA_PATH=${HABITAT_R2R_DATA_PATH_OVERRIDE:-}
if [[ -z "$HABITAT_R2R_DATA_PATH" ]]; then
  HABITAT_R2R_DATA_PATH="$WORK/data/vln_eval_datasets/r2r/{split}/{split}.json.gz"
fi
HABITAT_PORT=${HABITAT_PORT:-8002}
HABITAT_POOL_SIZE=${HABITAT_POOL_SIZE:-2}
HABITAT_GPU_IDS=${HABITAT_GPU_IDS:-0,1}
TRAIN_FILE=${TRAIN_FILE_OVERRIDE:-$WORK/data/manifests/vln_r2r_train_filtered_966.parquet}
VAL_FILE=${VAL_FILE_OVERRIDE:-$WORK/data/manifests/vln_r2r_val_unseen_1839.parquet}
MANIFEST_DIR=${MANIFEST_DIR_OVERRIDE:-$WORLDMODEL/artifacts/qwen35-conda-env}
EVIDENCE_ROOT=${SMOKE_EVIDENCE_ROOT_OVERRIDE:-$WORLDMODEL/results/qwen-conda-smoke/evidence}
EXPERIMENT_PREFIX=${EXPERIMENT_PREFIX:-qwen35-4b-base-conda-smoke}
RUN_ID=${SMOKE_RUN_ID:-$SLURM_JOB_ID}
EXP=${EXPERIMENT_PREFIX}-${RUN_ID}
RESULT_DIR=$WORLDMODEL/results/qwen-conda-smoke/$MODEL_FAMILY/$RUN_ID
ENV_LOG=$RESULT_DIR/env-server.log
TRAIN_LOG=$RESULT_DIR/train.log

mkdir -p "$RESULT_DIR" "$WORK/tmp/qwen35-conda" "$WORLDMODEL/wandb"
test -x "$TRAIN_ENV/bin/python"
test -x "$HABITAT_PY"
test -s "$MODEL/config.json"

export PATH="$TRAIN_ENV/bin:$PATH"
export WORLDMODEL
export HABITAT_SCENES_DIR HABITAT_R2R_DATA_PATH
DRIVER_VERSION=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | \
  sed -n '1{s/[[:space:]]//g;p;}' || true)
if [[ -n "$DRIVER_VERSION" ]] && \
   dpkg --compare-versions "$DRIVER_VERSION" lt 575.57.08 && \
   [[ -d /usr/local/cuda-12.9/compat ]]; then
  case ":${LD_LIBRARY_PATH:-}:" in
    *:/usr/local/cuda-12.9/compat:*) ;;
    *) export LD_LIBRARY_PATH="/usr/local/cuda-12.9/compat${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" ;;
  esac
fi
export PYTHONPATH="$WORLDMODEL/vln/reinforcement_learning:$WORLDMODEL:$WORLDMODEL/vln:${PYTHONPATH:-}"
export TMPDIR=$WORK/tmp/qwen35-conda
export RAY_TMPDIR=${RAY_TMPDIR_OVERRIDE:-/tmp/q35-ray-$SLURM_JOB_ID-$MODEL_FAMILY}
export HF_HOME=$WORK/.cache/huggingface-qwen35
export WANDB_MODE=disabled
export TOKENIZERS_PARALLELISM=false
export VLLM_MM_INPUT_CACHE_GIB=8
export VLLM_USE_V1=1
export RAY_DEDUP_LOGS=0
export NCCL_DEBUG=WARN

echo "timestamp=$(date -Is) host=$(hostname) job=$SLURM_JOB_ID run=$RUN_ID exp=$EXP"
nvidia-smi --query-gpu=name,driver_version,memory.total,compute_cap --format=csv,noheader
EXPECTED_MODEL_TYPE=qwen3_5
if [[ "$MODEL_FAMILY" == qwen3vl ]]; then
  EXPECTED_MODEL_TYPE=qwen3_vl
fi
python "$SCRIPTS_DIR/check_stack.py" \
  --model "$MODEL" --expect-model-type "$EXPECTED_MODEL_TYPE" --run-kernel

cd "$WORLDMODEL"
setsid env PYTHONPATH="$PYTHONPATH" "$HABITAT_PY" \
  -m recipe.vln_navida.env_server.launch \
  --exp-config "$HABITAT_CONFIG" \
  --port "$HABITAT_PORT" --pool-size "$HABITAT_POOL_SIZE" \
  --gpu-ids "$HABITAT_GPU_IDS" --split train \
  > "$ENV_LOG" 2>&1 &
ENV_PID=$!

cleanup() {
  kill -TERM -- "-$ENV_PID" 2>/dev/null || true
  sleep 2
  kill -KILL -- "-$ENV_PID" 2>/dev/null || true
  wait "$ENV_PID" 2>/dev/null || true
}
trap cleanup EXIT

READY=0
for _ in $(seq 1 180); do
  if ! kill -0 "$ENV_PID" 2>/dev/null; then
    tail -n 120 "$ENV_LOG"
    exit 1
  fi
  STATS=$(curl -fsS "http://127.0.0.1:$HABITAT_PORT/stats" 2>/dev/null || true)
  if [[ "$STATS" == *"\"free_workers\":$HABITAT_POOL_SIZE"* ]]; then
    echo "env_server ready: $STATS"
    READY=1
    break
  fi
  sleep 5
done
if (( READY == 0 )); then
  tail -n 120 "$ENV_LOG"
  exit 1
fi

set +e
MODEL_PATH="$MODEL" \
TRAIN_FILE="$TRAIN_FILE" VAL_FILE="$VAL_FILE" \
TOTAL_STEPS=1 SAVE_FREQ=-1 TEST_FREQ=-1 \
TRAIN_BATCH_SIZE=4 ROLLOUT_N=2 NUM_WORKERS=2 \
ROLLOUT_WINDOW=2 ROLLOUT_SCHEDULER=sliding \
PPO_MINI_BATCH_SIZE=4 PPO_MICRO_BATCH_SIZE=1 LOG_PROB_MICRO=1 \
ROLLOUT_TP=8 FSDP_SIZE=8 GPU_MEM_UTIL=0.45 \
EXPERIMENT="$EXP" PROJECT=qwen35-conda-smoke \
bash vln/reinforcement_learning/recipe/vln_navida/run_grpo.sh \
  actor_rollout_ref.model.use_remove_padding=False \
  actor_rollout_ref.actor.strategy=fsdp2 \
  actor_rollout_ref.ref.strategy=fsdp2 \
  actor_rollout_ref.actor.use_torch_compile=False \
  actor_rollout_ref.ref.use_torch_compile=False \
  actor_rollout_ref.actor.fsdp_config.use_torch_compile=False \
  actor_rollout_ref.ref.fsdp_config.use_torch_compile=False \
  trainer.logger='[console]' \
  trainer.val_before_train=False \
  trainer.resume_mode=disable \
  2>&1 | tee "$TRAIN_LOG"
TRAIN_RC=${PIPESTATUS[0]}
set -e

if (( TRAIN_RC != 0 )); then
  echo "RL smoke failed rc=$TRAIN_RC" >&2
  exit "$TRAIN_RC"
fi

if ! grep -Eq 'step:1 - .*training/global_step:1($| - )' "$TRAIN_LOG"; then
  echo 'Training command exited zero but no completed global step 1 was logged' >&2
  exit 1
fi
if ! grep -Eq 'step:1 - .*actor/grad_norm:' "$TRAIN_LOG"; then
  echo 'Global step was logged but no actor optimizer metric was found' >&2
  exit 1
fi
if grep -Eqi 'step:1 - .*actor/grad_norm:([-+]?inf|nan)($| - )' "$TRAIN_LOG"; then
  echo 'Actor gradient norm is non-finite; optimizer.step may have been skipped' >&2
  exit 1
fi

EVIDENCE_DIR="$EVIDENCE_ROOT/$MODEL_FAMILY/$RUN_ID"
mkdir -p "$EVIDENCE_DIR"
grep -E 'step:1 - .*training/global_step:1($| - )' "$TRAIN_LOG" \
  > "$EVIDENCE_DIR/step1-metrics.txt"
tail -n 300 "$TRAIN_LOG" > "$EVIDENCE_DIR/train-tail.log"
tail -n 120 "$ENV_LOG" > "$EVIDENCE_DIR/env-server-tail.log"
python "$SCRIPTS_DIR/check_stack.py" \
  --model "$MODEL" --expect-model-type "$EXPECTED_MODEL_TYPE" --run-kernel \
  > "$EVIDENCE_DIR/stack-check.txt" 2>&1
python -m pip list --format=freeze > "$EVIDENCE_DIR/pip-list.txt"
R2R_TRAIN_PATH=${HABITAT_R2R_DATA_PATH//\{split\}/train}
{
  echo "train_parquet=$TRAIN_FILE"
  echo "val_parquet=$VAL_FILE"
  echo "habitat_config=$HABITAT_CONFIG"
  echo "habitat_scenes_dir=$HABITAT_SCENES_DIR"
  echo "habitat_r2r_train=$R2R_TRAIN_PATH"
  echo "model=$MODEL"
  for input_file in "$TRAIN_FILE" "$VAL_FILE" "$HABITAT_CONFIG" "$R2R_TRAIN_PATH"; do
    if [[ -f "$input_file" ]]; then
      sha256sum "$input_file"
    else
      echo "MISSING_FILE $input_file"
    fi
  done
} > "$EVIDENCE_DIR/input-provenance.txt"

{
  echo "timestamp=$(date -Is)"
  echo "job_id=$SLURM_JOB_ID"
  echo "run_id=$RUN_ID"
  echo "model=$MODEL"
  echo "env=$TRAIN_ENV"
  echo "verl_commit=$(git -C "$WORLDMODEL/vln/reinforcement_learning" rev-parse HEAD)"
  echo "model_family=$MODEL_FAMILY"
  echo "worldmodel_commit=$(git -C "$WORLDMODEL" rev-parse HEAD)"
  echo "artifact_manifest_sha256=$(sha256sum "$MANIFEST_DIR/SHA256SUMS" | awk '{print $1}')"
  nvidia-smi --query-gpu=name,driver_version,compute_cap --format=csv,noheader
  echo 'status=RL_STEP_1_SUCCESS'
} > "$EVIDENCE_DIR/SMOKE_SUCCESS"
cp "$EVIDENCE_DIR/SMOKE_SUCCESS" "$RESULT_DIR/SMOKE_SUCCESS"

echo "RL_STEP_1_SUCCESS result=$RESULT_DIR"
