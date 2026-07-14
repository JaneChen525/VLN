#!/bin/bash
#SBATCH --job-name=vln-std-eval
#SBATCH --account=general_perflab
#SBATCH --partition=backfill_singlenode
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=64
#SBATCH --time=12:00:00
#SBATCH --requeue
#SBATCH --output=/lustre/fsw/portfolios/general/users/jiaychen/logs/%x-%j.log
#SBATCH --error=/lustre/fsw/portfolios/general/users/jiaychen/logs/%x-%j.log

set -euo pipefail
umask 007

WORK=${WORK:-/lustre/fsw/portfolios/general/users/jiaychen}
WORLDMODEL=${WORLDMODEL:-$WORK/WorldModel}
MODEL_PATH=${MODEL_PATH:-$WORK/checkpoints/Qwen3VL_4B_Final_swift}
RESULT_NAME=${RESULT_NAME:-final-swift-val-unseen-pass4}
RESULT_DIR=${RESULT_DIR:-$WORK/results/$RESULT_NAME}
PASS_K=${PASS_K:-4}
SPLIT_NUM=${SPLIT_NUM:-24}
EXPECTED_EPISODES=${EXPECTED_EPISODES:-1839}
EXPECTED_TRIALS=$((EXPECTED_EPISODES * PASS_K))
PORT=${PORT:-$((18000 + ${SLURM_JOB_ID:-0} % 10000))}
PYTHON=${PYTHON:-$WORK/miniconda3/envs/vln/bin/python}
IMAGE=${IMAGE:-verlai/verl:vllm011.latest}
CONFIG=$RESULT_DIR/vln_r2r_env4_eval.yaml
VLLM_LOG=$WORK/logs/${RESULT_NAME}-vllm-${SLURM_JOB_ID:-local}.log
METRICS=$RESULT_DIR/metrics.txt
DONE=$RESULT_DIR/EVAL_COMPLETE

echo "MODEL_PATH=$MODEL_PATH"
echo "RESULT_DIR=$RESULT_DIR"
echo "PORT=$PORT PASS_K=$PASS_K SPLIT_NUM=$SPLIT_NUM"

if [[ ${DRY_RUN:-0} == 1 ]]; then
  exit 0
fi

mkdir -p "$WORK/logs" "$RESULT_DIR"
if [[ -f "$DONE" && -f "$METRICS" ]]; then
  echo "Already complete: $RESULT_DIR"
  exit 0
fi

test -x "$PYTHON"
test -f "$MODEL_PATH/config.json"
test -f "$WORK/data/vln_eval_datasets/r2r/val_unseen/val_unseen.json.gz"
test -d "$WORK/data/scene_datasets"

cp "$WORLDMODEL/config/vln_r2r.yaml" "$CONFIG"
sed -i -E 's/(hfov:).*/\1 90/' "$CONFIG"
sed -i -E 's/(gpu_device_id:).*/\1 0/' "$CONFIG"
sed -i -E 's/(split:).*/\1 val_unseen/' "$CONFIG"
sed -i -E "s|(scenes_dir:).*|\1 $WORK/data/scene_datasets|" "$CONFIG"
sed -i -E "s|(data_path:).*|\1 $WORK/data/vln_eval_datasets/r2r/{split}/{split}.json.gz|" "$CONFIG"

VLLM_PID=""
cleanup() {
  if [[ -n "$VLLM_PID" ]]; then
    kill "$VLLM_PID" 2>/dev/null || true
    wait "$VLLM_PID" 2>/dev/null || true
  fi
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

srun --overlap \
  --container-image="$IMAGE" \
  --container-mounts="$WORK:$WORK" \
  bash -lc "CUDA_VISIBLE_DEVICES=0 python3 -m vllm.entrypoints.openai.api_server \
    --model '$MODEL_PATH' --served-model-name qwen3vl \
    --tensor-parallel-size 1 --max-model-len 51200 --max-num-seqs 32 \
    --gpu-memory-utilization 0.7 --trust-remote-code \
    --host 0.0.0.0 --port '$PORT'" \
  > "$VLLM_LOG" 2>&1 &
VLLM_PID=$!

for i in $(seq 1 180); do
  if curl -fsS --max-time 3 "http://localhost:$PORT/v1/models" >/dev/null; then
    echo "vLLM ready after $((i * 5))s"
    break
  fi
  if ! kill -0 "$VLLM_PID" 2>/dev/null; then
    tail -n 100 "$VLLM_LOG"
    exit 1
  fi
  if (( i == 180 )); then
    tail -n 100 "$VLLM_LOG"
    exit 1
  fi
  sleep 5
done

cd "$WORLDMODEL"
OPENAI_API_BASE="http://localhost:$PORT/v1" \
OPENAI_API_KEY=EMPTY \
__GL_SHADER_DISK_CACHE=0 \
PYTHONPATH=. \
  "$PYTHON" -u vln/eval_vllm_navida.py \
    --exp-config "$CONFIG" --split-num "$SPLIT_NUM" \
    --forward-distance 25 --turn-angle 15 --max-action-history 200 \
    --num-generations 1 --pass-k "$PASS_K" --result-path "$RESULT_DIR"

"$PYTHON" data/task10/compute_metrics.py "$RESULT_DIR" | tee "$METRICS"
grep -Eq "n_episodes +: $EXPECTED_EPISODES$" "$METRICS"
grep -Eq "n_complete_pass_k +: $EXPECTED_EPISODES$" "$METRICS"
grep -Eq "n_trials_total +: $EXPECTED_TRIALS$" "$METRICS"
touch "$DONE"
echo "Standard eval complete: $RESULT_DIR"
