#!/bin/bash
#SBATCH --job-name=r2r-diff-map
#SBATCH --account=general_perflab
#SBATCH --partition=backfill_singlenode
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=8
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
MANIFEST=${MANIFEST:-$WORK/data/manifests/vln_r2r_train_filtered_966.parquet}
RESULT_NAME=${RESULT_NAME:-final-swift-filtered966-difficulty-map}
RESULT_DIR=${RESULT_DIR:-$WORK/results/$RESULT_NAME}
MAX_EPISODE=${MAX_EPISODE:-0}
PASS_K=${PASS_K:-8}
SPLIT_NUM=${SPLIT_NUM:-24}
PORT=${PORT:-$((20000 + ${SLURM_JOB_ID:-0} % 10000))}
PYTHON=${PYTHON:-$WORK/miniconda3/envs/vln/bin/python}
IMAGE=${IMAGE:-verlai/verl:vllm011.latest}
CONFIG=$RESULT_DIR/vln_r2r_env4_difficulty.yaml
VLLM_LOG=$WORK/logs/${RESULT_NAME}-vllm-${SLURM_JOB_ID:-local}.log
DONE=$RESULT_DIR/EVAL_COMPLETE

echo "MODEL_PATH=$MODEL_PATH"
echo "MANIFEST=$MANIFEST"
echo "RESULT_DIR=$RESULT_DIR"
echo "MAX_EPISODE=$MAX_EPISODE PASS_K=$PASS_K PORT=$PORT"

if [[ ${DRY_RUN:-0} == 1 ]]; then
  exit 0
fi

mkdir -p "$WORK/logs" "$RESULT_DIR"
if [[ -f "$DONE" ]]; then
  echo "Already complete: $RESULT_DIR"
  exit 0
fi

test -x "$PYTHON"
test -f "$MODEL_PATH/config.json"
test -f "$MANIFEST"
test -f "$WORK/data/vln_eval_datasets/r2r/train/train.json.gz"
test -d "$WORK/data/scene_datasets"

cp "$WORLDMODEL/config/vln_r2r.yaml" "$CONFIG"
sed -i -E 's/(hfov:).*/\1 90/' "$CONFIG"
sed -i -E 's/(gpu_device_id:).*/\1 1/' "$CONFIG"
sed -i -E 's/(split:).*/\1 train/' "$CONFIG"
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
OMP_NUM_THREADS=1 \
PYTHONPATH=. \
  "$PYTHON" -u vln/difficulty_eval.py \
    --exp-config "$CONFIG" --episode-manifest "$MANIFEST" \
    --split-num "$SPLIT_NUM" --pass-k "$PASS_K" --temperature 0.6 \
    --max-episode "$MAX_EPISODE" --max-action-history 200 \
    --forward-distance 25 --turn-angle 15 --num-generations 1 \
    --num-render-gpus 7 --render-gpu-offset 1 \
    --save-trajectory-map \
    --result-path "$RESULT_DIR" \
    --output-csv "$RESULT_DIR/episode_difficulty.csv"

EXPECTED_EPISODES=$(
  "$PYTHON" -c 'import pyarrow.parquet as pq,sys; print(pq.read_metadata(sys.argv[1]).num_rows)' "$MANIFEST"
)
if (( MAX_EPISODE > 0 && MAX_EPISODE < EXPECTED_EPISODES )); then
  EXPECTED_EPISODES=$MAX_EPISODE
fi
EXPECTED_TRIALS=$((EXPECTED_EPISODES * PASS_K))

"$PYTHON" - "$RESULT_DIR/result.json" "$RESULT_DIR/trajectory_maps" \
  "$EXPECTED_EPISODES" "$EXPECTED_TRIALS" <<'PY'
import json
import pathlib
import sys

result_file, map_dir, expected_episodes, expected_trials = sys.argv[1:]
expected_episodes, expected_trials = int(expected_episodes), int(expected_trials)
trials = {}
for line in open(result_file):
    try:
        item = json.loads(line)
    except json.JSONDecodeError:
        continue
    if all(key in item for key in ("scene_id", "episode_id", "trial_id")):
        key = (str(item["scene_id"]), str(item["episode_id"]), int(item["trial_id"]))
        trials[key] = item
episodes = {(key[0], key[1]) for key in trials}
maps = list(pathlib.Path(map_dir).glob("*_8rollouts.png"))
assert len(episodes) == expected_episodes, (len(episodes), expected_episodes)
assert len(trials) == expected_trials, (len(trials), expected_trials)
assert len(maps) == expected_episodes, (len(maps), expected_episodes)
print(f"COMPLETE episodes={len(episodes)} trials={len(trials)} maps={len(maps)}")
PY

touch "$DONE"
echo "Difficulty trajectory maps complete: $RESULT_DIR"
