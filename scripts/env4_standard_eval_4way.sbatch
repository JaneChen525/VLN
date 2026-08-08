#!/bin/bash
#SBATCH --job-name=vln-std-eval-4way
#SBATCH --account=general_perflab
#SBATCH --partition=polar,polar3,polar4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=8
#SBATCH --cpus-per-task=64
#SBATCH --time=4:00:00
#SBATCH --signal=B:TERM@180
#SBATCH --output=/lustre/fsw/portfolios/general/users/jiaychen/logs/%x-%j.log
#SBATCH --error=/lustre/fsw/portfolios/general/users/jiaychen/logs/%x-%j.log

set -euo pipefail
umask 007

WORK=${WORK:-/lustre/fsw/portfolios/general/users/jiaychen}
WORLDMODEL=${WORLDMODEL:-$WORK/WorldModel}
MODEL_PATH=${MODEL_PATH:-$WORK/checkpoints/Qwen3VL_4B_Final_swift}
RESULT_NAME=${RESULT_NAME:-final-swift-val-unseen-pass4-fast4}
RESULT_ROOT=${RESULT_ROOT:-$WORK/results/$RESULT_NAME}
SEED_RESULT=${SEED_RESULT:-$WORK/results/final-swift-val-unseen-pass4/result.json}
PASS_K=${PASS_K:-4}
NUM_SHARDS=4
WORKERS_PER_SHARD=${WORKERS_PER_SHARD:-12}
EXPECTED_EPISODES=${EXPECTED_EPISODES:-1839}
EXPECTED_TRIALS=$((EXPECTED_EPISODES * PASS_K))
BASE_PORT=${BASE_PORT:-$((20000 + ${SLURM_JOB_ID:-0} % 20000))}
PYTHON=${PYTHON:-$WORK/miniconda3/envs/vln/bin/python}
IMAGE=${IMAGE:-verlai/verl:vllm011.latest}
VLLM_GPU_IDS=${VLLM_GPU_IDS:-0,1,2,3}
RENDER_GPU_IDS=${RENDER_GPU_IDS:-4,5,6,7}
VLLM_MAX_NUM_SEQS=${VLLM_MAX_NUM_SEQS:-16}
VLLM_GPU_MEM_UTIL=${VLLM_GPU_MEM_UTIL:-0.7}
VLLM_MM_CACHE_GB=${VLLM_MM_CACHE_GB:-8}
VLLM_MEDIA_THREADS=${VLLM_MEDIA_THREADS:-4}
SEED_SNAPSHOT=$RESULT_ROOT/seed_result.json
MERGED_RESULT=$RESULT_ROOT/result.json
METRICS=$RESULT_ROOT/metrics.txt
DONE=$RESULT_ROOT/EVAL_COMPLETE

IFS=, read -r -a VLLM_GPUS <<< "$VLLM_GPU_IDS"
IFS=, read -r -a RENDER_GPUS <<< "$RENDER_GPU_IDS"
if (( ${#VLLM_GPUS[@]} != NUM_SHARDS || ${#RENDER_GPUS[@]} != NUM_SHARDS )); then
  echo "Expected four vLLM and four render GPUs" >&2
  exit 2
fi
for gpu in "${VLLM_GPUS[@]}"; do
  if [[ ",$RENDER_GPU_IDS," == *",$gpu,"* ]]; then
    echo "GPU $gpu is assigned to both vLLM and rendering" >&2
    exit 2
  fi
done
if [[ "$PASS_K" != 4 ]]; then
  echo "Standard evaluation requires PASS_K=4" >&2
  exit 2
fi

echo "MODEL_PATH=$MODEL_PATH"
echo "RESULT_ROOT=$RESULT_ROOT"
echo "SEED_RESULT=$SEED_RESULT"
echo "Standard protocol: val_unseen pass@4 T=0.3 top_p=0.95 max_tokens=512 max_steps=400"
echo "Four endpoints: vLLM GPUs=$VLLM_GPU_IDS render GPUs=$RENDER_GPU_IDS workers/shard=$WORKERS_PER_SHARD"
echo "vLLM: TP=1 max_num_seqs=$VLLM_MAX_NUM_SEQS mm_cache_gb=$VLLM_MM_CACHE_GB media_threads=$VLLM_MEDIA_THREADS"

if [[ ${DRY_RUN:-0} == 1 ]]; then
  exit 0
fi

mkdir -p "$WORK/logs" "$RESULT_ROOT"
exec 9> "$RESULT_ROOT/.eval.lock"
if ! flock -n 9; then
  echo "Another evaluation job is using $RESULT_ROOT" >&2
  exit 3
fi
if [[ -f "$DONE" && -f "$METRICS" ]]; then
  echo "Already complete: $RESULT_ROOT"
  exit 0
fi

test -x "$PYTHON"
test -f "$MODEL_PATH/config.json"
compgen -G "$MODEL_PATH/*.safetensors" >/dev/null
test -f "$WORK/data/vln_eval_datasets/r2r/val_unseen/val_unseen.json.gz"
test -d "$WORK/data/scene_datasets"
test -f "$WORLDMODEL/vln/eval_vllm_navida.py"
test -f "$WORLDMODEL/scripts/merge_eval_results.py"
command -v flock >/dev/null
command -v setsid >/dev/null

VISIBLE_GPU_COUNT=$(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l)
if (( VISIBLE_GPU_COUNT != 8 )); then
  echo "Expected 8 visible GPUs, found $VISIBLE_GPU_COUNT; resubmit on a healthy node" >&2
  exit 4
fi

grep -Fq 'self.temperature = 0.3' "$WORLDMODEL/vln/eval_vllm_navida.py"
grep -Fq 'self.top_p = 0.95' "$WORLDMODEL/vln/eval_vllm_navida.py"
grep -Fq 'self.max_tokens = 512' "$WORLDMODEL/vln/eval_vllm_navida.py"
grep -Fq 'EARLY_STOP_STEPS = 400' "$WORLDMODEL/vln/eval_vllm_navida.py"

"$PYTHON" - "$MODEL_PATH" <<'PY'
import json
import pathlib
import sys

model = pathlib.Path(sys.argv[1])
config = json.loads((model / "config.json").read_text())
tokenizer = json.loads((model / "tokenizer_config.json").read_text())
rope = config["text_config"]["rope_scaling"]
assert rope["mrope_section"] == [24, 20, 20], rope
assert isinstance(tokenizer.get("extra_special_tokens", {}), dict)
print("MODEL_METADATA_OK", rope)
PY

# Freeze the old single-server result exactly once. Continuation jobs reuse it.
if [[ ! -f "$SEED_SNAPSHOT" ]]; then
  seed_tmp=$SEED_SNAPSHOT.tmp.${SLURM_JOB_ID:-$$}
  if [[ -n "$SEED_RESULT" && -f "$SEED_RESULT" ]]; then
    cp "$SEED_RESULT" "$seed_tmp"
  else
    : > "$seed_tmp"
  fi
  mv "$seed_tmp" "$SEED_SNAPSHOT"
fi
"$PYTHON" - "$SEED_SNAPSHOT" <<'PY'
import json
import sys

keys = set()
for line in open(sys.argv[1]):
    try:
        item = json.loads(line)
        keys.add((item["scene_id"], str(item["episode_id"]), item["episode_instruction"],
                  int(item["trial_id"]), int(item["trial_total"])))
    except (KeyError, TypeError, ValueError, json.JSONDecodeError):
        pass
print(f"Frozen seed trials: {len(keys)}")
PY

declare -a CONFIGS VLLM_LOGS EVAL_LOGS PORTS VLLM_PIDS EVAL_PIDS
for shard in $(seq 0 $((NUM_SHARDS - 1))); do
  shard_dir=$RESULT_ROOT/shard$shard
  mkdir -p "$shard_dir"
  CONFIGS[$shard]=$shard_dir/vln_r2r_env4_eval.yaml
  VLLM_LOGS[$shard]=$WORK/logs/${RESULT_NAME}-vllm-shard${shard}-${SLURM_JOB_ID:-local}.log
  EVAL_LOGS[$shard]=$WORK/logs/${RESULT_NAME}-eval-shard${shard}-${SLURM_JOB_ID:-local}.log
  PORTS[$shard]=$((BASE_PORT + shard))

  cp "$WORLDMODEL/config/vln_r2r.yaml" "${CONFIGS[$shard]}"
  sed -i -E 's/(hfov:).*/\1 90/' "${CONFIGS[$shard]}"
  sed -i -E "s/(gpu_device_id:).*/\1 ${RENDER_GPUS[$shard]}/" "${CONFIGS[$shard]}"
  sed -i -E 's/(split:).*/\1 val_unseen/' "${CONFIGS[$shard]}"
  sed -i -E "s|(scenes_dir:).*|\1 $WORK/data/scene_datasets|" "${CONFIGS[$shard]}"
  sed -i -E "s|(data_path:).*|\1 $WORK/data/vln_eval_datasets/r2r/{split}/{split}.json.gz|" "${CONFIGS[$shard]}"

  test "$(grep -Ec '^[[:space:]]+hfov:[[:space:]]+90$' "${CONFIGS[$shard]}")" -eq 2
  test "$(grep -Ec '^[[:space:]]+success_distance:[[:space:]]+3\.0$' "${CONFIGS[$shard]}")" -eq 2
  grep -Eq '^[[:space:]]+max_episode_steps:[[:space:]]+400$' "${CONFIGS[$shard]}"
  grep -Eq '^[[:space:]]+forward_step_size:[[:space:]]+0\.25$' "${CONFIGS[$shard]}"
  grep -Eq '^[[:space:]]+turn_angle:[[:space:]]+15$' "${CONFIGS[$shard]}"

  # A time-limit signal can interrupt the final JSONL append. Drop only that
  # incomplete tail before workers append during the next continuation.
  "$PYTHON" - "$shard_dir/result.json" <<'PY'
import os
import sys

path = sys.argv[1]
if os.path.isfile(path):
    with open(path, "rb+") as file:
        data = file.read()
        if data and not data.endswith(b"\n"):
            last_newline = data.rfind(b"\n")
            file.seek(0)
            file.truncate(last_newline + 1)
PY
done
echo "EVAL_CONFIG_OK: four configs, hfov=90, success_distance=3.0, max_steps=400"

cleanup() {
  set +e
  for pid in "${EVAL_PIDS[@]:-}"; do
    [[ -n "$pid" ]] && kill -TERM -- "-$pid" 2>/dev/null
  done
  for pid in "${VLLM_PIDS[@]:-}"; do
    [[ -n "$pid" ]] && kill -TERM "$pid" 2>/dev/null
  done
  sleep 5
  for pid in "${EVAL_PIDS[@]:-}"; do
    [[ -n "$pid" ]] && kill -KILL -- "-$pid" 2>/dev/null
  done
  for pid in "${VLLM_PIDS[@]:-}"; do
    [[ -n "$pid" ]] && kill -KILL "$pid" 2>/dev/null
  done
  wait 2>/dev/null
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

for shard in $(seq 0 $((NUM_SHARDS - 1))); do
  srun --overlap --ntasks=1 --cpus-per-task=4 \
    --container-image="$IMAGE" \
    --container-mounts="$WORK:$WORK" \
    bash -lc "export CUDA_VISIBLE_DEVICES='${VLLM_GPUS[$shard]}'
export VLLM_MEDIA_LOADING_THREAD_COUNT='$VLLM_MEDIA_THREADS'
python3 -m vllm.entrypoints.openai.api_server \
  --model '$MODEL_PATH' --served-model-name qwen3vl \
  --tensor-parallel-size 1 --max-model-len 51200 \
  --max-num-seqs '$VLLM_MAX_NUM_SEQS' \
  --gpu-memory-utilization '$VLLM_GPU_MEM_UTIL' \
  --mm-processor-cache-gb '$VLLM_MM_CACHE_GB' --trust-remote-code \
  --host 0.0.0.0 --port '${PORTS[$shard]}'" \
    > "${VLLM_LOGS[$shard]}" 2>&1 &
  VLLM_PIDS[$shard]=$!
done

for attempt in $(seq 1 180); do
  ready=0
  for shard in $(seq 0 $((NUM_SHARDS - 1))); do
    if ! kill -0 "${VLLM_PIDS[$shard]}" 2>/dev/null; then
      echo "vLLM shard $shard exited before readiness" >&2
      tail -n 100 "${VLLM_LOGS[$shard]}"
      exit 1
    elif curl -fsS --max-time 3 "http://localhost:${PORTS[$shard]}/v1/models" \
      | grep -Fq '"id":"qwen3vl"'; then
      ready=$((ready + 1))
    fi
  done
  if (( ready == NUM_SHARDS )); then
    echo "All four vLLM endpoints ready after $((attempt * 5))s"
    break
  fi
  if (( attempt == 180 )); then
    echo "Timed out waiting for vLLM endpoints ($ready/$NUM_SHARDS ready)" >&2
    exit 1
  fi
  sleep 5
done

cd "$WORLDMODEL"
for shard in $(seq 0 $((NUM_SHARDS - 1))); do
  OPENAI_API_BASE="http://localhost:${PORTS[$shard]}/v1" \
  OPENAI_API_KEY=EMPTY \
  __GL_SHADER_DISK_CACHE=0 \
  PYTHONPATH=. \
    setsid "$PYTHON" -u vln/eval_vllm_navida.py \
      --exp-config "${CONFIGS[$shard]}" --split-num "$WORKERS_PER_SHARD" \
      --forward-distance 25 --turn-angle 15 --max-action-history 200 \
      --num-generations 1 --pass-k "$PASS_K" \
      --shard-id "$shard" --num-shards "$NUM_SHARDS" \
      --skip-result-file "$SEED_SNAPSHOT" \
      --result-path "$RESULT_ROOT/shard$shard" \
      > "${EVAL_LOGS[$shard]}" 2>&1 &
  EVAL_PIDS[$shard]=$!
  EVAL_DONE[$shard]=0
  echo "Started evaluator shard $shard: endpoint=${PORTS[$shard]} render_gpu=${RENDER_GPUS[$shard]} log=${EVAL_LOGS[$shard]}"
done

eval_status=0
remaining=$NUM_SHARDS
while (( remaining > 0 && eval_status == 0 )); do
  for shard in $(seq 0 $((NUM_SHARDS - 1))); do
    if [[ ${EVAL_DONE[$shard]} == 1 ]]; then
      continue
    fi
    if ! kill -0 "${VLLM_PIDS[$shard]}" 2>/dev/null; then
      echo "vLLM shard $shard died during evaluation; see ${VLLM_LOGS[$shard]}" >&2
      eval_status=1
      break
    fi
    if ! kill -0 "${EVAL_PIDS[$shard]}" 2>/dev/null; then
      if wait "${EVAL_PIDS[$shard]}"; then
        EVAL_DONE[$shard]=1
        remaining=$((remaining - 1))
        echo "Evaluator shard $shard completed ($remaining remaining)"
      else
        echo "Evaluator shard $shard failed; see ${EVAL_LOGS[$shard]}" >&2
        eval_status=1
        break
      fi
    fi
  done
  if (( remaining > 0 && eval_status == 0 )); then
    sleep 5
  fi
done
if (( eval_status != 0 )); then
  exit 1
fi

"$PYTHON" scripts/merge_eval_results.py \
  --output "$MERGED_RESULT" \
  "$SEED_SNAPSHOT" \
  "$RESULT_ROOT"/shard{0,1,2,3}/result.json

"$PYTHON" data/task10/compute_metrics.py "$RESULT_ROOT" | tee "$METRICS"
grep -Eq "n_episodes +: $EXPECTED_EPISODES$" "$METRICS"
grep -Eq "n_complete_pass_k +: $EXPECTED_EPISODES$" "$METRICS"
grep -Eq "n_trials_total +: $EXPECTED_TRIALS$" "$METRICS"
touch "$DONE"
echo "Standard eval complete: $RESULT_ROOT"
