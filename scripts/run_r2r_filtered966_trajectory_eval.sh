#!/bin/bash
# One-node ipp6 evaluation: R2R filtered 966, pass@8, one trajectory PNG/episode.
# Usage: bash ~/janec/run_r2r_filtered966_trajectory_eval.sh <JOBID> [RESULT_DIR]

set -euo pipefail
source /etc/profile 2>/dev/null || true

JOBID=${1:?"Usage: bash $0 <JOBID> [RESULT_DIR]"}
RESULT_DIR=${2:-r2r_filtered966_trajectory}
CKPT=Qwen3VL_4B_Final_swift
RAID=/raid/$JOBID
NFS=$HOME/janec
RESULT_PATH=$NFS/results/task16/$RESULT_DIR
MANIFEST=${MANIFEST:-$NFS/data/manifests/vln_r2r_train_filtered_966.parquet}
EVAL_SCRIPT=$RAID/WorldModel/vln/difficulty_trajectory_eval.py
MAX_EPISODE=${MAX_EPISODE:-0}
SPLIT_NUM=${SPLIT_NUM:-24}

PYTHON_URL=https://raw.githubusercontent.com/JaneChen525/VLN/codex/add-r2r-rl-difficulty-script/vln/difficulty_trajectory_eval.py
MANIFEST_URL=https://huggingface.co/datasets/Janechen525/vln_r2r_manifests/resolve/main/vln_r2r_train_filtered_966.parquet

echo "=== R2R filtered966 trajectory eval: job=$JOBID result=$RESULT_DIR ==="
cd "$RAID"

# Reuse the proven report/019 environment layout.
if [[ ! -x "$RAID/bin/python" ]]; then
  cp "$NFS/images/vln-conda.tar.gz" .
  tar xzf vln-conda.tar.gz
  rm vln-conda.tar.gz
fi
[[ -f "$RAID/vllm-26.04.sqsh" ]] || cp "$NFS/images/vllm-26.04.sqsh" .
[[ -d "$RAID/WorldModel" ]] || cp -r "$NFS/WorldModel_eval_ready" WorldModel
[[ -d "$RAID/$CKPT" ]] || cp -r "$NFS/checkpoints/$CKPT" .

mkdir -p "$RAID/data" "$RAID/cache_vllm" "$RAID/mesa_cache" \
  "$NFS/logs" "$RESULT_PATH"
ln -sfn "$NFS/data/scene_datasets" "$RAID/data/scene_datasets"
ln -sfn "$NFS/data/vln_eval_datasets" "$RAID/data/vln_eval_datasets"

# This is the only uploaded launcher; fetch its versioned evaluator + manifest.
curl -fsSL --retry 3 "$PYTHON_URL" -o "$EVAL_SCRIPT"
if [[ ! -f "$MANIFEST" ]]; then
  mkdir -p "$(dirname "$MANIFEST")"
  curl -fsSL --retry 3 "$MANIFEST_URL" -o "$MANIFEST"
fi
"$RAID/bin/python" -c 'import pyarrow' || {
  echo "pyarrow is required to read $MANIFEST"
  exit 1
}

CFG=$RAID/WorldModel/config/vln_r2r_trajectory.yaml
cp "$RAID/WorldModel/config/vln_r2r.yaml" "$CFG"
sed -i 's/hfov: 79/hfov: 90/g' "$CFG"
sed -i -E 's/(gpu_device_id:).*/\1 1/' "$CFG"
sed -i -E 's/(split:).*/\1 train/' "$CFG"
sed -i 's|max_episode_steps: 400|max_episode_steps: 200|' "$CFG"
sed -i "s|scenes_dir:.*|scenes_dir: $RAID/data/scene_datasets|" "$CFG"
sed -i "s|data_path:.*|data_path: $RAID/data/vln_eval_datasets/r2r/{split}/{split}.json.gz|" "$CFG"

echo "=== Config ==="
grep -n 'hfov\|gpu_device_id\|split:\|scenes_dir\|data_path' "$CFG" | grep -v '#'

VLLM_LOG=$NFS/logs/vllm_${RESULT_DIR}.log
if ! curl -fsS --max-time 3 http://localhost:8001/v1/models >/dev/null 2>&1; then
  setsid srun --jobid="$JOBID" --overlap \
    --container-image="$RAID/vllm-26.04.sqsh" \
    --container-mounts="$RAID:$RAID,$RAID/cache_vllm:/home/svcperf_ssg" \
    bash -c "CUDA_VISIBLE_DEVICES=0 python3 -m vllm.entrypoints.openai.api_server \
      --model '$RAID/$CKPT' --tensor-parallel-size 1 --port 8001 \
      --gpu-memory-utilization 0.6 --max-model-len 51200 \
      --max-num-seqs 32 --trust-remote-code" \
    > "$VLLM_LOG" 2>&1 &
fi

echo "=== Waiting for vLLM ==="
for i in $(seq 1 60); do
  if curl -fsS --max-time 3 http://localhost:8001/v1/models >/dev/null 2>&1; then
    echo "vLLM READY"
    break
  fi
  if [[ "$i" -eq 60 ]]; then
    tail -n 100 "$VLLM_LOG"
    exit 1
  fi
  sleep 10
done

cd "$RAID/WorldModel"
EVAL_LOG=$NFS/logs/difficulty_${RESULT_DIR}.log
nohup bash -c "
export __GL_SHADER_DISK_CACHE=0
export MESA_SHADER_CACHE_DIR=$RAID/mesa_cache
export XDG_CACHE_HOME=$RAID/mesa_cache
export OPENAI_API_KEY=EMPTY
export OPENAI_API_BASE=http://localhost:8001/v1
PYTHONPATH=. '$RAID/bin/python' -u vln/difficulty_trajectory_eval.py \
  --exp-config '$CFG' --episode-manifest '$MANIFEST' \
  --split-num '$SPLIT_NUM' --pass-k 8 --temperature 0.6 \
  --max-episode '$MAX_EPISODE' --max-action-history 200 \
  --forward-distance 25 --turn-angle 15 --num-generations 1 \
  --num-render-gpus 0 --num-shards 1 --shard-id 0 \
  --result-path '$RESULT_PATH' \
  --output-csv '$RESULT_PATH/episode_difficulty.csv'
" > "$EVAL_LOG" 2>&1 &

echo "=== Eval launched ==="
echo "Log:      tail -f $EVAL_LOG"
echo "Progress: wc -l $RESULT_PATH/result.json"
echo "Maps:     $RESULT_PATH/trajectory_maps/"
