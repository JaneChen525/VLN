#!/bin/bash
# R2R RL 难度评测修正版：关键推理/导航参数对齐 run_rxr_difficulty.sh
# 用法: bash ~/janec/run_r2r_rl_difficulty_fix.sh <JOBID> <SHARD_ID> <NUM_SHARDS>
# 示例: bash ~/janec/run_r2r_rl_difficulty_fix.sh 290500 0 1
# 数据: Lelouchrx/r2r_rl（4594 episodes, 已合并到 ~/janec/data/r2r_rl_merged/train.json.gz）

source /etc/profile 2>/dev/null
JOBID=$1
SHARD_ID=$2
NUM_SHARDS=${3:-1}
CKPT=Qwen3VL_4B_Final_swift
RESULT_DIR=r2r_rl_diff_fix_shard${SHARD_ID}

if [ -z "$JOBID" ] || [ -z "$SHARD_ID" ]; then
  echo "Usage: bash $0 <JOBID> <SHARD_ID> [NUM_SHARDS]"
  exit 1
fi

echo "=== Setup: JOBID=$JOBID SHARD=$SHARD_ID/$NUM_SHARDS ==="
cd /raid/$JOBID

# 环境
if [ ! -f /raid/$JOBID/bin/python ]; then
  cp ~/janec/images/vln-conda.tar.gz . && tar xzf vln-conda.tar.gz && rm vln-conda.tar.gz
fi
if [ ! -f /raid/$JOBID/vllm-26.04.sqsh ]; then
  cp ~/janec/images/vllm-26.04.sqsh .
fi
if [ ! -d /raid/$JOBID/WorldModel ]; then
  cp -r ~/janec/WorldModel_eval_ready WorldModel
fi
if [ ! -d /raid/$JOBID/$CKPT ]; then
  cp -r ~/janec/checkpoints/$CKPT .
fi
mkdir -p data
ln -sf ~/janec/data/scene_datasets data/scene_datasets
ln -sf ~/janec/data/r2r_rl_merged data/r2r_rl_merged

# config: 从 vln_r2r.yaml 创建 r2r_rl 版
cp WorldModel/config/vln_r2r.yaml WorldModel/config/vln_r2r_rl.yaml
sed -i "s|scenes_dir:.*|scenes_dir: /raid/$JOBID/data/scene_datasets|" WorldModel/config/vln_r2r_rl.yaml
sed -i "s|data_path:.*|data_path: /raid/$JOBID/data/r2r_rl_merged/{split}.json.gz|" WorldModel/config/vln_r2r_rl.yaml
sed -i 's|split: val_unseen|split: train|' WorldModel/config/vln_r2r_rl.yaml
sed -i 's|# - /habitat/simulator/sensor_setups|- /habitat/simulator/sensor_setups|' WorldModel/config/vln_r2r_rl.yaml
sed -i 's|  - /habitat/simulator/agents@|  # - /habitat/simulator/agents@|' WorldModel/config/vln_r2r_rl.yaml
sed -i 's|max_episode_steps: 400|max_episode_steps: 200|' WorldModel/config/vln_r2r_rl.yaml
sed -i 's|gpu_device_id: 0|gpu_device_id: 1|' WorldModel/config/vln_r2r_rl.yaml

# vLLM（TP=1, GPU 0；参数对齐 run_rxr_difficulty.sh）
mkdir -p /raid/$JOBID/cache_vllm
setsid srun --jobid=$JOBID --overlap \
  --container-image=/raid/$JOBID/vllm-26.04.sqsh \
  --container-mounts=/raid/$JOBID:/raid/$JOBID,/raid/$JOBID/cache_vllm:/home/svcperf_ssg \
  bash -c "CUDA_VISIBLE_DEVICES=0 python3 -m vllm.entrypoints.openai.api_server \
    --model /raid/$JOBID/$CKPT \
    --tensor-parallel-size 1 --port 8001 \
    --gpu-memory-utilization 0.6 --max-model-len 51200 \
    --max-num-seqs 32 \
    --trust-remote-code" \
  > ~/janec/logs/vllm_${RESULT_DIR}.log 2>&1 &

echo "=== Waiting for vLLM ==="
for i in $(seq 1 30); do
  curl -s --max-time 3 http://localhost:8001/v1/models >/dev/null 2>&1 && echo 'vLLM READY' && break
  echo "waiting... ($i/30)"; sleep 10
done

# eval
mkdir -p ~/janec/results/task16/${RESULT_DIR}
cd /raid/$JOBID/WorldModel
nohup bash -c "
export __GL_SHADER_DISK_CACHE=0
OPENAI_API_KEY=EMPTY OPENAI_API_BASE=http://localhost:8001/v1 \
  /raid/$JOBID/bin/python -u vln/difficulty_eval.py \
    --exp-config config/vln_r2r_rl.yaml --split-num 32 \
    --pass-k 8 --temperature 0.6 \
    --max-action-history 200 \
    --forward-distance 25 --turn-angle 15 \
    --shard-id $SHARD_ID --num-shards $NUM_SHARDS \
    --num-render-gpus 0 \
    --result-path ~/janec/results/task16/${RESULT_DIR} \
    --output-csv ~/janec/results/task16/${RESULT_DIR}/episode_difficulty.csv
" > ~/janec/logs/difficulty_${RESULT_DIR}.log 2>&1 &

echo "=== Eval launched: shard $SHARD_ID/$NUM_SHARDS ==="
echo "Monitor: tail -f ~/janec/logs/difficulty_${RESULT_DIR}.log"
echo "Progress: wc -l ~/janec/results/task16/${RESULT_DIR}/result.json"
