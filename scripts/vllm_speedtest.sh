#!/bin/bash
# Task 12 测速用 vLLM 服务（env1 A100）。
# 关键：必须开 prefix caching（setting 2 KV-cache 复用依赖它）。vLLM 0.16 V1 默认开启，
# 这里显式写出以防被改；--no-enable-prefix-caching 可做对照实验（setting 2 应退化成 setting 1）。
#
# 用法：
#   bash scripts/vllm_speedtest.sh              # 默认 hfov=90 merged, TP=1, GPU 0
#   GPU=4,5 TP=2 bash scripts/vllm_speedtest.sh # TP=2
set -e

MODEL="${MODEL:-/var/data0/sandbox/janec/WorldModel/checkpoints/Qwen3-VL-4B-vln-r2r-merged}"
GPU="${GPU:-0}"
TP="${TP:-1}"
PORT="${PORT:-8001}"
CONTEXT_LENGTH="${CONTEXT_LENGTH:-51200}"
GPU_MEM="${GPU_MEM:-0.85}"

export CUDA_VISIBLE_DEVICES="$GPU"

echo "[vllm_speedtest] model=$MODEL gpu=$GPU tp=$TP port=$PORT prefix_caching=ON"

VLLM_USE_MODELSCOPE=false vllm serve "$MODEL" \
    --served-model-name qwen3vl \
    --max-model-len "$CONTEXT_LENGTH" \
    --gpu-memory-utilization "$GPU_MEM" \
    --tensor-parallel-size "$TP" \
    --enable-prefix-caching \
    --trust-remote-code \
    --host 0.0.0.0 \
    --port "$PORT"
