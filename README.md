# VLN

VLN（Vision-and-Language Navigation）RL 训练框架。基于 Qwen3-VL + LoRA，在 Habitat 仿真环境中用 GRPO 做强化学习。

## 分支 & 版本

| 分支/Tag | 说明 |
|----------|------|
| `vlnrlV3.1.0` | RL 框架开发主线（最新） |
| tag `VLNrlV3.1.0` | 稳定版（P10 验证通过，全指标 +1pp vs SFT baseline） |
| `main` | 上游 baseline（DAgger + SFT + eval） |

## 快速开始

详细复现指南见 [028-teammate-reproduction-guide.md](028-teammate-reproduction-guide.md)。

```bash
git clone https://github.com/JaneChen525/VLN.git WorldModel && cd WorldModel
git checkout VLNrlV3.1.0
git submodule update --init vln/reinforcement_learning
cd vln/reinforcement_learning && git checkout VLNrlV3.1.0 && cd ../..
```

### 依赖

- **Host**：Conda Python 3.10 + habitat-sim/lab v0.3.3（渲染）
- **Docker**：`verlai/verl:vllm011.latest`（训练，含 vllm 0.11 + ray + torch 2.8）
- **数据**：MP3D 场景 + R2R episodes + SFT checkpoint + manifest parquet

### 训练（GRPO）

```bash
# 1. Host: 启动 env_server
PYTHONPATH=vln/reinforcement_learning:.:vln python \
  -m recipe.vln_navida.env_server.launch \
  --exp-config config/vln_r2r.yaml --port 8002 --pool-size 32 \
  --gpu-ids 0,1,2,3,4,5,6,7 --split train

# 2. 启动 Docker 容器
docker run -d --name verl-dev \
  --gpus all --network host --ipc host --privileged --shm-size 32g \
  --entrypoint sleep \
  -v <你的工作目录>:/workspace \
  verlai/verl:vllm011.latest infinity

# 3. Docker: 快速测试 8*2（smoke test，~30min）
nohup docker exec -e VLLM_MM_INPUT_CACHE_GIB=8 verl-dev bash -c '
cd /workspace/WorldModel && \
TRAIN_FILE=/workspace/WorldModel/data/manifests/vln_r2r_train_filtered_966.parquet \
VAL_FILE=/workspace/WorldModel/data/manifests/vln_r2r_val_unseen_1839.parquet \
TOTAL_STEPS=3 SAVE_FREQ=999 \
TRAIN_BATCH_SIZE=8 ROLLOUT_N=2 NUM_WORKERS=16 \
PPO_MINI_BATCH_SIZE=8 ROLLOUT_TP=4 GPU_MEM_UTIL=0.5 \
ROLLOUT_WINDOW=16 \
EXPERIMENT=phase1-smoke-3step \
bash vln/reinforcement_learning/recipe/vln_navida/run_grpo.sh' > /tmp/grpo_phase1.log 2>&1 &

# 4. Docker: 正式训练 Window 模式（P10 验证通过）
nohup docker exec -e VLLM_MM_INPUT_CACHE_GIB=8 verl-dev bash -c '
cd /workspace/WorldModel && \
TRAIN_FILE=/workspace/WorldModel/data/manifests/vln_r2r_train_filtered_966.parquet \
VAL_FILE=/workspace/WorldModel/data/manifests/vln_r2r_val_unseen_1839.parquet \
TOTAL_STEPS=30 SAVE_FREQ=10 \
TRAIN_BATCH_SIZE=32 ROLLOUT_N=4 NUM_WORKERS=32 \
PPO_MINI_BATCH_SIZE=32 PPO_MICRO_BATCH_SIZE=4 LOG_PROB_MICRO=4 \
ROLLOUT_TP=4 GPU_MEM_UTIL=0.5 \
ROLLOUT_WINDOW=32 \
EXPERIMENT=p10-window \
bash vln/reinforcement_learning/recipe/vln_navida/run_grpo.sh' > /tmp/grpo_window.log 2>&1 &

# 4. Docker: 正式训练 Sliding 模式（待测试，消除 window barrier bubble）
nohup docker exec -e VLLM_MM_INPUT_CACHE_GIB=8 verl-dev bash -c '
cd /workspace/WorldModel && \
TRAIN_FILE=/workspace/WorldModel/data/manifests/vln_r2r_train_filtered_966.parquet \
VAL_FILE=/workspace/WorldModel/data/manifests/vln_r2r_val_unseen_1839.parquet \
TOTAL_STEPS=30 SAVE_FREQ=10 \
TRAIN_BATCH_SIZE=32 ROLLOUT_N=4 NUM_WORKERS=32 \
PPO_MINI_BATCH_SIZE=32 PPO_MICRO_BATCH_SIZE=4 LOG_PROB_MICRO=4 \
ROLLOUT_TP=4 GPU_MEM_UTIL=0.5 \
ROLLOUT_WINDOW=32 \
ROLLOUT_SCHEDULER=sliding \
EXPERIMENT=p10-sliding \
bash vln/reinforcement_learning/recipe/vln_navida/run_grpo.sh' > /tmp/grpo_sliding.log 2>&1 &
```

### 评测

```bash
# vLLM 服务
bash scripts/vllm_qwenvln.sh

# NaVIDA pass@k 评测
python vln/eval_vllm_navida.py \
  --exp-config config/vln_r2r.yaml --split-num 16 --pass-k 4 \
  --result-path ./results/eval_output
```

## 项目结构

```
VLN/
├── vln/
│   ├── eval_vllm_navida.py          # NaVIDA pass@k 评测
│   ├── dagger.py                    # DAgger 数据收集
│   └── reinforcement_learning/      # verl submodule（RL 框架）
│       └── recipe/vln_navida/       # VLN GRPO recipe
│           ├── run_grpo.sh          # 训练入口
│           ├── online_rollout_manager.py  # rollout + flatten
│           ├── full_episode_agent_loop.py # episode 循环
│           ├── env_server/          # Habitat 渲染服务
│           └── reward.py            # reward 函数
├── config/vln_r2r.yaml              # Habitat + 评测配置
├── tool/convert.py                  # 轨迹 → ShareGPT
└── scripts/                         # vLLM / eval 启动脚本
```

## 文档

| 文档 | 内容 |
|------|------|
| [复现指南](028-teammate-reproduction-guide.md) | 通用机器复现指南（从零开始） |

