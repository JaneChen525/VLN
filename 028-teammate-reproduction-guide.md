# VLN RL 训练复现指南（通用机器）

在自己机器上复现 VLN GRPO 训练的指南。

前提：x86_64 Linux, NVIDIA GPU (>=40GB VRAM x8), NVIDIA driver 570+, Docker + nvidia-container-toolkit 已安装。

## 快速概览

```
代码 (GitHub)  →  Conda env (habitat)  →  Docker (verl)  →  env_server + GRPO 训练
                  host 侧渲染            容器内训练
```

训练流程：host 上跑 Habitat env_server（渲染导航环境）→ Docker 容器内跑 verl GRPO（模型训练 + vLLM 推理），两者通过 HTTP localhost:8002 通信。

## 1. 代码

两个版本可选：

| 版本 | 分支/tag | 说明 |
|------|----------|------|
| **稳定版** | tag `VLNrlV3.1.0` | P10 验证通过，全指标 +1pp vs SFT baseline |
| **最新版** | branch `vlnrlV3.1.0` HEAD | 在稳定版基础上新增 sliding rollout queue（待测试） |

```bash
WORKDIR=<你的工作目录>   # 例如 /data/username
cd $WORKDIR

# 克隆 VLN（含 verl 子模块）
git clone https://github.com/JaneChen525/VLN.git WorldModel
cd WorldModel
git config submodule.vln/reinforcement_learning.update checkout

# ── 选其一 ──────────────────────────────────────────────────
# 稳定版（推荐先跑通）
git checkout VLNrlV3.1.0
git submodule update --init vln/reinforcement_learning
cd vln/reinforcement_learning && git checkout VLNrlV3.1.0 && cd ../..

# 最新版（测试 sliding queue）
git checkout vlnrlV3.1.0
git submodule update --init vln/reinforcement_learning
cd vln/reinforcement_learning && git checkout vlnrlV3.1.0 && cd ../..
# ──────────────────────────────────────────────────────────────
```

### 稳定版 vs 最新版差异

最新版在 `online_rollout_manager.py` 新增 **sliding rollout queue**，消除 fixed-window barrier bubble：

| | 稳定版（window 模式） | 最新版（sliding 模式） |
|---|---|---|
| 调度 | 固定窗口，等最慢 episode 结束才开下一窗口 | 滚动队列，episode 完成即启动下一个 |
| Bubble | 窗口内快慢不一致导致空闲等待（估 ~30%） | 无 barrier，理论零 bubble |
| 启用方式 | 默认行为 | `ROLLOUT_SCHEDULER=sliding` |
| 状态 | P10 验证通过 | **待测试**，需验证正确性 |

最新版**默认仍走 window 模式**，只有显式设 `ROLLOUT_SCHEDULER=sliding` 才走 sliding 路径。

### Sliding queue 测试方法

用稳定版跑通全链路后，切到最新版测试 sliding：

```bash
# 1. 对照组：window 模式（默认）
ROLLOUT_SCHEDULER=window TOTAL_STEPS=1 EXPERIMENT=test-window ...

# 2. 实验组：sliding 模式
ROLLOUT_SCHEDULER=sliding TOTAL_STEPS=1 EXPERIMENT=test-sliding ...
```

对比验证：
- `traj_count` 两组一致
- `decision` 数量接近（不会完全一样因为随机性）
- `vln/traj_sr` 正常范围
- `timing_s/gen` sliding 应更短（bubble 消除）
- 无 KeyError / crash

## 2. Conda 环境（host 侧，用于 Habitat）

```bash
cd $WORKDIR

# Miniconda
wget -q https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh -b -p $WORKDIR/miniconda3

# Python 3.10 环境
$WORKDIR/miniconda3/bin/conda create -p $WORKDIR/miniconda3/envs/vln python=3.10 -y
PIP=$WORKDIR/miniconda3/envs/vln/bin/pip3

# 构建工具
$WORKDIR/miniconda3/bin/conda install -p $WORKDIR/miniconda3/envs/vln \
  -c conda-forge cmake -y

# habitat-lab v0.3.3
cd $WORKDIR
git clone --branch stable https://github.com/facebookresearch/habitat-lab.git
cd habitat-lab && git checkout v0.3.3
$PIP install -e habitat-lab && $PIP install -e habitat-baselines
cd ..

# habitat-sim v0.3.3（源码编译，~15min）
git clone --branch stable https://github.com/facebookresearch/habitat-sim.git
cd habitat-sim && git checkout v0.3.3
HABITAT_BUILD_GUI_VIEWERS=OFF HABITAT_WITH_CUDA=ON HABITAT_WITH_BULLET=ON HABITAT_WITH_AUDIO=OFF \
  $PIP install . --no-build-isolation
cd ..

# 其他依赖
$PIP install fastdtw peft accelerate
```

### EGL / GPU 渲染验证

habitat-sim 需要 EGL 支持做 GPU 渲染。编译完后验证：

```bash
$WORKDIR/miniconda3/envs/vln/bin/python -c "
import habitat_sim
cfg = habitat_sim.SimulatorConfiguration()
cfg.scene_id = 'NONE'
cfg.gpu_device_id = 0
agent_cfg = habitat_sim.AgentConfiguration()
s = habitat_sim.CameraSensorSpec()
s.uuid, s.sensor_type, s.resolution = 'rgb', habitat_sim.SensorType.COLOR, [64,64]
agent_cfg.sensor_specifications = [s]
sim = habitat_sim.Simulator(habitat_sim.Configuration(cfg, [agent_cfg]))
print('GPU rendering OK')
sim.close()
"
```

如果报 `GL::Context: cannot retrieve OpenGL version`：
- 检查 `/usr/lib/x86_64-linux-gnu/libEGL_nvidia.so*` 是否存在
- 检查 `/usr/share/glvnd/egl_vendor.d/10_nvidia.json` 是否存在
- 可能需要 `apt-get install libegl1-mesa-dev libgl1-mesa-dev`
- 驱动升级后需重新编译 habitat-sim

## 3. 数据

| 数据 | 大小 | 来源 | 目标路径（相对 WorldModel） |
|------|------|------|---------------------------|
| SFT checkpoint | 8.3GB | HF `Lelouchrx/Qwen3-VL-4B_lora_vln_r2r_90` 或团队内传 | `checkpoints/Qwen3VL_4B_R2R_RxR_swift/` |
| MP3D 场景 | 21GB | HF `Lelouchrx/vln_scenes` | `data/scene_datasets/mp3d/` |
| R2R episodes | 620MB | HF `Lelouchrx/vln_eval_datasets` | `data/vln_eval_datasets/r2r/` |
| Manifest parquet | ~1MB | 团队内传 | `data/manifests/`（3 个 parquet） |

```bash
cd $WORKDIR/WorldModel

# HF 下载（需 huggingface-cli login）
huggingface-cli download Lelouchrx/vln_scenes --repo-type dataset --local-dir data/scene_datasets
huggingface-cli download Lelouchrx/vln_eval_datasets --repo-type dataset --local-dir data/vln_eval_datasets

# Manifest（从有数据的机器 scp）
# vln_r2r_train_10819.parquet       — 全量训练集
# vln_r2r_train_filtered_966.parquet — 难度过滤集（pass@8 pass_rate 0.2-0.8）
# vln_r2r_val_unseen_1839.parquet   — 验证集
```

## 4. Config 适配

`config/vln_r2r.yaml` 需改路径和 hfov：

```bash
cd $WORKDIR/WorldModel

# hfov 90（必须与 LoRA checkpoint 匹配）
sed -i 's|hfov: 79|hfov: 90|g' config/vln_r2r.yaml

# 场景路径
sed -i "s|scenes_dir:.*|scenes_dir: $WORKDIR/WorldModel/data/scene_datasets|" config/vln_r2r.yaml
sed -i "s|data_path:.*|data_path: $WORKDIR/WorldModel/data/vln_eval_datasets/r2r/{split}/{split}.json.gz|" config/vln_r2r.yaml
```

确认 yaml 中 `gpu_device_id: -1`（由 env_server `--gpu-ids` 参数控制实际 GPU 分配）。

## 5. Docker 容器（verl 训练）

```bash
docker pull verlai/verl:vllm011.latest

docker run -d --name verl-dev \
  --gpus all --network host --ipc host --privileged --shm-size 32g \
  --entrypoint sleep \
  -v $WORKDIR:/workspace \
  verlai/verl:vllm011.latest infinity

# wandb
docker exec -it verl-dev wandb login
```

容器内环境：vllm 0.11.0 + ray 2.49.2 + torch 2.8.0 + Python 3.12。verl 用 submodule 版本（不需要 pip install verl）。

验证：

```bash
docker exec verl-dev python3 -c 'import torch; print("GPUs:", torch.cuda.device_count())'
```

## 6. 运行训练

### 6.1 启动 Habitat env_server（host 侧）

```bash
cd $WORKDIR/WorldModel

pkill -9 -f 'envs/vln/bin/python' 2>/dev/null  # 清残留

PYTHONPATH=vln/reinforcement_learning:.:vln \
  nohup $WORKDIR/miniconda3/envs/vln/bin/python \
  -m recipe.vln_navida.env_server.launch \
  --exp-config config/vln_r2r.yaml --port 8002 --pool-size 32 \
  --gpu-ids 0,1,2,3,4,5,6,7 --split train \
  > /tmp/env_server.log 2>&1 &

# 等 32 worker 初始化（GPU 渲染 ~3min，CPU 渲染 ~8min）
tail -f /tmp/env_server.log
# 看到 "Application startup complete" 即可
```

验证：`curl -s http://localhost:8002/health` 应返回 `{"ok":true}`

### 6.2 GRPO 训练（容器内）

#### A. 稳定版 / Window 模式（默认，P10 验证通过）

```bash
docker restart verl-dev && sleep 6

# Smoke test（3 step 验证全链路，~30min）
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
```

Smoke 通过后，正式训练（P10 复现配置）：

```bash
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
```

#### B. 最新版 / Sliding 模式（待测试，需最新版代码）

和 Window 模式相比，唯一区别是加 `ROLLOUT_SCHEDULER=sliding`。

```bash
docker restart verl-dev && sleep 6

# Smoke test
nohup docker exec -e VLLM_MM_INPUT_CACHE_GIB=8 verl-dev bash -c '
cd /workspace/WorldModel && \
TRAIN_FILE=/workspace/WorldModel/data/manifests/vln_r2r_train_filtered_966.parquet \
VAL_FILE=/workspace/WorldModel/data/manifests/vln_r2r_val_unseen_1839.parquet \
TOTAL_STEPS=3 SAVE_FREQ=999 \
TRAIN_BATCH_SIZE=8 ROLLOUT_N=2 NUM_WORKERS=16 \
PPO_MINI_BATCH_SIZE=8 ROLLOUT_TP=4 GPU_MEM_UTIL=0.5 \
ROLLOUT_WINDOW=16 \
ROLLOUT_SCHEDULER=sliding \
EXPERIMENT=phase1-smoke-3step-sliding \
bash vln/reinforcement_learning/recipe/vln_navida/run_grpo.sh' > /tmp/grpo_sliding_smoke.log 2>&1 &
```

正式训练：

```bash
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

#### Sliding 测试验证清单

跑完 1 step smoke 后对比日志：

| 检查项 | Window 日志 | Sliding 日志 |
|--------|------------|-------------|
| 调度模式 | `[VLN rollout] window 1/16` | `[VLN rollout] sliding mode: 128 episodes, concurrency=8` |
| traj_count | ~128 | ~128（应一致） |
| decisions 数 | ~3400 | ~3400（接近，允许随机差异） |
| vln/traj_sr | 正常（~0.6-0.7） | 同上 |
| timing_s/gen | 基准值 | 应更短（bubble 消除） |
| crash / KeyError | 无 | **无**（重点验证） |

### 6.3 关键参数

| 变量 | 默认 | 说明 |
|------|------|------|
| `TRAIN_BATCH_SIZE` | 32 | 每 step 采样 episode 数 |
| `ROLLOUT_N` | 4 | GRPO group size（每 episode 跑 N 次） |
| `ROLLOUT_TP` | 8 | vLLM tensor parallel（8 GPU 全用=1 replica，4=2 replica 推理更快） |
| `GPU_MEM_UTIL` | 0.5 | vLLM GPU 显存占比（TP=4 时 0.5 够用） |
| `PPO_MICRO_BATCH_SIZE` | 1 | actor update micro batch（4 更快，需 per-GPU mini_batch >= 4） |
| `TOTAL_STEPS` | 339 | 训练总步数（966ep / 32 ≈ 30 步 = 1 epoch） |
| `SAVE_FREQ` | 10 | checkpoint 保存频率（每个 ~16GB） |
| `NUM_WORKERS` | 8 | 并发 rollout worker 数（<= env_server pool-size） |
| `VLN_ROLLOUT_WINDOW` | 8 | 最大并发 episode 数 |
| `ROLLOUT_SCHEDULER` | window | `window`=固定窗口（稳定），`sliding`=滚动队列（最新版，待测试） |
| `EXPERIMENT` | — | wandb run 名称 |

### 6.4 GPU 显存参考（8xH100 96GB / 8xA100 40GB）

| 配置 | H100 96GB | A100 40GB |
|------|-----------|-----------|
| TP=4, GPU_MEM_UTIL=0.5 | OK | OK |
| TP=8, GPU_MEM_UTIL=0.5 | OK | 偏紧 |
| batch=32, n=4 | ~244GB CPU RAM | ~244GB CPU RAM |

## 7. 监控

```bash
# 训练日志
tail -f /tmp/grpo.log

# 关键指标（每 step 输出）
grep -E '\[VLN rollout\]|critic/rewards|vln/traj_sr' /tmp/grpo.log

# GPU 利用率
nvidia-smi

# env_server 状态
curl -s http://localhost:8002/stats | python3 -m json.tool

# 磁盘
df -h $WORKDIR
```

## 8. Checkpoint 评测

训练产出 FSDP checkpoint，eval 前需合并为 HF 格式：

```bash
docker exec verl-dev python3 -m verl.model_merger merge \
  --backend fsdp \
  --local_dir /workspace/WorldModel/checkpoints/vln-grpo/<exp>/global_step_<N>/actor \
  --target_dir /workspace/WorldModel/checkpoints/merged_<name> \
  --trust-remote-code
```


## 9. 清理

```bash
pkill -9 -f 'envs/vln/bin/python'         # 杀 env_server
docker restart verl-dev                     # 清 Ray
rm -rf $WORKDIR/WorldModel/checkpoints/vln-grpo/*/  # 清 checkpoint
```

## 10. 常见问题

| 问题 | 原因 | 解决 |
|------|------|------|
| `GL::Context: cannot retrieve OpenGL version` | EGL 不兼容（驱动升级后） | 重编 habitat-sim 或不传 `--gpu-ids` 走 CPU 渲染 |
| step:1 后 checkpoint 52GB | FSDP 全量保存 | 正常，merge 后 ~8GB |
