# VLN

Vision-and-Language Navigation：在 Habitat 仿真环境中，给定自然语言指令，VLM 根据 RGB 第一视角图像输出离散动作完成导航。

**Tag `VLNVerlV3.0.0`** = verl GRPO RL 框架 + P5/P6 实验完成。

## 目录

- [环境安装](#环境安装)
- [数据准备](#数据准备)
- [SFT 训练](#sft-训练llamafactory)
- [RL 训练（GRPO）](#rl-训练grpo)
- [评测（NaVIDA pass@k）](#评测navida-passk)
- [已有实验结果](#已有实验结果)

---

## 环境安装

### 硬件要求

| 组件 | 最低配置 | 推荐配置 |
|------|---------|---------|
| GPU | 4×A100-40GB | 8×A100-40GB / 8×A100-80GB |
| RAM | 128GB | 256GB |
| 磁盘 | 200GB（模型+场景） | 500GB |

RL 训练需要 8 GPU：FSDP(8) + vLLM TP(8) colocated。

### Step 1: Clone（含 verl submodule）

```bash
git clone --recurse-submodules https://github.com/JaneChen525/VLN.git
cd VLN

# 若已 clone 但缺 submodule:
git submodule update --init --recursive
```

verl fork 在 `vln/reinforcement_learning/`（branch `vln-rl`），包含 VLN NaVIDA GRPO recipe。

### Step 2: Conda 环境（Habitat + eval）

```bash
conda create -n vln python=3.10 -y
conda activate vln
pip install -r requirements.txt
bash tool/habitat.sh          # 安装 habitat-sim 0.2.4 + habitat-lab 0.2.4
pip install vllm==0.16.0      # eval 用（非 RL 训练）
```

### Step 3: Docker 容器（RL 训练）

RL 训练在 verl 容器中运行，host 上的 conda `vln` 环境运行 Habitat env_server。

```bash
# 拉取 verl 官方镜像
docker pull verlai/verl:vllm011.latest

# 启动容器（挂载代码 + 数据）
docker run -d --gpus all --ipc=host --net=host \
  --name verl-dev \
  -v $(pwd):/workspace/VLN \
  -v /path/to/data:/root/data \
  -v /path/to/checkpoints:/workspace/VLN/checkpoints \
  verlai/verl:vllm011.latest sleep infinity
```

容器内环境：verl 0.8.0 + vLLM 0.11.0 + PyTorch 2.8.0 + CUDA 12.8。

### Step 4: 安装 verl VLN recipe（容器内）

```bash
docker exec -it verl-dev bash
cd /workspace/VLN
pip install -e vln/reinforcement_learning/
```

---

## 数据准备

### 场景数据

```bash
# MP3D 场景（需申请访问权限）
conda activate vln
python tool/download_mp.py

# 目录结构:
# data/scene_datasets/mp3d/{scene_id}/{scene_id}.glb
```

### VLN-CE Episodes

```bash
# R2R（必须）
# 下载: https://drive.google.com/file/d/1fo8F4NKgZDH-bPSdVU3cONAkt5EW-tyr/view
# 解压后重命名 R2R_VLNCE_v1-3_preprocessed/ → data/datasets/r2r/

# RxR（可选）
# 下载: https://drive.google.com/file/d/145xzLjxBaNTbVgBfQ8e9EsBAV8W-SM0t/view
# 解压后重命名 RxR_VLNCE_v0/ → data/datasets/rxr/
```

### 模型权重

```bash
# SFT checkpoint（RL 训练的起点）
pip install huggingface_hub
huggingface-cli download Lelouchrx/Qwen3VL_4B_R2R_RxR_swift --local-dir checkpoints/Qwen3VL_4B_R2R_RxR_swift

# 或 LoRA checkpoint（需额外 merge 步骤）
huggingface-cli download Lelouchrx/Qwen3-VL-4B_lora_vln_r2r_90 --local-dir checkpoints/lora_r2r_90
```

> swift checkpoint 的 `tokenizer_config.json` 中 `extra_special_tokens` 可能为 list 格式，vLLM 需要 dict 格式。修复：
> ```python
> import json
> p = "checkpoints/Qwen3VL_4B_R2R_RxR_swift/tokenizer_config.json"
> c = json.load(open(p))
> if isinstance(c.get("extra_special_tokens"), list):
>     c["extra_special_tokens"] = {}
>     json.dump(c, open(p, "w"), indent=2)
> ```

### 生成 RL episode manifest

```bash
conda activate vln
python vln/reinforcement_learning/recipe/vln_navida/make_manifest.py \
    --data-dir data/datasets/r2r \
    --splits train val_unseen \
    --out-dir /root/data
# 输出: vln_r2r_train_10819.parquet, vln_r2r_val_unseen_1839.parquet
```

### Hydra config 路径适配

`config/vln_r2r.yaml` 中的场景和数据路径需要改为本地路径：

```bash
sed -i "s|/mnt/share172/.*/scene_datasets|$(pwd)/data/scene_datasets|g" config/vln_r2r.yaml
sed -i "s|/mnt/share172/.*/r2rv1|$(pwd)/data/datasets/r2r|g" config/vln_r2r.yaml
```

确认 hfov=90（必须与 SFT 训练时一致，否则 SR=0%）：
```bash
grep 'hfov' config/vln_r2r.yaml
# 应输出: hfov: 90
```

---

## SFT 训练（LLaMA-Factory）

```bash
conda create -n llamafactory python=3.12 -y
conda activate llamafactory
pip install -e ".[torch,metrics]" --no-build-isolation
```

使用 `LLaMA-Factory/examples/train_lora/qwen3vl_lora_sft.yaml` 配置 LoRA SFT。训练数据由 DAgger 收集后经 `tool/convert.py` 转为 ShareGPT 格式。

---

## RL 训练（GRPO）

### 架构

```
┌─────────────────────────────────────────────────────────────┐
│  verl-dev 容器 (8×GPU colocated: FSDP训练 + vLLM推理)        │
│                                                             │
│  run_grpo.sh → verl main_ppo (GRPO)                         │
│    ├── VLNOnlineRolloutManager (rollout_window 并发控制)      │
│    │     └── VLNFullEpisodeAgentLoop × num_workers           │
│    │           ├── apply_chat_template → vLLM generate       │
│    │           └── VLNEnv (HTTP) ──→ env_server              │
│    ├── Actor FSDP update (GRPO loss)                         │
│    └── Ref model logprob                                     │
└─────────────────────────────────────────────────────────────┘
                            │ HTTP :8002
┌─────────────────────────────────────────────────────────────┐
│  Host (conda vln)                                           │
│  env_server (FastAPI)                                       │
│    └── WorkerPool: 8 × Habitat Env (每 GPU 1 worker)        │
│        R2R episodes → RGB渲染 → JPEG base64                  │
└─────────────────────────────────────────────────────────────┘
```

### Step 1: 启动 env_server（host 侧，conda vln）

```bash
conda activate vln
cd /path/to/VLN

PYTHONPATH=vln/reinforcement_learning:.:vln:$PYTHONPATH \
nohup python -m recipe.vln_navida.env_server.launch \
    --exp-config config/vln_r2r.yaml \
    --port 8002 \
    --pool-size 8 \
    --session-ttl-sec 120 \
    --gpu-ids 0,1,2,3,4,5,6,7 \
    --split train \
    > /tmp/env_server.log 2>&1 &

# 等待 "8 workers ready"
tail -f /tmp/env_server.log
```

### Step 2: 启动 GRPO 训练（容器内）

```bash
docker exec -it verl-dev bash
cd /workspace/VLN

# 设置 wandb（可选）
export WANDB_API_KEY=your_key

# 默认配置: batch=32, rollout_n=4, 30 steps
TOTAL_STEPS=30 EXPERIMENT=my-grpo-exp \
bash vln/reinforcement_learning/recipe/vln_navida/run_grpo.sh
```

自定义参数：

```bash
TRAIN_BATCH_SIZE=64 \
ROLLOUT_N=8 \
TEMPERATURE=0.6 \
TOTAL_STEPS=30 \
SAVE_FREQ=10 \
MODEL_PATH=/workspace/VLN/checkpoints/Qwen3VL_4B_R2R_RxR_swift \
TRAIN_FILE=/root/data/vln_r2r_train_10819.parquet \
EXPERIMENT=p7-filtered-966ep \
bash vln/reinforcement_learning/recipe/vln_navida/run_grpo.sh
```

### 关键超参

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `TRAIN_BATCH_SIZE` | 32 | 每 step 采样的 episode 数 |
| `ROLLOUT_N` | 4 | GRPO group size（每 episode rollout 次数） |
| `ROLLOUT_WINDOW` | 8 | 并发窗口（解决 vLLM 多模态缓存竞态） |
| `TEMPERATURE` | 0.6 | 采样温度 |
| `TOTAL_STEPS` | 339 | 训练步数（ceil(10819/32) = 1 epoch） |
| `ACTOR_LR` | 5e-7 | 学习率 |
| `KL_LOSS_COEF` | 0.001 | KL 正则化 |
| `GPU_MEM_UTIL` | 0.5 | vLLM 显存利用率（40GB 卡用 0.5，80GB 可提高） |

### Checkpoint 导出（eval 前必做）

FSDP checkpoint → HF safetensors：

```bash
docker exec verl-dev python3 -m verl.model_merger merge \
  --backend fsdp \
  --local_dir /workspace/VLN/checkpoints/vln-grpo/<exp>/global_step_<N>/actor \
  --target_dir /workspace/VLN/checkpoints/<merged_name> \
  --trust-remote-code
```

### Reward 设计

```python
reward = success + progress_coef * oracle_success
# success: 1.0（到达目标 3m 内）/ 0.0
# oracle_success: 轨迹中是否曾到达 3m 内
# progress_coef: 默认 0.1（config/agent_loop.yaml）
```

---

## 评测（NaVIDA pass@k）

### 方式 A：裸机（conda 环境，适合调试/小集）

```bash
conda activate vln

# 1) 启动 vLLM
CUDA_VISIBLE_DEVICES=0 vllm serve checkpoints/<model> \
  --served-model-name qwen3vl --max-model-len 51200 \
  --gpu-memory-utilization 0.7 --tensor-parallel-size 1 \
  --trust-remote-code --host 0.0.0.0 --port 8001 &

# 2) 等 vLLM 就绪
curl http://localhost:8001/v1/models

# 3) 跑 eval
export PYTHONPATH=$(pwd):$PYTHONPATH
export OPENAI_API_KEY=EMPTY OPENAI_API_BASE=http://localhost:8001/v1
export __GL_SHADER_DISK_CACHE=0

python -u vln/eval_vllm_navida.py \
  --exp-config config/vln_r2r.yaml --split-num 16 \
  --forward-distance 25 --turn-angle 15 --max-action-history 200 \
  --num-generations 1 --pass-k 4 --result-path results/<name>
```

### 方式 B：Habitat import 兼容

如果遇到 `from habitat_baselines.config.default import get_config` 报错，需 patch：

```bash
sed -i 's|from habitat_baselines.config.default import get_config|from habitat.config.default import get_config|' vln/eval_vllm_navida.py
```

### 输出格式

`result.json`（JSONL）每行一条 trial：

```json
{"scene_id": "x8F5xyUWy9e", "episode_id": 1348, "trial_id": 0, "trial_total": 4, "success": 1.0, "spl": 0.85, "os": 1.0, "ne": 1.20, "steps": 53, "episode_instruction": "..."}
```

最后一行为聚合指标：`sucs_all`（SR）, `spls_all`（SPL）, `pass@4`, `ones_all`（NE）。

---

## 已有实验结果

### Baseline（val_unseen, 1839ep, pass@4, hfov=90）

| 指标 | LoRA SFT | Swift SFT |
|------|---------|-----------|
| pass@1 (SR) | 39.79% | 56.48% |
| pass@4 | 61.28% | 76.40% |
| SPL | — | 69.57% |
| OSR | — | 79.66% |
| NE | 4.35m | 2.74m |

### P6: GRPO 30 step（过滤 966ep, batch=32, rollout_n=4）

| 指标 | P6 GRPO | Swift Baseline | Delta |
|------|---------|---------------|-------|
| pass@1 | 56.81% | 56.48% | +0.33pp |
| pass@4 | 76.45% | 76.40% | +0.05pp |
| SPL | 69.50% | 69.57% | -0.07pp |
| OSR | 80.21% | 79.66% | +0.55pp |
| NE | 2.82m | 2.74m | +0.08m |

> P6 delta < 1pp，当前配置未产生显著提升。下一步：增大 rollout_n=8 / batch=64、调整 reward 函数。

---

## 文件结构

```
VLN/
├── vln/
│   ├── eval_vllm_navida.py              # NaVIDA pass@k 评测（含 pass_k fix）
│   ├── dagger.py                        # DAgger 数据收集
│   ├── habitat_extensions/              # R2R/RxR 数据加载、指标
│   └── reinforcement_learning/          # verl fork (submodule, branch vln-rl)
│       └── recipe/vln_navida/
│           ├── run_grpo.sh              # RL 训练启动脚本
│           ├── online_rollout_manager.py
│           ├── verl_agent_loop.py
│           ├── full_episode_agent_loop.py
│           ├── reward.py
│           ├── dataset.py / make_manifest.py
│           ├── env_server/              # Habitat FastAPI 服务
│           └── config/agent_loop.yaml
├── tool/
│   ├── convert.py                       # 轨迹 → ShareGPT
│   └── habitat.sh                       # Habitat 安装
├── config/
│   └── vln_r2r.yaml                     # Hydra 配置（路径需本地化）
├── scripts/
│   ├── vllm_qwenvln.sh                  # vLLM 启动
│   └── eval_vllm.sh                     # 评测启动
└── LLaMA-Factory/                       # SFT 训练框架
```

## Troubleshooting

| 现象 | 原因 | 解决 |
|------|------|------|
| SR=0%, 模型从不 stop | hfov 不匹配 | 确认 `vln_r2r.yaml` 中 hfov=90 |
| `habitat_baselines` import 报错 | habitat-lab 版本差异 | `sed -i 's/habitat_baselines/habitat/'` |
| `extra_special_tokens` 报错 | swift tokenizer 格式不兼容 | list → dict（见数据准备节） |
| vLLM `mm_hash` 竞态 crash | 并发过高 | `ROLLOUT_WINDOW≤8` + `enable_prefix_caching=False` |
| advantage=0, pg_loss=0 | group 内 rollout 结果一致 | 增大 `ROLLOUT_N` / 提高 `TEMPERATURE` |
| OOM during vLLM | gpu_mem_util 过高 | 40GB 卡用 `GPU_MEM_UTIL=0.5`，80GB 可用 0.7 |
| 503 NO_FREE_WORKER | env_server session 泄漏 | 重启 env_server（TTL 自动回收） |

## DAgger 数据收集

```bash
conda activate vln
bash scripts/habitat.sh
bash scripts/dagger.sh
```

## 参考论文

- [NaVIDA](https://arxiv.org/abs/2601.18188) — 本项目主要参考
- [StreamVLN](https://arxiv.org/abs/2507.05240) — 基线对比
- [LongNav-R1](https://arxiv.org/abs/2602.12351) — RL 相关
- [WMPO](https://arxiv.org/abs/2511.09515) — VLA-RL 相关
