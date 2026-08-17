# Qwen3.5 + Verl 裸机 Conda 环境

本目录只负责在一台裸机 Linux 服务器上创建项目的 Qwen3.5/Verl 训练
Conda 环境，不依赖 Docker 或 Slurm，也不安装、启动或修改 Habitat。
同事已有的 Habitat 环境和 smoke workflow 保持独立。

## 已验证范围

该配方已在 8×A100 上完成两次 Qwen3.5 真实 one-step RL 验证：一次使用
原始环境，一次从空 Conda prefix 重建。随后又在同一个重建 prefix 上完成
Qwen3-VL one-step。三次均走通 FSDP2 forward/backward、optimizer 调用、
vLLM rollout 与权重同步，并输出 `training/global_step:1`。

固定的软件栈如下：

| 组件 | 版本 |
|---|---|
| Python | 3.12.13 |
| PyTorch | 2.10.0 + cu129 |
| torchvision / torchaudio | 0.25.0 / 2.10.0 |
| vLLM | 0.18.0 |
| Transformers | 5.3.0 |
| FlashAttention | 2.8.3 |
| causal-conv1d | 1.6.2.post1 |
| fla-core / flash-linear-attention | 0.4.2 |

`requirements-portable.txt` 固定其余 Python 依赖。构建脚本还包含已验证的
Transformers 5.3 Qwen3.5 3-D M-RoPE/FlashAttention 修复。

## Qwen3-VL 与 Qwen3.5 的区别

本配方是一个同时兼容 Qwen3-VL 与 Qwen3.5 的运行栈超集，但新增构建工作
主要来自 Qwen3.5：其 hybrid backbone 含 GDN linear-attention 层，需要
causal-conv1d/FLA；Transformers 5.3 还必须修复 3-D M-RoPE 被误判为 packed
sequence 的 FlashAttention 越界问题。Qwen3.5 的已验证 RL 路径另外显式使用
FSDP2、关闭 actor/ref torch.compile，并关闭 remove padding。

这些都是运行栈适配，不改变项目自己的 RL 算法配置。完整差异表和
必要 Hydra 覆盖见
[`QWEN3_VS_QWEN35.md`](QWEN3_VS_QWEN35.md)。

## 1. 裸机要求

- x86_64 Ubuntu 24.04；
- 8×H200，MIG 关闭；
- NVIDIA Driver 570 或更新版本；
- 建议至少 256 GiB RAM、64 GiB `/dev/shm`、100 GiB 可用磁盘；
- 构建阶段需要访问 conda-forge、PyPI、PyTorch cu129 index、Hugging Face
  和 GitHub。

安装宿主依赖：

```bash
sudo apt-get update
sudo apt-get install -y \
  linux-headers-$(uname -r) build-essential ca-certificates curl wget \
  git git-lfs bzip2 unzip xz-utils pkg-config cmake ninja-build \
  numactl libnuma1 libnuma-dev rdma-core libibverbs1 libibverbs-dev \
  librdmacm1 librdmacm-dev pciutils procps lsof iproute2 tmux
git lfs install
```

### Driver 570

本环境使用 CUDA 12.9 wheel。Driver 570 低于 CUDA 12.9 Update 1 的完整
驱动基线，因此需要安装并启用 forward-compat 用户态库：

```bash
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt-get update
sudo apt-get install -y cuda-compat-12-9
export LD_LIBRARY_PATH=/usr/local/cuda-12.9/compat${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
```

如果是 HGX/NVSwitch H200，必须安装与驱动分支匹配的 Fabric Manager：

```bash
sudo apt-get install -y cuda-drivers-fabricmanager-570
sudo systemctl enable --now nvidia-fabricmanager
nvidia-smi -q -d FABRIC
nvidia-smi topo -m
```

上面的包名只适用于 R570；若使用 R575/R580，Fabric Manager 包也必须替换为
相同驱动分支。

PCIe-only H200 不需要 Fabric Manager；运行预检时将
`REQUIRE_FABRIC_MANAGER=0`。

## 2. 克隆项目与 Verl

仓库当前将 Verl 配置为 `update = none`，所以必须显式覆盖该选项：

```bash
git clone --branch codex/qwen35-conda-handoff --single-branch \
  https://github.com/JaneChen525/VLN.git
cd VLN
git -c submodule.vln/reinforcement_learning.update=checkout \
  submodule update --init --recursive
```

## 3. 准备 Conda

机器上需要已有 Miniconda 或 Miniforge。这里只要求一个可写的 Conda root，
不要提前手工创建训练环境。

如果尚未安装 Conda，可以安装到用户目录：

```bash
MINICONDA=Miniconda3-py312_26.5.3-2-Linux-x86_64.sh
curl -fsSLo "/tmp/$MINICONDA" \
  "https://repo.anaconda.com/miniconda/$MINICONDA"
echo '37606f9f03ced8ef60f4ffc76b21dda01728eac8a632dcab316c891cea4fe2f5  /tmp/'"$MINICONDA" \
  | sha256sum --check
bash "/tmp/$MINICONDA" -b -p "$HOME/miniconda3"
```

```bash
HANDOFF=$PWD/environment/qwen35_verl_conda
cp "$HANDOFF/scripts/baremetal.env.example" "$HOME/qwen35-baremetal.env"
```

编辑 `$HOME/qwen35-baremetal.env` 中的路径，然后加载：

```bash
source "$HOME/qwen35-baremetal.env"
mkdir -p "$WORK"
DRIVER=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -n1)
if dpkg --compare-versions "$DRIVER" lt 575.57.08; then
  export LD_LIBRARY_PATH=/usr/local/cuda-12.9/compat${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
fi
```

`ENV_PREFIX` 是最终训练环境目录，例如
`/data/qwen35-verl-work/conda-envs/qwen35-verl-vllm018`；它不需要位于 Git
仓库内部，也不要在机器之间直接复制。

## 4. 宿主预检

HGX/NVSwitch H200：

```bash
EXPECTED_GPUS=8 REQUIRE_H200=1 REQUIRE_FABRIC_MANAGER=1 \
  TARGET_DIR="$WORK" \
  bash "$HANDOFF/scripts/h200_host_preflight.sh"
```

PCIe-only H200 将 `REQUIRE_FABRIC_MANAGER=0`。预检必须以
`HOST_PREFLIGHT_SUMMARY failures=0` 结束。

## 5. 创建环境

建议在 `tmux` 中执行。构建 FlashAttention 时实测峰值内存约 69 GiB，
通常需要 40–70 分钟；不需要系统 CUDA toolkit，`nvcc` 会安装到 Conda
环境中。

```bash
tmux new -s qwen35-env
```

进入新建的 tmux 会话后执行：

```bash
set -o pipefail
source "$HOME/qwen35-baremetal.env"
DRIVER=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -n1)
if dpkg --compare-versions "$DRIVER" lt 575.57.08; then
  export LD_LIBRARY_PATH=/usr/local/cuda-12.9/compat${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
fi
bash "$HANDOFF/scripts/baremetal_build.sh" 2>&1 | tee "$WORK/qwen35-conda-build.log"
```

构建默认只暴露 GPU 0 做 kernel 检查，但生成的 CUDA 扩展同时包含 sm80
和 sm90。首次构建要求 `ENV_PREFIX` 不存在、`MANIFEST_DIR` 为空；中断后
确认目录内容可复用时，才设置 `ALLOW_RESUME=1`。

## 6. 验收环境

成功时最后会打印：

```text
BUILD_SUCCESS env=... model=... manifests=...
```

再执行以下检查：

```bash
"$ENV_PREFIX/bin/python" "$HANDOFF/scripts/check_stack.py" \
  --model "$MODEL_DIR" --expect-model-type qwen3_5 --run-kernel

cat "$MANIFEST_DIR/pip-check.txt"
test ! -s "$MANIFEST_DIR/pip-check-unexpected.txt"
```

`check_stack.py` 必须输出 `STACK_CHECK_OK`。`pip check` 允许且仅允许下面
两条来自参考 Docker 环境的 metadata 冲突，构建脚本已经验证没有其他
冲突：

- vLLM 0.18.0 声明 `transformers<5`，项目实际固定 Transformers 5.3.0；
- vLLM 0.18.0 声明 OpenCV ≥4.13，项目为兼容 NumPy 1.26 固定 4.11.0.86。

完整包清单和扩展 wheel 会写入 `$MANIFEST_DIR`。同事之后可直接把
`ENV_PREFIX` 和现有 Habitat 环境路径接入自己的 smoke workflow。

## 文件说明

```text
QWEN3_VS_QWEN35.md                  # 两个 backbone 的运行配置差异
scripts/
├── baremetal_build.sh              # 裸机唯一入口
├── baremetal.env.example           # 路径配置模板
├── build_env.sh                    # Conda/pip/CUDA 扩展构建
├── check_stack.py                  # CUDA、FA2、GDN、模型加载检查
├── h200_host_preflight.sh          # Ubuntu/H200/Driver/Fabric/系统库预检
├── patch_transformers_qwen35_fa.py # Transformers 5.3 Qwen3.5 修复
├── requirements-portable.txt       # 精确 Python 依赖版本
├── qwen35-model-revision.txt       # 固定 Hugging Face revision
└── qwen35-model-files.sha256       # 模型文件完整性
```

本次交付不包含 Habitat、RL smoke、Slurm、模型权重、数据集、Conda prefix
或预编译 wheel。同一 Conda prefix 已分别完成 Qwen3.5 和 Qwen3-VL
one-step RL 路径验证；科学实验参数仍由同事的 workflow 显式固定。
