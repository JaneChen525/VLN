# Qwen3-VL 与 Qwen3.5 运行配置差异

本文只说明本项目中两条 RL 路径的实际差异。Qwen3 指已有的
`Qwen3VL_4B_Final_swift`，Qwen3.5 指 `Qwen35_4B_Final_swift`；二者都复用
同一个 Habitat env server、VLN recipe、数据格式和 rollout 接口。

## 一览

| 项目 | 原 Qwen3-VL 实验 | Qwen3.5 Conda 实验 | 性质 |
|---|---|---|---|
| 模型类型 | `qwen3_vl` | `qwen3_5` | backbone 差异 |
| 主干结构 | 标准视觉语言 Transformer | 24 层 GDN linear attention + 8 层 full attention（当前 4B 权重） | backbone 差异 |
| RL 运行环境 | Verl Docker：PyTorch 2.8 / CUDA 12.8 / vLLM 0.11 | 裸机 Conda：PyTorch 2.10+cu129 / Transformers 5.3 / vLLM 0.18 | 运行栈迁移 |
| GDN 依赖 | 不需要 | `causal-conv1d==1.6.2.post1`、`fla-core==0.4.2`、`flash-linear-attention==0.4.2` | Qwen3.5 必需 |
| Attention CUDA 扩展 | 使用原容器提供的实现 | 本机构建 FA2 2.8.3 和 causal-conv1d，并检查 sm80/sm90 cubin | Conda/跨 GPU 适配 |
| M-RoPE/FA2 修复 | 原容器未额外打补丁 | Transformers 5.3 增加 3-D `position_ids` guard | Qwen3.5 当前栈必需 |
| remove padding | 旧 launcher 未覆盖，服从原栈默认值 | `use_remove_padding=False` | 已验证 Qwen3.5 路径 |
| FSDP | 原 P17 使用原栈默认路径 | actor/ref 显式 `fsdp2` | 已验证 Qwen3.5 路径 |
| torch.compile | 服从原栈默认值 | actor/ref 顶层和 nested engine 均显式关闭 | 稳定性适配 |
| vLLM engine | 原 launcher 未显式指定 V1 | `VLLM_USE_V1=1` | vLLM 0.18 路径 |
| full-attention 实现 | 原容器默认 | 保持 `flash_attention_2`，不降级到 SDPA | 性能与正确性设置 |

Qwen3.5 的 GDN 依赖、Transformers guard 和本地 CUDA 扩展是新增的运行栈
适配。同一 Conda 环境也可以加载 Qwen3-VL，Qwen3-VL 不会调用
额外安装的 GDN kernel。

本仓库的环境构建默认下载并校验官方
`Qwen/Qwen3.5-4B` qualification model。P17 使用的
`honglyhly/Qwen35_4B_Final_swift` 和旧 Qwen3-VL 权重均不随环境仓库分发；
应由实验 workflow 独立下载、固定 revision/hash，再把模型路径传给 Verl。
换成同属 `qwen3_5` 的微调权重不需要重装运行栈，但不能拿官方模型的哈希文件
去校验 P17 权重。

## Qwen3.5 必要运行覆盖

把下面的片段合并到同事已有的 RL workflow；这不是一份独立 smoke 脚本：

```bash
export VLLM_USE_V1=1

bash vln/reinforcement_learning/recipe/vln_navida/run_grpo.sh \
  actor_rollout_ref.model.use_remove_padding=False \
  actor_rollout_ref.actor.strategy=fsdp2 \
  actor_rollout_ref.ref.strategy=fsdp2 \
  actor_rollout_ref.actor.use_torch_compile=False \
  actor_rollout_ref.ref.use_torch_compile=False \
  actor_rollout_ref.actor.fsdp_config.use_torch_compile=False \
  actor_rollout_ref.ref.fsdp_config.use_torch_compile=False
```

四个 `use_torch_compile=False` 不能只写顶层两个：当前 Verl fork 的实际
engine 读取 nested `fsdp_config`。FlashAttention 继续使用环境中固定的 2.8.3；
不要为了绕过 3-D M-RoPE 问题切成 SDPA，构建脚本已经应用并校验正确 guard。

环境验收命令：

```bash
# Qwen3.5
"$ENV_PREFIX/bin/python" "$HANDOFF/scripts/check_stack.py" \
  --model /path/to/Qwen35_4B_Final_swift \
  --expect-model-type qwen3_5 --run-kernel

# Qwen3-VL（同一个 Conda prefix）
"$ENV_PREFIX/bin/python" "$HANDOFF/scripts/check_stack.py" \
  --model /path/to/Qwen3VL_4B_Final_swift \
  --expect-model-type qwen3_vl --run-kernel
```

两条命令都应以 `STACK_CHECK_OK` 结束。这里验证的是依赖、模型加载和 CUDA
kernel 前反向；完整 RL 正确性仍由项目自己的 smoke workflow 验证。

## 为什么需要 Transformers guard

Transformers 5.3 的 `_is_packed_sequence` 会把 Qwen3.5 的三维 M-RoPE
`position_ids` 误判成 packed sequence，从而构造比实际 Q/K/V 更长的
`cu_seqlens`，最终在 FlashAttention forward/backward 中越界。错误通常延迟到
FSDP/NCCL reduce-scatter 才暴露为 `CUDA illegal memory access`。

`patch_transformers_qwen35_fa.py` 在模型 worker 启动前增加
`position_ids.dim() > 2` 的 guard；`check_stack.py` 会对该回归和 GDN、FA2、
causal-conv1d 的前反向同时做检查。
