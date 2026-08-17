# `Dockerfile.stable.vllm` parity

Reference: `vln/reinforcement_learning/docker/Dockerfile.stable.vllm` at the
commit recorded in `verl-commit.txt`. Qualification target: this project's
Qwen3.5/Qwen3-VL dense-model RL path with FSDP2 actor/ref and vLLM rollout.

| Docker component | Conda/host treatment | Used by this RL path |
|---|---|---|
| Ubuntu 24.04, CUDA 12.9.1 base | Ubuntu host + PyTorch cu129 wheels; R570 additionally uses `cuda-compat-12-9` | Yes |
| Python 3.12 | Exact patch recorded in `python-version.txt` | Yes |
| Torch 2.10.0, torchvision 0.25.0, torchaudio 2.10.0 | Exact match | Yes |
| vLLM 0.18.0 | Exact match | Yes, rollout |
| Transformers 5.3.0 | Exact match | Yes |
| FlashAttention 2.8.3 | Source-built wheel on glibc 2.31; sm80+sm90 verified | Yes, HF full-attention layers |
| `nvidia-mathdx`, pybind11, ninja | Installed in Conda | Build/runtime dependency |
| causal-conv1d + FLA | Explicitly pinned and kernel-tested | Yes, Qwen3.5 GDN layers |
| Transformers Qwen3.5 FA2 guard | Idempotent source hotfix for 3-D M-RoPE IDs | Yes; 5.3.0 otherwise can issue invalid `cu_seqlens` |
| Project Verl fork | Editable install at recorded commit, `--no-deps` | Yes |
| cuDNN | Use the version carried/resolved by the tested PyTorch stack; do not mix a second untested loader path | Indirect; accepted by RL smoke |
| Apex | Not installed | No: actor/ref optimizer uses PyTorch FSDP2 |
| Transformer Engine | Not installed | No: attention implementation is FA2, not TE/FP8 |
| MBridge + Megatron-Core | Not installed | No: strategy is `fsdp2`, not `megatron` |
| DeepEP + GDRCopy + NVSHMEM | Not installed | No: Qwen3.5-4B is dense and has no expert-parallel route |
| Nsight Systems | Not installed; Python `nvtx` markers retained | No: `global_profiler.steps=null` |
| Habitat | Existing independent `vln` Conda env via env-server API | Yes, intentionally decoupled |

The omitted general-image components are not imported by the effective smoke
configuration. They should be added only if the project switches to Megatron,
MoE/DeepEP, FP8/Transformer Engine, or Nsight profiling; such a switch requires
a separate qualification rather than reusing this environment claim.
