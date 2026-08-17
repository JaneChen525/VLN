#!/bin/bash
set -euo pipefail

WORK=${WORK:-/lustre/fsw/portfolios/general/users/jiaychen}
CONDA_ROOT=${CONDA_ROOT:-$WORK/miniconda3}
ENV_PREFIX=${ENV_PREFIX:-$WORK/conda-envs/qwen35-verl-vllm018}
WORLDMODEL=${WORLDMODEL:-$WORK/WorldModel}
SCRIPTS_DIR=${QWEN35_SCRIPTS_DIR:-$WORLDMODEL/scripts/qwen35_conda}
VERL_SRC=${VERL_SRC:-$WORLDMODEL/vln/reinforcement_learning}
MODEL_DIR=${MODEL_DIR:-$WORK/checkpoints/Qwen3.5-4B-base}
QWEN3VL_MODEL_DIR=${QWEN3VL_MODEL_DIR:-$WORK/checkpoints/Qwen3VL_4B_Final_swift}
MANIFEST_DIR=${MANIFEST_DIR:-$WORLDMODEL/artifacts/qwen35-conda-env}

export CONDA_PKGS_DIRS=${CONDA_PKGS_DIRS:-$WORK/.conda/pkgs}
export PIP_CACHE_DIR=${PIP_CACHE_DIR:-$WORK/.cache/pip}
export HF_HOME=${HF_HOME:-$WORK/.cache/huggingface-qwen35}
export TOKENIZERS_PARALLELISM=false
mkdir -p "$CONDA_PKGS_DIRS" "$PIP_CACHE_DIR" "$HF_HOME" "$MANIFEST_DIR" "$(dirname "$ENV_PREFIX")"
rm -f "$MANIFEST_DIR/BUILD_SUCCESS"

if [[ ! -x "$CONDA_ROOT/bin/conda" ]]; then
  echo "Missing Conda: $CONDA_ROOT/bin/conda" >&2
  exit 1
fi

if [[ ! -x "$ENV_PREFIX/bin/python" ]]; then
  "$CONDA_ROOT/bin/conda" create -y -p "$ENV_PREFIX" -c conda-forge \
    python=3.12 pip setuptools wheel packaging ninja cmake git git-lfs
fi

PY="$ENV_PREFIX/bin/python"
PIP=("$PY" -m pip)

repack_portable_wheel() {
  local wheel_path=$1
  local repack_root unpacked patched rpath so_file
  repack_root=$(mktemp -d "$MANIFEST_DIR/wheel-repack.XXXXXXXX")
  mkdir -p "$repack_root/unpack" "$repack_root/out"
  "$PY" -m wheel unpack "$wheel_path" -d "$repack_root/unpack" >/dev/null
  unpacked=$(find "$repack_root/unpack" -mindepth 1 -maxdepth 1 -type d -print -quit)
  test -n "$unpacked"
  while IFS= read -r -d '' so_file; do
    rpath=$("$ENV_PREFIX/bin/patchelf" --print-rpath "$so_file")
    if [[ "$rpath" == *"$ENV_PREFIX"* || "$rpath" == *'/lustre/'* ]]; then
      "$ENV_PREFIX/bin/patchelf" --remove-rpath "$so_file"
    fi
  done < <(find "$unpacked" -type f -name '*.so' -print0)
  "$PY" -m wheel pack "$unpacked" -d "$repack_root/out" >/dev/null
  patched=$(find "$repack_root/out" -maxdepth 1 -type f -name '*.whl' -print -quit)
  test -n "$patched"
  mv -f "$patched" "$wheel_path"
  rm -rf "$repack_root"
}

"${PIP[@]}" install --upgrade \
  pip 'setuptools>=77.0.3,<81.0.0' wheel 'packaging>=25.0,<26.0' ninja

# Match verl's Qwen3.5 vLLM image. CUDA is carried by PyTorch/vLLM wheels;
# the host only supplies the NVIDIA driver.
"${PIP[@]}" install \
  torch==2.10.0 torchvision==0.25.0 torchaudio==2.10.0 \
  --index-url https://download.pytorch.org/whl/cu129
"${PIP[@]}" install vllm==0.18.0
"${PIP[@]}" install transformers==5.3.0
"$PY" "$SCRIPTS_DIR/patch_transformers_qwen35_fa.py"

# verl runtime dependencies used by the FSDP2 + vLLM VLN path.
"${PIP[@]}" install \
  'numpy==1.26.4' \
  accelerate codetiming datasets dill hydra-core pandas peft 'pyarrow>=19.0.0' \
  pybind11 'ray[default]>=2.41.0' torchdata \
  'tensordict>=0.8.0,<=0.10.0,!=0.9.0' \
  wandb tensorboard cachetools pytest-asyncio \
  mathruler pylatexenc qwen_vl_utils nvtx matplotlib liger_kernel nvidia-mathdx \
  'pyelftools==0.32' \
  'einops==0.8.2'
# vLLM's unconstrained resolver currently selects OpenCV 5, whose metadata
# requires NumPy 2. Verl intentionally uses NumPy 1.x, so keep the last OpenCV
# release line compatible with NumPy 1.26.
"${PIP[@]}" install --force-reinstall --no-deps opencv-python-headless==4.11.0.86
"${PIP[@]}" install --no-deps trl==0.27.0

# Qwen3.5 has GDN linear-attention layers. Build causal-conv1d on the oldest
# target host (env4, glibc 2.31). Upstream's CUDA 12 build matrix includes both
# A100 (sm80) and H200 (sm90), so the wheel is portable across both GPUs.
# The resulting wheel can be handed to the H200 user without a host-GLIBC trap.
"$CONDA_ROOT/bin/conda" install -y -p "$ENV_PREFIX" \
  --override-channels -c nvidia/label/cuda-12.9.1 -c conda-forge \
  cuda-nvcc=12.9.86 cuda-cudart-dev=12.9.79 cuda-cuobjdump=12.9.82
# The cluster's conda-forge mirror does not expose patchelf 0.18. Pin the
# self-contained PyPI binary used only to sanitize build-host RPATH entries.
"${PIP[@]}" install patchelf==0.19.1.0
"$ENV_PREFIX/bin/patchelf" --version
export CUDA_HOME="$ENV_PREFIX"
export PATH="$CUDA_HOME/bin:$PATH"
CUDA_TARGET="$ENV_PREFIX/targets/x86_64-linux"
export CPATH="$CUDA_TARGET/include${CPATH:+:$CPATH}"
export LIBRARY_PATH="$CUDA_TARGET/lib${LIBRARY_PATH:+:$LIBRARY_PATH}"
export LD_LIBRARY_PATH="$CUDA_TARGET/lib:$CUDA_HOME/lib:${LD_LIBRARY_PATH:-}"
export TORCH_CUDA_ARCH_LIST='8.0;9.0'
WHEEL_DIR="$MANIFEST_DIR/wheels"
mkdir -p "$WHEEL_DIR"
"${PIP[@]}" uninstall -y causal-conv1d || true
CAUSAL_WHEEL=$(find "$WHEEL_DIR" -maxdepth 1 -type f -name 'causal_conv1d-1.6.2.post1-*.whl' -print -quit)
if [[ -z "$CAUSAL_WHEEL" ]]; then
  CAUSAL_CONV1D_FORCE_BUILD=TRUE MAX_JOBS=8 \
    "${PIP[@]}" wheel --no-build-isolation --no-deps --no-cache-dir \
    'git+https://github.com/Dao-AILab/causal-conv1d.git@v1.6.2.post1' \
    --wheel-dir "$WHEEL_DIR"
  CAUSAL_WHEEL=$(find "$WHEEL_DIR" -maxdepth 1 -type f -name 'causal_conv1d-1.6.2.post1-*.whl' -print -quit)
fi
test -n "$CAUSAL_WHEEL"
repack_portable_wheel "$CAUSAL_WHEEL"
"${PIP[@]}" install --no-deps "$CAUSAL_WHEEL"
(cd "$WHEEL_DIR" && sha256sum "$(basename "$CAUSAL_WHEEL")") \
  > "$MANIFEST_DIR/causal-conv1d-wheel.sha256"
CAUSAL_SO=$("$PY" -c 'import torch, causal_conv1d_cuda; print(causal_conv1d_cuda.__file__)')
"$ENV_PREFIX/bin/cuobjdump" --list-elf "$CAUSAL_SO" \
  > "$MANIFEST_DIR/causal-conv1d-cubin-list.txt"
grep -q 'sm_80' "$MANIFEST_DIR/causal-conv1d-cubin-list.txt"
grep -q 'sm_90' "$MANIFEST_DIR/causal-conv1d-cubin-list.txt"

# Match Dockerfile.stable.vllm: full-attention layers default to FA2 even when
# sequence unpadding is disabled. Build once on glibc 2.31 and bundle the wheel.
"${PIP[@]}" uninstall -y flash-attn || true
FLASH_WHEEL=$(find "$WHEEL_DIR" -maxdepth 1 -type f -name 'flash_attn-2.8.3*.whl' -print -quit)
if [[ -z "$FLASH_WHEEL" ]]; then
  FLASH_ATTENTION_FORCE_BUILD=TRUE FLASH_ATTN_CUDA_ARCHS='80;90' MAX_JOBS=4 \
    "${PIP[@]}" wheel --no-build-isolation --no-deps --no-cache-dir \
    flash_attn==2.8.3 --wheel-dir "$WHEEL_DIR"
  FLASH_WHEEL=$(find "$WHEEL_DIR" -maxdepth 1 -type f -name 'flash_attn-2.8.3*.whl' -print -quit)
fi
test -n "$FLASH_WHEEL"
repack_portable_wheel "$FLASH_WHEEL"
"${PIP[@]}" install --no-deps "$FLASH_WHEEL"
(cd "$WHEEL_DIR" && sha256sum "$(basename "$FLASH_WHEEL")") \
  > "$MANIFEST_DIR/flash-attn-wheel.sha256"
FLASH_SO=$("$PY" -c 'import torch, flash_attn_2_cuda; print(flash_attn_2_cuda.__file__)')
"$ENV_PREFIX/bin/cuobjdump" --list-elf "$FLASH_SO" \
  > "$MANIFEST_DIR/flash-attn-cubin-list.txt"
grep -q 'sm_80' "$MANIFEST_DIR/flash-attn-cubin-list.txt"
grep -q 'sm_90' "$MANIFEST_DIR/flash-attn-cubin-list.txt"

# Pin the known-good FLA line; 0.5.0 has a reported Qwen3.5-4B regression.
"${PIP[@]}" install --no-deps \
  fla-core==0.4.2 flash-linear-attention==0.4.2

# Keep the project fork as source of truth, like the container bind mount.
# --no-deps avoids the older setup.py vLLM upper bound replacing vLLM 0.18.
"${PIP[@]}" install --no-deps -e "$VERL_SRC"

echo '--- installed core versions ---'
"$PY" - <<'PY'
import importlib
for name in ('torch', 'torchvision', 'transformers', 'vllm', 'ray', 'tensordict', 'fla', 'causal_conv1d', 'verl'):
    module = importlib.import_module(name)
    print(f'{name}={getattr(module, "__version__", "unknown")} path={getattr(module, "__file__", "unknown")}')
PY

"$PY" "$SCRIPTS_DIR/check_stack.py" --run-kernel

# Download the official base checkpoint once. The fine-tuned checkpoint is not
# needed for an environment smoke test.
MODEL_REVISION_FILE="$MANIFEST_DIR/qwen35-model-revision.txt"
if [[ ! -s "$MODEL_DIR/config.json" || ! -s "$MODEL_REVISION_FILE" ]]; then
  "$PY" - <<PY
from pathlib import Path
from huggingface_hub import HfApi, snapshot_download
revision = HfApi().model_info('Qwen/Qwen3.5-4B', revision='main').sha
snapshot_download(
    repo_id='Qwen/Qwen3.5-4B',
    revision=revision,
    local_dir='$MODEL_DIR',
    max_workers=2,
)
Path('$MODEL_REVISION_FILE').write_text(revision + '\n', encoding='utf-8')
PY
fi

"$PY" "$SCRIPTS_DIR/check_stack.py" \
  --model "$MODEL_DIR" --expect-model-type qwen3_5 --run-kernel

if [[ -s "$QWEN3VL_MODEL_DIR/config.json" ]]; then
  "$PY" "$SCRIPTS_DIR/check_stack.py" \
    --model "$QWEN3VL_MODEL_DIR" --expect-model-type qwen3_vl
  find "$QWEN3VL_MODEL_DIR" -maxdepth 1 -type f \
    \( -name config.json -o -name '*.index.json' \) -print0 | sort -z | \
    xargs -0 sha256sum > "$MANIFEST_DIR/qwen3vl-metadata.sha256"
else
  echo "Missing Qwen3-VL compatibility checkpoint: $QWEN3VL_MODEL_DIR" >&2
  exit 1
fi

WORK="$WORK" WORLDMODEL="$WORLDMODEL" QWEN35_SCRIPTS_DIR="$SCRIPTS_DIR" \
  MANIFEST_DIR="$MANIFEST_DIR" ENV_PREFIX="$ENV_PREFIX" \
  CONDA_EXE="$CONDA_ROOT/bin/conda" VERL_SRC="$VERL_SRC" \
  QWEN35_MODEL_DIR="$MODEL_DIR" QWEN3VL_MODEL_DIR="$QWEN3VL_MODEL_DIR" \
  bash "$SCRIPTS_DIR/export_env.sh"
{
  echo "timestamp=$(date -Is)"
  echo "env=$ENV_PREFIX"
  echo "model=$MODEL_DIR"
  echo 'status=BUILD_SUCCESS'
} > "$MANIFEST_DIR/BUILD_SUCCESS"
(cd "$MANIFEST_DIR" && find . -type f ! -name SHA256SUMS -print0 | sort -z | \
  xargs -0 sha256sum > SHA256SUMS)
echo "BUILD_SUCCESS env=$ENV_PREFIX model=$MODEL_DIR manifests=$MANIFEST_DIR"
