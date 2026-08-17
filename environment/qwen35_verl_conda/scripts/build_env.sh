#!/bin/bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
WORK=${WORK:-$HOME/qwen35-verl-work}
CONDA_ROOT=${CONDA_ROOT:-$WORK/miniconda3}
ENV_PREFIX=${ENV_PREFIX:-$WORK/conda-envs/qwen35-verl-vllm018}
WORLDMODEL=${WORLDMODEL:-$(cd "$SCRIPT_DIR/../../.." && pwd -P)}
SCRIPTS_DIR=${QWEN35_SCRIPTS_DIR:-$SCRIPT_DIR}
VERL_SRC=${VERL_SRC:-$WORLDMODEL/vln/reinforcement_learning}
MODEL_DIR=${MODEL_DIR:-$WORK/checkpoints/Qwen3.5-4B-base}
MANIFEST_DIR=${MANIFEST_DIR:-$WORLDMODEL/artifacts/qwen35-conda-env}
LOCK_FILE=${LOCK_FILE:-$SCRIPTS_DIR/requirements-portable.txt}
PINNED_MODEL_REVISION_FILE=${PINNED_MODEL_REVISION_FILE:-$SCRIPTS_DIR/qwen35-model-revision.txt}
PINNED_MODEL_HASH_FILE=${PINNED_MODEL_HASH_FILE:-$SCRIPTS_DIR/qwen35-model-files.sha256}

export CONDA_PKGS_DIRS=${CONDA_PKGS_DIRS:-$WORK/.conda/pkgs}
export PIP_CACHE_DIR=${PIP_CACHE_DIR:-$WORK/.cache/pip}
export HF_HOME=${HF_HOME:-$WORK/.cache/huggingface-qwen35}
export TMPDIR=${TMPDIR:-$WORK/tmp/qwen35-build}
export TRITON_CACHE_DIR=${TRITON_CACHE_DIR:-$WORK/.cache/triton}
export TORCHINDUCTOR_CACHE_DIR=${TORCHINDUCTOR_CACHE_DIR:-$WORK/.cache/torchinductor}
export TOKENIZERS_PARALLELISM=false
mkdir -p "$CONDA_PKGS_DIRS" "$PIP_CACHE_DIR" "$HF_HOME" "$TMPDIR" \
  "$TRITON_CACHE_DIR" "$TORCHINDUCTOR_CACHE_DIR" "$MANIFEST_DIR" \
  "$(dirname "$ENV_PREFIX")"
rm -f "$MANIFEST_DIR/BUILD_SUCCESS" \
  "$MANIFEST_DIR/pip-check-unexpected.txt"

if [[ ! -x "$CONDA_ROOT/bin/conda" ]]; then
  echo "Missing Conda: $CONDA_ROOT/bin/conda" >&2
  exit 1
fi
for required in "$LOCK_FILE" "$PINNED_MODEL_REVISION_FILE" \
  "$PINNED_MODEL_HASH_FILE"; do
  test -s "$required" || {
    echo "Missing pinned input: $required" >&2
    exit 1
  }
done

if [[ ! -x "$ENV_PREFIX/bin/python" ]]; then
  "$CONDA_ROOT/bin/conda" create -y -p "$ENV_PREFIX" \
    --override-channels -c conda-forge \
    python=3.12.13 pip setuptools wheel packaging ninja cmake git git-lfs
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
    if [[ "$rpath" == *"$ENV_PREFIX"* ]]; then
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
  pip==26.2.1 setuptools==80.10.2 wheel==0.48.0 \
  packaging==25.0 ninja==1.13.0

# Match verl's Qwen3.5 vLLM image. CUDA is carried by PyTorch/vLLM wheels;
# the host only supplies the NVIDIA driver.
"${PIP[@]}" install \
  torch==2.10.0 torchvision==0.25.0 torchaudio==2.10.0 \
  --index-url https://download.pytorch.org/whl/cu129
"${PIP[@]}" install --no-deps vllm==0.18.0 transformers==5.3.0

# Install the exact non-core dependency closure from the validated environment.
"${PIP[@]}" install -r "$LOCK_FILE"
"$PY" "$SCRIPTS_DIR/patch_transformers_qwen35_fa.py"
# vLLM's unconstrained resolver currently selects OpenCV 5, whose metadata
# requires NumPy 2. Verl intentionally uses NumPy 1.x, so keep the last OpenCV
# release line compatible with NumPy 1.26.
"${PIP[@]}" install --force-reinstall --no-deps opencv-python-headless==4.11.0.86

# Qwen3.5 has GDN linear-attention layers. Build causal-conv1d locally so its
# host ABI matches this machine. Upstream's CUDA 12 build matrix includes both
# A100 (sm80) and H200 (sm90), and the checks below require both device images.
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
# sequence unpadding is disabled. Build it locally for this host ABI.
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

# Download the exact public Qwen3.5 base snapshot used for validation.
PINNED_MODEL_REVISION=$(tr -d '[:space:]' < "$PINNED_MODEL_REVISION_FILE")
test -n "$PINNED_MODEL_REVISION"
if [[ ! -s "$MODEL_DIR/config.json" ]]; then
  "$PY" - <<PY
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id='Qwen/Qwen3.5-4B',
    revision='$PINNED_MODEL_REVISION',
    local_dir='$MODEL_DIR',
    max_workers=2,
)
PY
fi
printf '%s\n' "$PINNED_MODEL_REVISION" > "$MANIFEST_DIR/qwen35-model-revision.txt"
(cd "$MODEL_DIR" && sha256sum --check "$PINNED_MODEL_HASH_FILE")

"$PY" "$SCRIPTS_DIR/check_stack.py" \
  --model "$MODEL_DIR" --expect-model-type qwen3_5 --run-kernel

"$PY" -m pip list --format=freeze > "$MANIFEST_DIR/pip-list.txt"
"$PY" -m pip freeze --all > "$MANIFEST_DIR/requirements-lock.txt"
"$CONDA_ROOT/bin/conda" list -p "$ENV_PREFIX" --explicit \
  > "$MANIFEST_DIR/conda-explicit.txt"
set +e
"$PY" -m pip check > "$MANIFEST_DIR/pip-check.txt" 2>&1
PIP_CHECK_RC=$?
set -e
if (( PIP_CHECK_RC != 0 )); then
  grep -vE '^vllm 0\.18\.0 has requirement transformers<5,>=4\.56\.0, but you have transformers 5\.3\.0\.$' \
    "$MANIFEST_DIR/pip-check.txt" | \
  grep -vE '^vllm 0\.18\.0 has requirement opencv-python-headless>=4\.13\.0, but you have opencv-python-headless 4\.11\.0\.86\.$' \
    > "$MANIFEST_DIR/pip-check-unexpected.txt" || true
  if [[ -s "$MANIFEST_DIR/pip-check-unexpected.txt" ]]; then
    cat "$MANIFEST_DIR/pip-check-unexpected.txt" >&2
    exit 1
  fi
fi
{
  echo "timestamp=$(date -Is)"
  echo "env=$ENV_PREFIX"
  echo "model=$MODEL_DIR"
  echo 'status=BUILD_SUCCESS'
} > "$MANIFEST_DIR/BUILD_SUCCESS"
(cd "$MANIFEST_DIR" && find . -type f ! -name SHA256SUMS -print0 | sort -z | \
  xargs -0 sha256sum > SHA256SUMS)
echo "BUILD_SUCCESS env=$ENV_PREFIX model=$MODEL_DIR manifests=$MANIFEST_DIR"
