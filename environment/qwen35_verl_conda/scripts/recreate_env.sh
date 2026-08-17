#!/bin/bash
# Rebuild from the tested lock file. Run on an NVIDIA Linux host with Conda.
set -euo pipefail

if (( $# < 5 )); then
  echo "Usage: $0 <conda-root> <env-prefix> <artifact-dir> <worldmodel-source> <verl-source>" >&2
  exit 2
fi

CONDA_ROOT=$1
ENV_PREFIX=$2
ARTIFACT_DIR=$3
WORLDMODEL_SRC=$4
VERL_SRC=$5

DRIVER_VERSION=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | \
  sed -n '1{s/[[:space:]]//g;p;}' || true)
if [[ -n "$DRIVER_VERSION" ]] && \
   dpkg --compare-versions "$DRIVER_VERSION" lt 575.57.08 && \
   [[ -d /usr/local/cuda-12.9/compat ]]; then
  case ":${LD_LIBRARY_PATH:-}:" in
    *:/usr/local/cuda-12.9/compat:*) ;;
    *) export LD_LIBRARY_PATH="/usr/local/cuda-12.9/compat${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" ;;
  esac
fi
if [[ -e "$ENV_PREFIX" ]]; then
  echo "Refusing to reuse an existing environment prefix: $ENV_PREFIX" >&2
  exit 1
fi
(cd "$ARTIFACT_DIR" && sha256sum --check SHA256SUMS)

PYTHON_VERSION=$(cat "$ARTIFACT_DIR/python-version.txt")
EXPECTED_VERL_COMMIT=$(cat "$ARTIFACT_DIR/verl-commit.txt")
EXPECTED_WORLDMODEL_COMMIT=$(cat "$ARTIFACT_DIR/worldmodel-commit.txt")
ACTUAL_VERL_COMMIT=$(git -C "$VERL_SRC" rev-parse HEAD)
ACTUAL_WORLDMODEL_COMMIT=$(git -C "$WORLDMODEL_SRC" rev-parse HEAD)
if [[ "$ACTUAL_VERL_COMMIT" != "$EXPECTED_VERL_COMMIT" ]]; then
  echo "Verl commit mismatch: expected=$EXPECTED_VERL_COMMIT actual=$ACTUAL_VERL_COMMIT" >&2
  exit 1
fi
if [[ "$ACTUAL_WORLDMODEL_COMMIT" != "$EXPECTED_WORLDMODEL_COMMIT" ]]; then
  echo "WorldModel commit mismatch: expected=$EXPECTED_WORLDMODEL_COMMIT actual=$ACTUAL_WORLDMODEL_COMMIT" >&2
  exit 1
fi

"$CONDA_ROOT/bin/conda" create -y -p "$ENV_PREFIX" -c conda-forge \
  "python=$PYTHON_VERSION" pip
"$ENV_PREFIX/bin/python" -m pip install \
  torch==2.10.0 torchvision==0.25.0 torchaudio==2.10.0 \
  --index-url https://download.pytorch.org/whl/cu129
"$ENV_PREFIX/bin/python" -m pip install vllm==0.18.0
"$ENV_PREFIX/bin/python" -m pip install transformers==5.3.0
"$ENV_PREFIX/bin/python" \
  "$ARTIFACT_DIR/scripts/patch_transformers_qwen35_fa.py"
"$ENV_PREFIX/bin/python" -m pip install -r "$ARTIFACT_DIR/requirements-portable.txt"
"$ENV_PREFIX/bin/python" -m pip install --force-reinstall --no-deps \
  opencv-python-headless==4.11.0.86
CAUSAL_WHEEL=$(find "$ARTIFACT_DIR/wheels" -maxdepth 1 -type f -name 'causal_conv1d-1.6.2.post1-*.whl' -print -quit)
test -n "$CAUSAL_WHEEL"
(cd "$ARTIFACT_DIR/wheels" && \
  sha256sum --check "$ARTIFACT_DIR/causal-conv1d-wheel.sha256")
"$ENV_PREFIX/bin/python" -m pip install --no-deps "$CAUSAL_WHEEL"
FLASH_WHEEL=$(find "$ARTIFACT_DIR/wheels" -maxdepth 1 -type f -name 'flash_attn-2.8.3*.whl' -print -quit)
test -n "$FLASH_WHEEL"
(cd "$ARTIFACT_DIR/wheels" && \
  sha256sum --check "$ARTIFACT_DIR/flash-attn-wheel.sha256")
"$ENV_PREFIX/bin/python" -m pip install --no-deps "$FLASH_WHEEL"
"$ENV_PREFIX/bin/python" -m pip install einops==0.8.2
"$ENV_PREFIX/bin/python" -m pip install --no-deps \
  fla-core==0.4.2 flash-linear-attention==0.4.2
"$ENV_PREFIX/bin/python" -m pip install --no-deps -e "$VERL_SRC"

CHECK_STACK="$ARTIFACT_DIR/scripts/check_stack.py"
test -s "$CHECK_STACK"
"$ENV_PREFIX/bin/python" "$CHECK_STACK" --run-kernel

PIP_CHECK=$(mktemp)
UNEXPECTED=$(mktemp)
trap 'rm -f "$PIP_CHECK" "$UNEXPECTED"' EXIT
set +e
"$ENV_PREFIX/bin/python" -m pip check > "$PIP_CHECK" 2>&1
PIP_CHECK_RC=$?
set -e
if (( PIP_CHECK_RC != 0 )); then
  grep -vE '^vllm 0\.18\.0 has requirement transformers<5,>=4\.56\.0, but you have transformers 5\.3\.0\.$' \
    "$PIP_CHECK" | \
  grep -vE '^vllm 0\.18\.0 has requirement opencv-python-headless>=4\.13\.0, but you have opencv-python-headless 4\.11\.0\.86\.$' \
    > "$UNEXPECTED" || true
  if [[ -s "$UNEXPECTED" ]]; then
    cat "$UNEXPECTED" >&2
    exit 1
  fi
fi

echo "RECREATE_AND_KERNEL_CHECK_SUCCESS env=$ENV_PREFIX"
