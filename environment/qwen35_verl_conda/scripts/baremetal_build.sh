#!/bin/bash
# Build the Qwen3.5 Verl environment directly on a dedicated Linux GPU host.
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
if REPO_ROOT=$(git -C "$SCRIPT_DIR" rev-parse --show-toplevel 2>/dev/null); then
  :
else
  REPO_ROOT=$(cd "$SCRIPT_DIR/../../.." && pwd -P)
fi

WORK=${WORK:-$HOME/qwen35-verl-work}
WORLDMODEL=${WORLDMODEL:-$REPO_ROOT}
if [[ -z ${CONDA_ROOT:-} ]]; then
  CONDA_BIN=$(type -P conda 2>/dev/null || true)
  if [[ -n $CONDA_BIN ]]; then
    CONDA_BIN=$(readlink -f "$CONDA_BIN")
    CONDA_ROOT=$(cd "$(dirname "$CONDA_BIN")/.." && pwd -P)
  else
    echo 'Set CONDA_ROOT to an existing Miniconda/Miniforge installation.' >&2
    exit 2
  fi
fi

ENV_PREFIX=${ENV_PREFIX:-$WORK/conda-envs/qwen35-verl-vllm018}
VERL_SRC=${VERL_SRC:-$WORLDMODEL/vln/reinforcement_learning}
MODEL_DIR=${MODEL_DIR:-$WORK/checkpoints/Qwen3.5-4B-base}
MANIFEST_DIR=${MANIFEST_DIR:-$WORLDMODEL/artifacts/qwen35-conda-env}
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
ALLOW_RESUME=${ALLOW_RESUME:-0}

test -x "$CONDA_ROOT/bin/conda" || {
  echo "Conda executable not found: $CONDA_ROOT/bin/conda" >&2
  exit 1
}
VERL_TOP=$(git -C "$VERL_SRC" rev-parse --show-toplevel 2>/dev/null || true)
if [[ -z $VERL_TOP || $(readlink -f "$VERL_TOP") != $(readlink -f "$VERL_SRC") || \
      ! -s "$VERL_SRC/recipe/vln_navida/run_grpo.sh" ]]; then
  echo "Verl submodule is missing: $VERL_SRC" >&2
  echo 'Run the submodule command from README.md.' >&2
  exit 1
fi
EXPECTED_VERL_COMMIT=$(git -C "$WORLDMODEL" rev-parse HEAD:vln/reinforcement_learning)
ACTUAL_VERL_COMMIT=$(git -C "$VERL_SRC" rev-parse HEAD)
if [[ $ACTUAL_VERL_COMMIT != "$EXPECTED_VERL_COMMIT" ]]; then
  echo "Verl commit mismatch: expected=$EXPECTED_VERL_COMMIT actual=$ACTUAL_VERL_COMMIT" >&2
  exit 1
fi
command -v nvidia-smi >/dev/null

if (( ! ALLOW_RESUME )); then
  if [[ -e $ENV_PREFIX ]]; then
    echo "Refusing existing ENV_PREFIX: $ENV_PREFIX" >&2
    echo 'Choose a new path, or set ALLOW_RESUME=1 for an interrupted build.' >&2
    exit 1
  fi
  if [[ -d $MANIFEST_DIR && -n $(find "$MANIFEST_DIR" -mindepth 1 -print -quit) ]]; then
    echo "Refusing non-empty MANIFEST_DIR: $MANIFEST_DIR" >&2
    exit 1
  fi
fi

export WORK WORLDMODEL CONDA_ROOT ENV_PREFIX VERL_SRC MODEL_DIR
export MANIFEST_DIR CUDA_VISIBLE_DEVICES
export QWEN35_SCRIPTS_DIR=$SCRIPT_DIR

echo "BAREMETAL_BUILD_START host=$(hostname)"
printf 'work=%s\nrepo=%s\nconda=%s\nenv=%s\nmodel=%s\n' \
  "$WORK" "$WORLDMODEL" "$CONDA_ROOT" "$ENV_PREFIX" "$MODEL_DIR"
nvidia-smi --query-gpu=index,name,driver_version,memory.total,compute_cap \
  --format=csv,noheader

exec bash "$SCRIPT_DIR/build_env.sh"
