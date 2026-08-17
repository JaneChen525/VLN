#!/bin/bash
set -euo pipefail

WORK=${WORK:-/lustre/fsw/portfolios/general/users/jiaychen}
ENV_PREFIX=${ENV_PREFIX:-$WORK/conda-envs/qwen35-verl-vllm018}
WORLDMODEL=${WORLDMODEL:-$WORK/WorldModel}
VERL_SRC=${VERL_SRC:-$WORLDMODEL/vln/reinforcement_learning}
SCRIPTS_DIR=${QWEN35_SCRIPTS_DIR:-$WORLDMODEL/scripts/qwen35_conda}
OUT=${MANIFEST_DIR:-$WORLDMODEL/artifacts/qwen35-conda-env}
CONDA_EXE=${CONDA_EXE:-$WORK/miniconda3/bin/conda}
QWEN35_MODEL_DIR=${QWEN35_MODEL_DIR:-$WORK/checkpoints/Qwen3.5-4B-base}
QWEN3VL_MODEL_DIR=${QWEN3VL_MODEL_DIR:-$WORK/checkpoints/Qwen3VL_4B_Final_swift}
PY="$ENV_PREFIX/bin/python"
mkdir -p "$OUT"
test -x "$CONDA_EXE"

"$PY" -m pip list --format=freeze > "$OUT/pip-list.txt"
"$PY" -m pip freeze --all > "$OUT/requirements-lock.txt"
"$PY" -m pip list --format=json > "$OUT/pip-list.json"
"$PY" -c 'import platform; print(platform.python_version())' > "$OUT/python-version.txt"
# `pip freeze` can preserve Conda build-host file:// URLs. Build the portable
# exact-version phase from `pip list` instead, and stage conflicting/core
# packages separately in recreate_env.sh.
grep -E '^[A-Za-z0-9_.-]+==[^[:space:]]+$' "$OUT/pip-list.txt" | \
  grep -Evi '^(torch|torchvision|torchaudio|vllm|transformers|opencv-python-headless|causal[-_]conv1d|flash[-_]attn|fla-core|flash-linear-attention|patchelf|verl)==' \
  > "$OUT/requirements-portable.txt"
if grep -Eq 'file://|/lustre/| @ |^-e |[^=]==.*[[:space:]]' "$OUT/requirements-portable.txt"; then
  echo "requirements-portable.txt still contains a host-local dependency:" >&2
  grep -En 'file://|/lustre/| @ |^-e |[^=]==.*[[:space:]]' "$OUT/requirements-portable.txt" >&2
  exit 1
fi
# Pip packages are already captured above. Disabling Conda's pip interop avoids
# a second full metadata scan on Lustre.
CONDA_PIP_INTEROP_ENABLED=false "$CONDA_EXE" list -p "$ENV_PREFIX" --explicit \
  > "$OUT/conda-explicit.txt"
cat > "$OUT/conda-environment.yml" <<EOF
# Informational bootstrap only. The authoritative Conda lock is
# conda-explicit.txt; pip packages are in requirements-lock.txt.
name: qwen35-verl-vllm018
channels:
  - conda-forge
dependencies:
  - python=$(cat "$OUT/python-version.txt")
  - pip
EOF

set +e
rm -f "$OUT/pip-check-unexpected.txt"
"$PY" -m pip check > "$OUT/pip-check.txt" 2>&1
PIP_CHECK_RC=$?
set -e
if (( PIP_CHECK_RC != 0 )); then
  grep -vE '^vllm 0\.18\.0 has requirement transformers<5,>=4\.56\.0, but you have transformers 5\.3\.0\.$' \
    "$OUT/pip-check.txt" | \
  grep -vE '^vllm 0\.18\.0 has requirement opencv-python-headless>=4\.13\.0, but you have opencv-python-headless 4\.11\.0\.86\.$' \
    > "$OUT/pip-check-unexpected.txt" || true
  if [[ -s "$OUT/pip-check-unexpected.txt" ]]; then
    cat "$OUT/pip-check-unexpected.txt" >&2
    exit 1
  fi
fi

git -C "$WORLDMODEL" rev-parse HEAD > "$OUT/worldmodel-commit.txt"
git -C "$VERL_SRC" rev-parse HEAD > "$OUT/verl-commit.txt"
git -C "$WORLDMODEL" remote get-url origin > "$OUT/worldmodel-origin.txt"
git -C "$VERL_SRC" remote get-url origin \
  > "$OUT/verl-origin.txt"
cp "$SCRIPTS_DIR/README.md" "$OUT/README.md"
cp "$SCRIPTS_DIR/H200_HOST_SETUP.md" "$OUT/H200_HOST_SETUP.md"
cp "$SCRIPTS_DIR/DOCKER_PARITY.md" "$OUT/DOCKER_PARITY.md"
cp "$SCRIPTS_DIR/h200_host_preflight.sh" "$OUT/h200_host_preflight.sh"
cp "$SCRIPTS_DIR/recreate_env.sh" "$OUT/recreate_env.sh"
mkdir -p "$OUT/scripts"
if [[ $(readlink -f "$SCRIPTS_DIR") != $(readlink -f "$OUT/scripts") ]]; then
  cp "$SCRIPTS_DIR/"*.sh \
     "$SCRIPTS_DIR/"*.sbatch \
     "$SCRIPTS_DIR/"*.py \
     "$SCRIPTS_DIR/"*.md \
     "$SCRIPTS_DIR/"*.yaml \
     "$OUT/scripts/"
fi
git -C "$WORLDMODEL" status --short > "$OUT/worldmodel-status.txt"
git -C "$VERL_SRC" status --short > "$OUT/verl-status.txt"
git -C "$WORLDMODEL" diff --binary > "$OUT/worldmodel-working-tree.patch"
git -C "$WORLDMODEL" diff --binary --cached > "$OUT/worldmodel-index.patch"
git -C "$VERL_SRC" diff --binary \
  > "$OUT/verl-working-tree.patch"
git -C "$VERL_SRC" diff --binary --cached \
  > "$OUT/verl-index.patch"
git -C "$WORLDMODEL" ls-files --others --exclude-standard \
  > "$OUT/worldmodel-untracked-files.txt"
git -C "$VERL_SRC" ls-files --others --exclude-standard \
  > "$OUT/verl-untracked-files.txt"

if [[ -s "$QWEN35_MODEL_DIR/config.json" ]]; then
  (cd "$QWEN35_MODEL_DIR" && find . -maxdepth 1 -type f -print0 | sort -z | \
    xargs -0 -r sha256sum) > "$OUT/qwen35-model-files.sha256"
fi
if [[ -s "$QWEN3VL_MODEL_DIR/config.json" ]]; then
  (cd "$QWEN3VL_MODEL_DIR" && find . -maxdepth 1 -type f -print0 | sort -z | \
    xargs -0 -r sha256sum) > "$OUT/qwen3vl-model-files.sha256"
fi

{
  echo "generated_at=$(date -Is)"
  echo "hostname=$(hostname)"
  echo "python=$($PY --version 2>&1)"
  nvidia-smi --query-gpu=name,driver_version,memory.total,compute_cap --format=csv,noheader 2>/dev/null || true
} > "$OUT/build-host.txt"

(cd "$OUT" && find . -type f ! -name SHA256SUMS -print0 | sort -z | \
  xargs -0 sha256sum > SHA256SUMS)

echo "EXPORTED_ENV_MANIFESTS=$OUT"
