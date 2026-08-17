#!/bin/bash
# Refresh code/docs provenance without re-scanning an unchanged pip environment.
set -euo pipefail

WORK=${WORK:-/lustre/fsw/portfolios/general/users/jiaychen}
ENV_PREFIX=${ENV_PREFIX:-$WORK/conda-envs/qwen35-verl-vllm018}
WORLDMODEL=${WORLDMODEL:-$WORK/WorldModel}
VERL_SRC=${VERL_SRC:-$WORLDMODEL/vln/reinforcement_learning}
SCRIPTS_DIR=${QWEN35_SCRIPTS_DIR:-$WORLDMODEL/scripts/qwen35_conda}
OUT=${MANIFEST_DIR:-$WORLDMODEL/artifacts/qwen35-conda-env}
CONDA_EXE=${CONDA_EXE:-$WORK/miniconda3/bin/conda}

for required in BUILD_SUCCESS pip-list.txt requirements-lock.txt \
  requirements-portable.txt pip-check.txt conda-explicit.txt \
  qwen35-model-files.sha256 qwen3vl-model-files.sha256; do
  test -s "$OUT/$required" || {
    echo "Cannot refresh incomplete base artifact: $OUT/$required" >&2
    exit 1
  }
done

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

cp "$SCRIPTS_DIR/README.md" "$OUT/README.md"
cp "$SCRIPTS_DIR/H200_HOST_SETUP.md" "$OUT/H200_HOST_SETUP.md"
cp "$SCRIPTS_DIR/DOCKER_PARITY.md" "$OUT/DOCKER_PARITY.md"
cp "$SCRIPTS_DIR/h200_host_preflight.sh" "$OUT/h200_host_preflight.sh"
cp "$SCRIPTS_DIR/recreate_env.sh" "$OUT/recreate_env.sh"
mkdir -p "$OUT/scripts"
cp "$SCRIPTS_DIR/"*.sh \
   "$SCRIPTS_DIR/"*.sbatch \
   "$SCRIPTS_DIR/"*.py \
   "$SCRIPTS_DIR/"*.md \
   "$SCRIPTS_DIR/"*.yaml \
   "$OUT/scripts/"

git -C "$WORLDMODEL" rev-parse HEAD > "$OUT/worldmodel-commit.txt"
git -C "$VERL_SRC" rev-parse HEAD > "$OUT/verl-commit.txt"
git -C "$WORLDMODEL" remote get-url origin > "$OUT/worldmodel-origin.txt"
git -C "$VERL_SRC" remote get-url origin \
  > "$OUT/verl-origin.txt"
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

{
  echo "refreshed_at=$(date -Is)"
  echo "environment=$ENV_PREFIX"
  echo 'scope=scripts_docs_source_provenance'
} > "$OUT/artifact-refresh.txt"
(cd "$OUT" && find . -type f ! -name SHA256SUMS -print0 | sort -z | \
  xargs -0 sha256sum > SHA256SUMS)
(cd "$OUT" && sha256sum --check SHA256SUMS >/dev/null)
echo "ARTIFACT_SCRIPT_REFRESH_SUCCESS=$OUT"
