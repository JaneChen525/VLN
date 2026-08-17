#!/bin/bash
#SBATCH --job-name=q35-conda-finalize
#SBATCH --account=general_perflab
#SBATCH --partition=polar,polar3,polar4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=/lustre/fsw/portfolios/general/users/jiaychen/logs/%x-%j.log
#SBATCH --error=/lustre/fsw/portfolios/general/users/jiaychen/logs/%x-%j.log

set -euo pipefail
umask 007

WORK=${WORK:-/lustre/fsw/portfolios/general/users/jiaychen}
WORLDMODEL=${WORLDMODEL:-$WORK/WorldModel}
BASE_ARTIFACT=${MANIFEST_DIR_OVERRIDE:-$WORLDMODEL/artifacts/qwen35-conda-env}
RESULT_ROOT=${RESULT_ROOT_OVERRIDE:-$WORLDMODEL/results/qwen-conda-smoke}
EVIDENCE_ROOT=${SMOKE_EVIDENCE_ROOT_OVERRIDE:-$RESULT_ROOT/evidence}
: "${ORIGINAL_Q35_JOB:?missing ORIGINAL_Q35_JOB}"
: "${RECREATE_JOB:?missing RECREATE_JOB}"
: "${FRESH_Q35_JOB:?missing FRESH_Q35_JOB}"
: "${FRESH_Q3VL_JOB:?missing FRESH_Q3VL_JOB}"

test -s "$BASE_ARTIFACT/BUILD_SUCCESS"
(cd "$BASE_ARTIFACT" && sha256sum --check SHA256SUMS)
test -s "$RESULT_ROOT/qwen35/$ORIGINAL_Q35_JOB/SMOKE_SUCCESS"
test -s "$RESULT_ROOT/recreate/$RECREATE_JOB/RECREATE_SUCCESS"
test -s "$RESULT_ROOT/qwen35/$FRESH_Q35_JOB/SMOKE_SUCCESS"
test -s "$RESULT_ROOT/qwen3vl/$FRESH_Q3VL_JOB/SMOKE_SUCCESS"
test -s "$EVIDENCE_ROOT/qwen35/$ORIGINAL_Q35_JOB/SMOKE_SUCCESS"
test -s "$EVIDENCE_ROOT/qwen35/$FRESH_Q35_JOB/SMOKE_SUCCESS"
test -s "$EVIDENCE_ROOT/qwen3vl/$FRESH_Q3VL_JOB/SMOKE_SUCCESS"
BASE_MANIFEST_DIGEST=$(sha256sum "$BASE_ARTIFACT/SHA256SUMS" | awk '{print $1}')
for evidence in \
  "$EVIDENCE_ROOT/qwen35/$ORIGINAL_Q35_JOB/SMOKE_SUCCESS" \
  "$EVIDENCE_ROOT/qwen35/$FRESH_Q35_JOB/SMOKE_SUCCESS" \
  "$EVIDENCE_ROOT/qwen3vl/$FRESH_Q3VL_JOB/SMOKE_SUCCESS"; do
  grep -Fxq "artifact_manifest_sha256=$BASE_MANIFEST_DIGEST" "$evidence" || {
    echo "Evidence was produced from a different artifact: $evidence" >&2
    exit 1
  }
done

RELEASE_DIR=${RELEASE_DIR_OVERRIDE:-$WORLDMODEL/artifacts/qwen35-conda-release-$SLURM_JOB_ID}
if [[ -e "$RELEASE_DIR" ]]; then
  echo "Refusing to overwrite release directory: $RELEASE_DIR" >&2
  exit 1
fi
mkdir -p "$RELEASE_DIR/validation"
cp -a "$BASE_ARTIFACT/." "$RELEASE_DIR/"
cp -a "$EVIDENCE_ROOT/qwen35/$ORIGINAL_Q35_JOB" \
  "$RELEASE_DIR/validation/original-qwen35"
cp -a "$RESULT_ROOT/recreate/$RECREATE_JOB" \
  "$RELEASE_DIR/validation/recreate"
cp -a "$EVIDENCE_ROOT/qwen35/$FRESH_Q35_JOB" \
  "$RELEASE_DIR/validation/recreated-qwen35"
cp -a "$EVIDENCE_ROOT/qwen3vl/$FRESH_Q3VL_JOB" \
  "$RELEASE_DIR/validation/recreated-qwen3vl"

cat > "$RELEASE_DIR/ACCEPTANCE.md" <<EOF
# Acceptance evidence

- Original environment Qwen3.5 RL step: $ORIGINAL_Q35_JOB
- Fresh-prefix reconstruction: $RECREATE_JOB
- Recreated environment Qwen3.5 RL step: $FRESH_Q35_JOB
- Recreated environment Qwen3-VL RL step: $FRESH_Q3VL_JOB
- Finalized: $(date -Is) on $(hostname)
EOF
{
  echo "timestamp=$(date -Is)"
  echo "original_qwen35_job=$ORIGINAL_Q35_JOB"
  echo "recreate_job=$RECREATE_JOB"
  echo "fresh_qwen35_job=$FRESH_Q35_JOB"
  echo "fresh_qwen3vl_job=$FRESH_Q3VL_JOB"
  echo 'status=RELEASE_SUCCESS'
} > "$RELEASE_DIR/RELEASE_SUCCESS"
(cd "$RELEASE_DIR" && find . -type f ! -name SHA256SUMS -print0 | sort -z | \
  xargs -0 sha256sum > SHA256SUMS)

ARCHIVE="$RELEASE_DIR.tar.gz"
tar -C "$(dirname "$RELEASE_DIR")" -czf "$ARCHIVE" "$(basename "$RELEASE_DIR")"
(cd "$(dirname "$ARCHIVE")" && sha256sum "$(basename "$ARCHIVE")") \
  > "$ARCHIVE.sha256"
echo "RELEASE_SUCCESS dir=$RELEASE_DIR archive=$ARCHIVE"
