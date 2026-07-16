#!/usr/bin/env bash
# Sync the newest local fragment for a fixed W&B run ID.

set -euo pipefail

WANDB_DIR=${1:?"Usage: $0 <wandb-dir> <run-id>"}
RUN_ID=${2:?"Usage: $0 <wandb-dir> <run-id>"}
MAX_ATTEMPTS=${WANDB_SYNC_MAX_ATTEMPTS:-3}

RUN_ROOT="$WANDB_DIR/wandb"
RUN_DIR=$(
  find "$RUN_ROOT" -mindepth 1 -maxdepth 1 -type d -name "run-*-$RUN_ID" \
    -print 2>/dev/null \
    | sort \
    | tail -n 1
)

if [[ -z "$RUN_DIR" ]]; then
  echo "No local W&B run found for ID: $RUN_ID" >&2
  exit 1
fi

for attempt in $(seq 1 "$MAX_ATTEMPTS"); do
  echo "W&B sync attempt $attempt/$MAX_ATTEMPTS: $RUN_DIR"
  if wandb sync --include-online --include-synced --append "$RUN_DIR"; then
    echo "W&B sync complete: $RUN_ID"
    exit 0
  fi
  if (( attempt < MAX_ATTEMPTS )); then
    sleep $((attempt * 5))
  fi
done

echo "W&B sync failed after $MAX_ATTEMPTS attempts: $RUN_ID" >&2
exit 1
