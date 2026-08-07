#!/usr/bin/env bash
# P17: P13 sparse trajectory reward with verl-native, turn-weighted GRPO statistics.

set -euo pipefail

export WORK=${WORK:-/lustre/fsw/portfolios/general/users/jiaychen}
export EXP=p17-turn-level-grpo-sparse-32x8
export TOTAL_STEPS=30
export STEPS_PER_JOB=4
export SAVE_FREQ=4
export PLANNED_JOBS=8
export BUFFER_JOBS=2
export VLN_REWARD_MODE=sparse_sr
export ADV_ESTIMATOR=grpo

exec bash "$WORK/WorldModel/scripts/env4_submit_grpo_chain.sh"
