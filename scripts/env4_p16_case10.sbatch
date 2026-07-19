#!/bin/bash
#SBATCH --job-name=p16-credit-case10
#SBATCH --account=general_perflab
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --partition=polar,polar3,polar4
#SBATCH --time=02:00:00
#SBATCH --gpus=8
#SBATCH --output=/lustre/fsw/portfolios/general/users/jiaychen/logs/%x-%j.log
#SBATCH --error=/lustre/fsw/portfolios/general/users/jiaychen/logs/%x-%j.log

set -euo pipefail
umask 007

WORK=/lustre/fsw/portfolios/general/users/jiaychen
WORLDMODEL=${WORLDMODEL:-$WORK/WorldModel-p16-case10}
PORT=${PORT:-8016}
EXP=${EXP:-p16-decision-credit-case10-${SLURM_JOB_ID}}
RESULT_DIR=$WORK/results/p16-decision-credit-case10-${SLURM_JOB_ID}
CASE_FILE=$RESULT_DIR/train_case10.parquet
TRACE_PATH=$RESULT_DIR/decision_trace.jsonl
ENV_LOG=$WORK/logs/env-server-p16-case10-${SLURM_JOB_ID}.log
CONFIG=$RESULT_DIR/vln_r2r_env4.yaml

mkdir -p "$WORK/logs" "$RESULT_DIR"

export CASE_SPECS='1LXtFkjw3qL:2134,VzqfbhrpDEA:5760,1LXtFkjw3qL:3273,VzqfbhrpDEA:9797,5q7pvUzZiYa:740,vyrNrziPKCB:1722,82sE5b5pLXE:3783,ac26ZMwG7aT:4983,kEZ7cmS4wCh:6803,8WUmhLawc2A:1080'
export CASE_SOURCE=$WORK/data/manifests/vln_r2r_train_filtered_966.parquet
export CASE_FILE
"$WORK/miniconda3/envs/vln/bin/python" - <<'PY'
import os
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

selected = [tuple(item.split(":")) for item in os.environ["CASE_SPECS"].split(",")]
table = pq.read_table(os.environ["CASE_SOURCE"])
lookup = {}
for index, row in enumerate(table.select(["extra_info"]).to_pylist()):
    info = row["extra_info"]
    scene = Path(str(info["scene_id"])).stem
    lookup[(scene, str(info["episode_id"]))] = index

missing = [key for key in selected if key not in lookup]
if missing:
    raise RuntimeError(f"case episodes missing from manifest: {missing}")
indices = [lookup[key] for key in selected]
case_table = table.take(pa.array(indices, type=pa.int64()))
if case_table.num_rows != 10 or len(set(indices)) != 10:
    raise RuntimeError(f"invalid case manifest: rows={case_table.num_rows}, indices={indices}")
pq.write_table(case_table, os.environ["CASE_FILE"])
print(f"CASE_MANIFEST_OK rows={case_table.num_rows} path={os.environ['CASE_FILE']}")
PY

cp "$WORLDMODEL/config/vln_r2r.yaml" "$CONFIG"
sed -i -E 's/(hfov:).*/\1 90/' "$CONFIG"
sed -i -E 's/(split:).*/\1 train/' "$CONFIG"
sed -i -E "s|(scenes_dir:).*|\1 $WORK/data/scene_datasets|" "$CONFIG"
sed -i -E "s|(data_path:).*|\1 $WORK/data/vln_eval_datasets/r2r/{split}/{split}.json.gz|" "$CONFIG"

cd "$WORLDMODEL"
PYTHONPATH="$WORLDMODEL/vln/reinforcement_learning:$WORLDMODEL:$WORLDMODEL/vln:${PYTHONPATH:-}" \
  "$WORK/miniconda3/envs/vln/bin/python" \
  -m recipe.vln_navida.env_server.launch \
  --exp-config "$CONFIG" \
  --port "$PORT" --pool-size 10 \
  --gpu-ids 0,1,2,3,4,5,6,7 --split train \
  > "$ENV_LOG" 2>&1 &
ENV_PID=$!

cleanup() {
  kill "$ENV_PID" 2>/dev/null || true
  wait "$ENV_PID" 2>/dev/null || true
}
trap cleanup EXIT

for _ in $(seq 1 180); do
  if ! kill -0 "$ENV_PID" 2>/dev/null; then
    tail -n 100 "$ENV_LOG"
    exit 1
  fi
  STATS=$(curl -fsS "http://127.0.0.1:$PORT/stats" 2>/dev/null || true)
  [[ "$STATS" == *'"free_workers":10'* ]] && break
  sleep 5
done
[[ "${STATS:-}" == *'"free_workers":10'* ]] || { tail -n 100 "$ENV_LOG"; exit 1; }
echo "env_server ready: $STATS"

export CASE_FIRST_EPISODE=2134
export CASE_PORT=$PORT
"$WORK/miniconda3/envs/vln/bin/python" - <<'PY'
import json
import os
import urllib.request

base = f"http://127.0.0.1:{os.environ['CASE_PORT']}"

def request(path, payload=None, method=None):
    data = None if payload is None else json.dumps(payload).encode()
    req = urllib.request.Request(
        base + path,
        data=data,
        method=method or ("POST" if data is not None else "GET"),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=120) as response:
        return json.load(response)

session = request("/v1/sessions", {"episode_id": os.environ["CASE_FIRST_EPISODE"]})
session_id = session["session_id"]
try:
    map_position = session["metrics"]["map_position"]
    assert len(map_position) == 2, map_position
    print(f"MAP_POSITION_OK episode={os.environ['CASE_FIRST_EPISODE']} xy={map_position}")
finally:
    request(f"/v1/sessions/{session_id}", method="DELETE")
PY

srun --overlap \
  --container-image=verlai/verl:vllm011.latest \
  --container-mounts=$WORK:$WORK \
  bash -c "
export WORLDMODEL=$WORLDMODEL
export AGENT_CFG=$WORLDMODEL/vln/reinforcement_learning/recipe/vln_navida/config/agent_loop.yaml
export PYTHONPATH=$WORLDMODEL/vln/reinforcement_learning:$WORLDMODEL:$WORLDMODEL/vln:\$PYTHONPATH
export VLN_ENV_SERVER_URL=http://127.0.0.1:$PORT
export VLLM_MM_INPUT_CACHE_GIB=16
export ROLLOUT_SCHEDULER=sliding
export ROLLOUT_WINDOW=10
export VLN_REWARD_MODE=p15_dense
export VLN_DENSE_GAMMA=0.95
export VLN_CREDIT_MODE=success_buffer
export VLN_DECISION_TRACE_PATH=$TRACE_PATH
export WANDB_MODE=disabled

cd $WORLDMODEL && \
MODEL_PATH=$WORK/checkpoints/Qwen3VL_4B_Final_swift \
TRAIN_FILE=$CASE_FILE \
VAL_FILE=$WORK/data/manifests/vln_r2r_val_unseen_1839.parquet \
TRAIN_BATCH_SIZE=10 ROLLOUT_N=8 TOTAL_STEPS=1 SAVE_FREQ=-1 \
NUM_WORKERS=10 ROLLOUT_WINDOW=10 \
PPO_MINI_BATCH_SIZE=10 PPO_MICRO_BATCH_SIZE=1 LOG_PROB_MICRO=1 \
ROLLOUT_TP=4 GPU_MEM_UTIL=0.4 ACTOR_LR=0 \
ADV_ESTIMATOR=decision_credit \
EXPERIMENT=$EXP \
bash vln/reinforcement_learning/recipe/vln_navida/run_grpo.sh \
trainer.logger='[console]'
"

test -s "$TRACE_PATH"
export TRACE_PATH
"$WORK/miniconda3/envs/vln/bin/python" - <<'PY'
import json
import math
import os
from collections import defaultdict

groups = defaultdict(set)
has_path = defaultdict(bool)
rollout_indices = defaultdict(set)
turn_ids = defaultdict(set)
trajectory_decisions = defaultdict(list)
decisions = 0
with open(os.environ["TRACE_PATH"]) as handle:
    for line in handle:
        row = json.loads(line)
        key = (row["scene_id"].split("/")[-1].split(".")[0], row["episode_id"])
        trajectory = row["trajectory_uid"]
        groups[key].add(trajectory)
        has_path[(key, trajectory)] |= bool(row.get("map_path"))
        rollout_indices[key].add(int(row["rollout_index"]))
        turn_ids[(key, trajectory)].add(int(row["turn_id"]))
        trajectory_decisions[(key, trajectory)].append(row)
        atomic_actions = row.get("atomic_actions") or []
        atomic_rewards = row.get("atomic_rewards") or []
        distance_path = row.get("distance_path") or []
        map_path = row.get("map_path") or []
        assert len(atomic_rewards) == len(atomic_actions), row
        assert len(distance_path) == len(atomic_actions) + 1, row
        assert len(map_path) == len(atomic_actions) + 1, row
        reconstructed = sum((0.95 ** index) * reward for index, reward in enumerate(atomic_rewards))
        assert math.isclose(reconstructed, row["decision_reward"], abs_tol=1e-5), row
        decisions += 1

assert len(groups) == 10, {key: len(value) for key, value in groups.items()}
assert all(len(value) == 8 for value in groups.values()), {
    key: len(value) for key, value in groups.items()
}
assert all(value == set(range(8)) for value in rollout_indices.values()), rollout_indices
assert len(has_path) == 80 and all(has_path.values()), {
    "trajectories": len(has_path), "with_path": sum(has_path.values())
}
assert len(turn_ids) == 80 and all(0 in value for value in turn_ids.values()), turn_ids
for rows in trajectory_decisions.values():
    running_return = 0.0
    for row in sorted(rows, key=lambda item: int(item["turn_id"]), reverse=True):
        running_return = row["decision_reward"] + row["discount_to_next"] * running_return
        assert math.isclose(running_return, row["p15_return"], abs_tol=1e-5), row
print(f"TRACE_AUDIT_OK groups={len(groups)} trajectories={len(has_path)} decisions={decisions}")
PY
echo "CASE10_TRACE_OK $TRACE_PATH"
