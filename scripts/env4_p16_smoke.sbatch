#!/bin/bash
#SBATCH --job-name=p16-credit-smoke
#SBATCH --account=general_perflab
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --partition=polar,polar3,polar4
#SBATCH --time=01:00:00
#SBATCH --gpus=8
#SBATCH --output=/lustre/fsw/portfolios/general/users/jiaychen/logs/%x-%j.log
#SBATCH --error=/lustre/fsw/portfolios/general/users/jiaychen/logs/%x-%j.log

set -euo pipefail

WORK=/lustre/fsw/portfolios/general/users/jiaychen
WORLDMODEL=${WORLDMODEL:-$WORK/WorldModel-p16-smoke}
EXP=${EXP:-p16-decision-credit-smoke-${SLURM_JOB_ID}}
ENV_LOG=$WORK/logs/env-server-${SLURM_JOB_ID}.log

mkdir -p "$WORK/logs"
cd "$WORLDMODEL"
PYTHONPATH="$WORLDMODEL/vln/reinforcement_learning:$WORLDMODEL:$WORLDMODEL/vln:${PYTHONPATH:-}" \
  "$WORK/miniconda3/envs/vln/bin/python" \
  -m recipe.vln_navida.env_server.launch \
  --exp-config config/vln_r2r.yaml \
  --port 8002 --pool-size 8 \
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
  STATS=$(curl -fsS http://127.0.0.1:8002/stats 2>/dev/null || true)
  [[ "$STATS" == *'"free_workers":8'* ]] && break
  sleep 5
done
[[ "${STATS:-}" == *'"free_workers":8'* ]] || { tail -n 100 "$ENV_LOG"; exit 1; }
echo "env_server ready: $STATS"

# Verify Habitat heading convention before using it for directional credit.
"$WORK/miniconda3/envs/vln/bin/python" - <<'PY'
import json
import math
import urllib.request

base = "http://127.0.0.1:8002"

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

for episode_id in request("/v1/episodes?limit=20")["episode_ids"]:
    session = request("/v1/sessions", {"episode_id": str(episode_id)})
    session_id = session["session_id"]
    try:
        turned = request(f"/v1/sessions/{session_id}/step", {"actions": [2]})
        if turned["done"]:
            continue
        moved = request(f"/v1/sessions/{session_id}/step", {"actions": [1]})
        start = turned["metrics"]["position"]
        end = moved["metrics"]["position"]
        heading = moved["metrics"]["heading"]
        delta = [end[0] - start[0], end[2] - start[2]]
        norm = math.hypot(*delta)
        if norm > 1e-4:
            dot = delta[0] * heading[0] + delta[1] * heading[1]
            assert dot > 0, (delta, heading, dot)
            print(f"POSE_HEADING_OK episode={episode_id} dot={dot:.6f}")
            break
    finally:
        try:
            request(f"/v1/sessions/{session_id}", method="DELETE")
        except Exception:
            pass
else:
    raise RuntimeError("could not find a non-colliding pose sanity episode")
PY

srun --overlap \
  --container-image=verlai/verl:vllm011.latest \
  --container-mounts=$WORK:$WORK \
  bash -c "
export WORLDMODEL=$WORLDMODEL
export AGENT_CFG=$WORLDMODEL/vln/reinforcement_learning/recipe/vln_navida/config/agent_loop.yaml
export PYTHONPATH=$WORLDMODEL/vln/reinforcement_learning:$WORLDMODEL:$WORLDMODEL/vln:\$PYTHONPATH
export VLLM_MM_INPUT_CACHE_GIB=4
export ROLLOUT_SCHEDULER=sliding
export ROLLOUT_WINDOW=8
export VLN_REWARD_MODE=p15_dense
export VLN_DENSE_GAMMA=0.95
export VLN_CREDIT_MODE=success_buffer

cd $WORLDMODEL && \
MODEL_PATH=$WORK/checkpoints/Qwen3VL_4B_Final_swift \
TRAIN_FILE=$WORK/data/manifests/vln_r2r_train_filtered_966.parquet \
VAL_FILE=$WORK/data/manifests/vln_r2r_val_unseen_1839.parquet \
TRAIN_BATCH_SIZE=1 ROLLOUT_N=8 TOTAL_STEPS=1 SAVE_FREQ=-1 \
NUM_WORKERS=8 ROLLOUT_WINDOW=8 \
PPO_MINI_BATCH_SIZE=1 PPO_MICRO_BATCH_SIZE=1 LOG_PROB_MICRO=1 \
ROLLOUT_TP=4 GPU_MEM_UTIL=0.4 \
ADV_ESTIMATOR=decision_credit \
EXPERIMENT=$EXP \
bash vln/reinforcement_learning/recipe/vln_navida/run_grpo.sh \
trainer.logger='[console]'
"
