"""7.4-2: host-side 1-step RL loop orchestrator (lock-step).

  rollout_0 (v0) -> docker train -> weight sync -> rollout_1 (v1) -> verify

Assumes env_server is already up (launch separately). Owns the vLLM lifecycle
via weight_sync. Trainer runs in the verl-dev docker via `docker exec`; host
and container share /var/data0/sandbox/janec (= /workspace), so parquet/frames/
ckpt pass through the mount with no copying.

  python -m vln.rl.loop \
    --env-server http://localhost:8002 --vllm-port 8001 --vllm-gpus 4,5 \
    --ckpt-v0 /var/data0/sandbox/janec/WorldModel/checkpoints/Qwen3-VL-4B-vln-r2r-merged \
    --work /var/data0/sandbox/janec/rl_test/loop \
    --split val_unseen_sample33 --num-episodes 2 --pass-k 4
"""
import argparse
import json
import os
import subprocess

import pandas as pd

from vln.rl.weight_sync.sync import start_vllm, wait_ready, sync

HOST_ROOT = "/var/data0/sandbox/janec"
CONT_ROOT = "/workspace"          # same bind mount inside verl-dev
VLLM_BASE = "http://localhost:{port}/v1"


def _host2cont(p: str) -> str:
    return p.replace(HOST_ROOT, CONT_ROOT, 1)


def rollout(env_server, vllm_base, split, policy_version, rl_step, out, frames_dir,
            num_episodes, pass_k, temperature):
    subprocess.run([
        "python", "-m", "vln.rl.rollout_client.run_rollout",
        "--env-server", env_server, "--vllm-base", vllm_base,
        "--split", split, "--policy-version", policy_version, "--rl-step", str(rl_step),
        "--num-episodes", str(num_episodes), "--pass-k", str(pass_k),
        "--out", out, "--frames-dir", frames_dir, "--temperature", str(temperature),
    ], check=True)


def train(parquet, frames_dir, save_dir, init_ckpt, fsdp_size, rl_step, max_rows):
    inner = (
        f"cd {CONT_ROOT}/WorldModel && PYTHONPATH=. CUDA_VISIBLE_DEVICES=0,1,2,3 "
        f"python -m vln.rl.trainer.train_step "
        f"--ckpt {_host2cont(init_ckpt)} --parquet {_host2cont(parquet)} --frames {_host2cont(frames_dir)} "
        f"--save-dir {_host2cont(save_dir)} --fsdp-size {fsdp_size} --max-rows {max_rows} --rl-step {rl_step}"
    )
    subprocess.run(["docker", "exec", "verl-dev", "bash", "-lc", inner], check=True)


def sr_of(parquet: str) -> tuple[float, float]:
    """(pass@k, mean_success) over trajectories in a rollout parquet."""
    df = pd.read_parquet(parquet).drop_duplicates("trajectory_id")
    pak = float(df.groupby("group_id")["success"].max().mean())
    ms = float(df["success"].mean())
    return pak, ms


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--env-server", required=True)
    p.add_argument("--vllm-port", type=int, default=8001)
    p.add_argument("--vllm-gpus", required=True, help="comma e.g. 4,5")
    p.add_argument("--ckpt-v0", required=True)
    p.add_argument("--work", required=True, help="host work dir for parquet/frames/ckpt")
    p.add_argument("--split", required=True)
    p.add_argument("--num-steps", type=int, default=1, help="RL steps; SR(v_i) is each step's rollout")
    p.add_argument("--num-episodes", type=int, default=2)
    p.add_argument("--pass-k", type=int, default=4)
    p.add_argument("--fsdp-size", type=int, default=4)
    p.add_argument("--max-rows", type=int, default=16)
    # T>0 required: greedy (T=0) makes all pass_k trials identical -> degenerate
    # GRPO group -> advantage=0 -> zero gradient (no learning).
    p.add_argument("--temperature", type=float, default=0.7)
    a = p.parse_args()

    gpus = [int(g) for g in a.vllm_gpus.split(",")]
    vbase = VLLM_BASE.format(port=a.vllm_port)
    os.makedirs(a.work, exist_ok=True)
    frames = os.path.join(a.work, "frames")

    # vLLM at v0 (SFT merged). served-model-name pinned so rollout model id is stable.
    start_vllm(a.ckpt_v0, gpus, a.vllm_port, "qwen3vl", tp=len(gpus),
               max_model_len=32768, log_path=os.path.join(a.work, "vllm.log"))
    wait_ready(a.vllm_port)
    print("[loop] vLLM up at v0")

    cur_ckpt = a.ckpt_v0          # HF dir the trainer inits from (chained each step)
    curve = []
    for step in range(a.num_steps):
        pv = f"v{step}"
        pq = os.path.join(a.work, f"rollout_step{step}.parquet")
        # rollout with the current policy v_step
        rollout(a.env_server, vbase, a.split, pv, step, pq, frames, a.num_episodes, a.pass_k, a.temperature)
        pak, ms = sr_of(pq)
        curve.append({"step": step, "policy": pv, "pass_at_k": pak, "mean_success": ms})
        print(f"[loop] step {step} rollout({pv}) pass@k={pak:.3f} mean_success={ms:.3f}")

        # train v_step -> ckpt_step{step+1}, chained from cur_ckpt
        ckpt_next = os.path.join(a.work, f"ckpt_step{step + 1}")
        train(pq, frames, ckpt_next, cur_ckpt, a.fsdp_size, step, a.max_rows)
        cur_ckpt = os.path.join(ckpt_next, "huggingface")

        # sync vLLM to v_{step+1}
        secs = sync(cur_ckpt, gpus, a.vllm_port, "qwen3vl", tp=len(gpus),
                    max_model_len=32768, log_path=os.path.join(a.work, "vllm.log"))
        print(f"[loop] step {step} sync {secs:.1f}s -> v{step + 1}")

    print("LEARNING CURVE:", json.dumps(curve))
    with open(os.path.join(a.work, "curve.json"), "w") as f:
        json.dump(curve, f, indent=2)
    print(f"{a.num_steps}-step RL loop DONE")


if __name__ == "__main__":
    main()
