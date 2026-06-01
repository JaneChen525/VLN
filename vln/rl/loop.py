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


def train(parquet, frames_dir, save_dir, ckpt_v0, fsdp_size, rl_step):
    inner = (
        f"cd {CONT_ROOT}/WorldModel && PYTHONPATH=. CUDA_VISIBLE_DEVICES=0,1,2,3 "
        f"python -m vln.rl.trainer.train_step "
        f"--ckpt {_host2cont(ckpt_v0)} --parquet {_host2cont(parquet)} --frames {_host2cont(frames_dir)} "
        f"--save-dir {_host2cont(save_dir)} --fsdp-size {fsdp_size} --max-rows 16 --rl-step {rl_step}"
    )
    subprocess.run(["docker", "exec", "verl-dev", "bash", "-lc", inner], check=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--env-server", required=True)
    p.add_argument("--vllm-port", type=int, default=8001)
    p.add_argument("--vllm-gpus", required=True, help="comma e.g. 4,5")
    p.add_argument("--ckpt-v0", required=True)
    p.add_argument("--work", required=True, help="host work dir for parquet/frames/ckpt")
    p.add_argument("--split", required=True)
    p.add_argument("--num-episodes", type=int, default=2)
    p.add_argument("--pass-k", type=int, default=4)
    p.add_argument("--fsdp-size", type=int, default=4)
    # T>0 required: greedy (T=0) makes all pass_k trials identical -> degenerate
    # GRPO group -> advantage=0 -> zero gradient (no learning).
    p.add_argument("--temperature", type=float, default=0.7)
    a = p.parse_args()

    gpus = [int(g) for g in a.vllm_gpus.split(",")]
    vbase = VLLM_BASE.format(port=a.vllm_port)
    os.makedirs(a.work, exist_ok=True)
    frames = os.path.join(a.work, "frames")
    p0 = os.path.join(a.work, "rollout_step0.parquet")
    p1 = os.path.join(a.work, "rollout_step1.parquet")
    ckpt1 = os.path.join(a.work, "ckpt_step1")

    # vLLM at v0 (SFT merged). served-model-name pinned so rollout model id is stable.
    start_vllm(a.ckpt_v0, gpus, a.vllm_port, "qwen3vl", tp=len(gpus),
               max_model_len=32768, log_path=os.path.join(a.work, "vllm.log"))
    wait_ready(a.vllm_port)
    print("[loop] vLLM up at v0")

    rollout(a.env_server, vbase, a.split, "v0", 0, p0, frames, a.num_episodes, a.pass_k, a.temperature)
    print("[loop] rollout_0 done ->", p0)

    train(p0, frames, ckpt1, a.ckpt_v0, a.fsdp_size, 0)
    print("[loop] train done ->", ckpt1)

    secs = sync(os.path.join(ckpt1, "huggingface"), gpus, a.vllm_port, "qwen3vl",
                tp=len(gpus), max_model_len=32768, log_path=os.path.join(a.work, "vllm.log"))
    print(f"[loop] weight sync done in {secs:.1f}s -> v1")

    rollout(a.env_server, vbase, a.split, "v1", 1, p1, frames, a.num_episodes, a.pass_k, a.temperature)
    print("[loop] rollout_1 done ->", p1)

    # M3 verify
    pv0 = pd.read_parquet(p0)["policy_version"].unique().tolist()
    pv1 = pd.read_parquet(p1)["policy_version"].unique().tolist()
    hf_ok = os.path.isdir(os.path.join(ckpt1, "huggingface"))
    print(json.dumps({"rollout0_policy_versions": pv0, "rollout1_policy_versions": pv1,
                      "ckpt_saved": hf_ok}))
    assert pv0 == ["v0"] and pv1 == ["v1"] and hf_ok, "M3 verify failed"
    print("7.4 1-step RL loop PASSED")


if __name__ == "__main__":
    main()
