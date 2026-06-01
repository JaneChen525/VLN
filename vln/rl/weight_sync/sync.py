"""Scheme A weight sync: restart the standalone vLLM server from a new ckpt.

Runs HOST-side (vln conda env, where `vllm serve` lives) — the rollout side
calls this once the trainer publishes a new ckpt. vLLM 0.16 standalone serve
has no weight-update HTTP endpoint (see README / 7.3d-0 spike), so the only
path for full-FT weights is restart-from-ckpt.

Gotcha encoded here: `pkill -f "vllm serve"` does NOT kill the EngineCore /
Worker_TP subprocesses; they leak GPU memory and block the next launch. We kill
by PID from `nvidia-smi --query-compute-apps` on the target GPUs.

    python -m vln.rl.weight_sync.sync \
        --ckpt /path/to/ckpt --gpus 4,5 --port 8001 \
        --served-model-name qwen3vl
"""
import argparse
import os
import subprocess
import time

import httpx


def _gpu_uuids(gpus: list[int]) -> dict[str, int]:
    out = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=index,uuid", "--format=csv,noheader"], text=True)
    idx_by_uuid = {}
    for line in out.strip().splitlines():
        idx, uuid = [x.strip() for x in line.split(",")]
        if int(idx) in gpus:
            idx_by_uuid[uuid] = int(idx)
    return idx_by_uuid


def find_vllm_pids(gpus: list[int]) -> list[int]:
    """PIDs holding compute memory on the target GPUs (the vLLM workers)."""
    uuids = _gpu_uuids(gpus)
    out = subprocess.check_output(
        ["nvidia-smi", "--query-compute-apps=gpu_uuid,pid", "--format=csv,noheader"], text=True)
    pids = []
    for line in out.strip().splitlines():
        if not line.strip():
            continue
        uuid, pid = [x.strip() for x in line.split(",")]
        if uuid in uuids:
            pids.append(int(pid))
    return sorted(set(pids))


def stop_vllm(gpus: list[int], timeout: float = 60.0):
    pids = find_vllm_pids(gpus)
    for pid in pids:
        subprocess.run(["kill", "-9", str(pid)], check=False)
    # wait until the target GPUs are actually free
    t0 = time.time()
    while time.time() - t0 < timeout:
        if not find_vllm_pids(gpus):
            return
        time.sleep(2)
    raise TimeoutError(f"GPUs {gpus} still busy after kill: {find_vllm_pids(gpus)}")


def start_vllm(ckpt: str, gpus: list[int], port: int, served_model_name: str,
               tp: int, max_model_len: int, log_path: str) -> subprocess.Popen:
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=",".join(str(g) for g in gpus))
    cmd = [
        "vllm", "serve", ckpt,
        "--served-model-name", served_model_name,
        "--tensor-parallel-size", str(tp),
        "--port", str(port),
        "--max-model-len", str(max_model_len),
        "--enforce-eager",          # skip cudagraph capture -> faster restart
        "--trust-remote-code",
    ]
    log = open(log_path, "w")
    return subprocess.Popen(cmd, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)


def wait_ready(port: int, timeout: float = 900.0) -> float:
    t0 = time.time()
    url = f"http://localhost:{port}/v1/models"
    while time.time() - t0 < timeout:
        try:
            if httpx.get(url, timeout=3).status_code == 200:
                return time.time() - t0
        except Exception:
            pass
        time.sleep(5)
    raise TimeoutError(f"vLLM not ready on :{port} after {timeout}s")


def sync(ckpt: str, gpus: list[int], port: int, served_model_name: str = "qwen3vl",
         tp: int = 2, max_model_len: int = 32768, log_path: str = "/tmp/vllm_sync.log") -> float:
    """Stop old vLLM on `gpus`, start from `ckpt`, block until ready. Returns
    end-to-end sync seconds."""
    t0 = time.time()
    stop_vllm(gpus)
    start_vllm(ckpt, gpus, port, served_model_name, tp, max_model_len, log_path)
    wait_ready(port)
    return time.time() - t0


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--gpus", required=True, help="comma-separated, e.g. 4,5")
    p.add_argument("--port", type=int, default=8001)
    p.add_argument("--served-model-name", default="qwen3vl")
    p.add_argument("--tp", type=int, default=2)
    p.add_argument("--max-model-len", type=int, default=32768)
    p.add_argument("--log-path", default="/tmp/vllm_sync.log")
    a = p.parse_args()
    gpus = [int(g) for g in a.gpus.split(",")]
    secs = sync(a.ckpt, gpus, a.port, a.served_model_name, a.tp, a.max_model_len, a.log_path)
    print(f"weight sync done in {secs:.1f}s; vLLM serving {a.ckpt} on :{a.port}")


if __name__ == "__main__":
    main()
