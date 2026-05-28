"""M1 parity test: direct habitat.Env vs env_server HTTP.

Runs a fixed action sequence on the same episode through both paths and asserts
metrics + image tensor max_abs_diff within threshold. Run AFTER `launch.py` is up.

Usage:
    python -m vln.rl.env_server.parity_test \
        --exp-config config/vln_r2r.yaml \
        --server-url http://localhost:8002 \
        --num-episodes 2 \
        --num-steps 20
"""
import argparse
import base64
import hashlib
import io
import os
import random

import habitat
import numpy as np
from habitat import Env
from habitat.config.default import get_config
from PIL import Image

from habitat_extensions import measures, task  # noqa: F401
from vln.rl.env_server.client import EnvClient
from vln.rl.env_server.worker import _load_env, _encode_jpeg_b64


def _jpeg_bytes(rgb: np.ndarray) -> bytes:
    """Same encoder used in env_server.worker._encode_jpeg_b64 and
    vln/rl/messages.py:encode_image_base64. This is what the model sees."""
    img = Image.fromarray(rgb.astype("uint8")).convert("RGB")
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return buf.getvalue()


def direct_rollout(env: Env, ep, actions: list[int]) -> dict:
    env.current_episode = ep
    obs = env.reset()
    raw_frames = [obs["rgb"].copy()]
    for a in actions:
        if env.episode_over:
            break
        obs = env.step({"action": a})
        raw_frames.append(obs["rgb"].copy())
    info = env.get_metrics()
    # JPEG bytes that direct path would feed to the model
    jpeg_hashes = [hashlib.sha256(_jpeg_bytes(f)).hexdigest() for f in raw_frames]
    return {
        "success": float(info["success"]),
        "spl": float(info["spl"]),
        "ne": float(info["distance_to_goal"]),
        "oracle_success": float(info["oracle_success"]),
        "raw_frames": raw_frames,
        "jpeg_hashes": jpeg_hashes,
        "num_steps": len(raw_frames) - 1,
    }


def server_rollout(client: EnvClient, episode_id: str, actions: list[int]) -> dict:
    r = client.reset(episode_id)
    env_id = r["env_id"]
    jpeg_b64_list = [r["obs"]["rgb_jpeg_b64"]]
    success = spl = ne = oracle = 0.0
    num_steps = 0
    try:
        for a in actions:
            s = client.step(env_id, a)
            jpeg_b64_list.append(s["obs"]["rgb_jpeg_b64"])
            success, spl, ne, oracle = s["success"], s["spl"], s["ne"], s["oracle_success"]
            num_steps = s["step_count"]
            if s["done"]:
                break
    finally:
        client.delete_env(env_id)
    jpeg_hashes = [hashlib.sha256(base64.b64decode(b)).hexdigest() for b in jpeg_b64_list]
    return {
        "success": success, "spl": spl, "ne": ne, "oracle_success": oracle,
        "jpeg_b64": jpeg_b64_list,
        "jpeg_hashes": jpeg_hashes,
        "num_steps": num_steps,
    }


def compare(direct: dict, server: dict) -> dict:
    """Compare env behavior + JPEG bytes (what the model actually sees)."""
    diffs = {
        "success_eq": direct["success"] == server["success"],
        "spl_abs_diff": abs(direct["spl"] - server["spl"]),
        "ne_abs_diff": abs(direct["ne"] - server["ne"]),
        "oracle_eq": direct["oracle_success"] == server["oracle_success"],
        "num_steps_eq": direct["num_steps"] == server["num_steps"],
    }
    n = min(len(direct["jpeg_hashes"]), len(server["jpeg_hashes"]))
    if n > 0:
        eq = sum(1 for i in range(n) if direct["jpeg_hashes"][i] == server["jpeg_hashes"][i])
        diffs["jpeg_n_compared"] = n
        diffs["jpeg_eq_count"] = eq
        diffs["jpeg_all_eq"] = eq == n
    return diffs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp-config", type=str, required=True)
    parser.add_argument("--server-url", type=str, default="http://localhost:8002")
    parser.add_argument("--num-episodes", type=int, default=2)
    parser.add_argument("--num-steps", type=int, default=30)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    # Direct path
    env, episodes_by_id = _load_env(args.exp_config)
    episodes = list(episodes_by_id.values())[: args.num_episodes]

    client = EnvClient(args.server_url)
    h = client.healthz()
    assert h["ok"], f"server unhealthy: {h}"
    print(f"server healthz OK, pool_size={h['pool_size']}")

    all_results = []
    for ep in episodes:
        ep_id = str(ep.episode_id)
        # Fixed random action seq for this episode
        actions = [random.randint(1, 3) for _ in range(args.num_steps)] + [0]

        direct = direct_rollout(env, ep, actions)
        server = server_rollout(client, ep_id, actions)
        diffs = compare(direct, server)
        all_results.append({"episode_id": ep_id, "scene_id": os.path.basename(ep.scene_id).split(".")[0], **diffs})
        print(f"[ep={ep_id}] {diffs}")

    env.close()
    client.close()

    # Summary
    n_pass = sum(
        1 for r in all_results
        if r["success_eq"] and r["spl_abs_diff"] < 1e-4 and r["ne_abs_diff"] < 1e-3
        and r["num_steps_eq"] and r.get("jpeg_all_eq", True)
    )
    print(f"\n=== parity {n_pass}/{len(all_results)} pass ===")
    if n_pass != len(all_results):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
