"""M1 parity test using expert (GT) action sequences from R2R DAgger/SFT data.

Stronger than random-action parity: expert traj should reach goal (success=1),
verify direct vs server produce identical metrics + JPEG bytes.

Usage:
    python -m vln.rl.env_server.parity_test_expert \
        --exp-config config/vln_r2r.yaml \
        --server-url http://localhost:8002 \
        --num-episodes 5
"""
import argparse
import base64
import gzip
import hashlib
import io
import json
import os
import re

import habitat
import numpy as np
from habitat import Env
from PIL import Image

from habitat_extensions import measures, task  # noqa: F401
from vln.rl.env_server.client import EnvClient
from vln.rl.env_server.worker import _load_env


def _jpeg_bytes(rgb: np.ndarray) -> bytes:
    img = Image.fromarray(rgb.astype("uint8")).convert("RGB")
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return buf.getvalue()


def _load_gt(gt_path: str) -> dict:
    with gzip.open(gt_path, "rt") as f:
        return json.load(f)


def _parse_action_list(raw) -> list[int]:
    if isinstance(raw, list):
        return [int(x) for x in raw]
    if isinstance(raw, str):
        # GT sometimes serializes as Python-style str "[1, 2, 3]"
        return [int(x) for x in re.findall(r"-?\d+", raw)]
    raise ValueError(f"unsupported actions field type: {type(raw)}")


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
    jpeg_hashes = [hashlib.sha256(_jpeg_bytes(f)).hexdigest() for f in raw_frames]
    return {
        "success": float(info["success"]),
        "spl": float(info["spl"]),
        "ne": float(info["distance_to_goal"]),
        "oracle_success": float(info["oracle_success"]),
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
        "jpeg_hashes": jpeg_hashes, "num_steps": num_steps,
    }


def compare(direct: dict, server: dict) -> dict:
    n = min(len(direct["jpeg_hashes"]), len(server["jpeg_hashes"]))
    return {
        "direct_success": direct["success"],
        "server_success": server["success"],
        "success_eq": direct["success"] == server["success"],
        "spl_abs_diff": abs(direct["spl"] - server["spl"]),
        "ne_abs_diff": abs(direct["ne"] - server["ne"]),
        "oracle_eq": direct["oracle_success"] == server["oracle_success"],
        "num_steps_eq": direct["num_steps"] == server["num_steps"],
        "jpeg_n_compared": n,
        "jpeg_eq_count": sum(1 for i in range(n) if direct["jpeg_hashes"][i] == server["jpeg_hashes"][i]),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp-config", type=str, required=True)
    parser.add_argument("--server-url", type=str, default="http://localhost:8002")
    parser.add_argument("--gt-path", type=str, default=None,
                        help="Path to _gt.json.gz. Default = derive from exp-config split.")
    parser.add_argument("--num-episodes", type=int, default=5)
    args = parser.parse_args()

    env, episodes_by_id = _load_env(args.exp_config)
    assert args.gt_path is not None, "must pass --gt-path (e.g. data/vln_eval_datasets/r2r/<split>/<split>_gt.json.gz)"
    print(f"loading GT: {args.gt_path}")
    gt = _load_gt(args.gt_path)

    client = EnvClient(args.server_url)
    assert client.healthz()["ok"]

    eps = list(episodes_by_id.values())[: args.num_episodes]
    rows = []
    for ep in eps:
        ep_id = str(ep.episode_id)
        traj_id = str(ep.trajectory_id) if hasattr(ep, "trajectory_id") else None
        gt_key = traj_id if traj_id in gt else ep_id
        if gt_key not in gt:
            print(f"[ep={ep_id}] no GT key ({traj_id} or {ep_id}), skip")
            continue
        actions = _parse_action_list(gt[gt_key]["actions"])
        if not actions:
            print(f"[ep={ep_id}] empty action list, skip")
            continue
        if actions[-1] != 0:
            actions = actions + [0]  # ensure stop

        direct = direct_rollout(env, ep, actions)
        server = server_rollout(client, ep_id, actions)
        diffs = compare(direct, server)
        diffs["episode_id"] = ep_id
        diffs["trajectory_id"] = traj_id
        diffs["n_expert_actions"] = len(actions)
        rows.append(diffs)
        print(f"[ep={ep_id} traj={traj_id} n_act={len(actions)}] {diffs}")

    env.close()
    client.close()

    n_pass = sum(
        1 for r in rows
        if r["success_eq"] and r["spl_abs_diff"] < 1e-4 and r["ne_abs_diff"] < 1e-3
        and r["num_steps_eq"] and r["jpeg_eq_count"] == r["jpeg_n_compared"]
    )
    n_expert_success = sum(1 for r in rows if r["direct_success"] >= 0.5)
    print(f"\n=== parity {n_pass}/{len(rows)} pass ===")
    print(f"=== expert reaches goal (direct) {n_expert_success}/{len(rows)} ===")
    if n_pass != len(rows):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
