"""CLI for the async, HTTP-backed rollout client (Task 7.3b).

Usage:
    python -m vln.rl.rollout_client.run_rollout \
        --env-server http://localhost:8002 \
        --vllm-base http://localhost:8001/v1 \
        --split val_unseen_sample33 \
        --policy-version sft-hfov79-merged \
        --num-episodes 2 --pass-k 4 \
        --out /tmp/rollout_v1.parquet
"""
import argparse
import asyncio
import json
import os
import time

import httpx
import pandas as pd

from vln.rl.env_server.client import AsyncEnvClient
from vln.rl.rollout_client.agent import AsyncRolloutAgent
from vln.rl.rollout_client.config import RolloutConfig
from vln.rl.rollout_client.rollout import run_episode, trajectory_seed, trajectory_to_rows
from vln.rl.rollout_client.selectors import build_selector


async def _main_async(args):
    cfg = RolloutConfig(
        forward_distance=args.forward_distance,
        turn_angle=args.turn_angle,
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_tokens,
    )
    selector = build_selector(cfg.frame_selection)

    env = AsyncEnvClient(args.env_server)
    eps = await env.episodes(limit=args.num_episodes)

    vllm = httpx.AsyncClient(base_url=args.vllm_base, timeout=120.0)
    model_list = await vllm.get("/models")
    model_list.raise_for_status()
    model_name = model_list.json()["data"][0]["id"]

    agent = AsyncRolloutAgent(
        vllm_client=vllm,
        model_name=model_name,
        cfg=cfg,
        selector=selector,
    )

    tasks = [(ep_id, trial_id) for ep_id in eps for trial_id in range(args.pass_k)]
    n_traj = len(tasks)
    sem = asyncio.Semaphore(args.concurrency)
    n_done = 0

    async def _one(ep_id, trial_id):
        nonlocal n_done
        async with sem:
            seed = trajectory_seed(args.seed, ep_id, trial_id)
            traj = await run_episode(env, agent, ep_id, trial_id, cfg, seed)
        n_done += 1
        print(f"[{n_done}/{n_traj}] ep={ep_id} trial={trial_id} "
              f"success={traj.success} spl={traj.spl:.2f} ne={traj.ne:.2f} "
              f"turns={len(traj.turns)} done_reason={traj.done_reason}")
        return traj

    t0 = time.time()
    trajs = await asyncio.gather(*(_one(ep, tr) for ep, tr in tasks))
    wall = time.time() - t0

    all_rows = []
    for traj in trajs:
        all_rows.extend(trajectory_to_rows(
            traj, cfg, split=args.split,
            policy_version=args.policy_version, rl_step=args.rl_step,
        ))

    await env.aclose()
    await vllm.aclose()

    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    df = pd.DataFrame(all_rows)
    df.to_parquet(args.out, index=False)
    summary = {
        "trajectories": n_traj,
        "rows": len(df),
        "unique_groups": int(df["group_id"].nunique()) if len(df) else 0,
        "unique_trajectories": int(df["trajectory_id"].nunique()) if len(df) else 0,
        "pass_at_k": float(df.drop_duplicates(["trajectory_id"]).groupby("group_id")["success"].max().mean()) if len(df) else 0.0,
        "mean_success": float(df.drop_duplicates(["trajectory_id"])["success"].mean()) if len(df) else 0.0,
        "concurrency": args.concurrency,
        "wall_time_s": wall,
        "s_per_traj_wall": wall / n_traj if n_traj else 0.0,
        "sum_vllm_s": float(sum(t.vllm_s for t in trajs)),
        "sum_env_s": float(sum(t.env_s for t in trajs)),
        "out": args.out,
    }
    print(json.dumps(summary, ensure_ascii=False))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env-server", required=True, help="env_server base URL, e.g. http://localhost:8002")
    parser.add_argument("--vllm-base", required=True, help="vLLM base URL incl. /v1, e.g. http://localhost:8001/v1")
    parser.add_argument("--split", required=True, help="dataset split name; only used for group_id stamping")
    parser.add_argument("--policy-version", required=True, help="opaque tag stamped into every row, e.g. sft-hfov79-merged")
    parser.add_argument("--rl-step", type=int, default=0)
    parser.add_argument("--num-episodes", type=int, default=2)
    parser.add_argument("--pass-k", type=int, default=4)
    parser.add_argument("--concurrency", type=int, default=1, help="max concurrent episodes; set <= env_server --pool-size")
    parser.add_argument("--forward-distance", type=int, default=25)
    parser.add_argument("--turn-angle", type=int, default=15)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    asyncio.run(_main_async(args))


if __name__ == "__main__":
    main()
