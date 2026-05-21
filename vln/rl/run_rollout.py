import argparse
import json
import os
import random
import time

import habitat
import numpy as np
import pandas as pd
import torch
from habitat import Env
from habitat.config.default import get_config
from habitat.config.default_structured_configs import (
    CollisionsMeasurementConfig,
    FogOfWarConfig,
    TopDownMapMeasurementConfig,
)
from tqdm import tqdm

from habitat_extensions import measures, task  # noqa: F401  (registers extensions; requires vln/ on PYTHONPATH)
from vln.rl.rollout import HabitatRolloutAgent, run_episode, trajectory_to_rows


def seed_all(seed: int = 0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def patch_config(config):
    with habitat.config.read_write(config):
        config.habitat.task.measurements.update(
            {
                "top_down_map": TopDownMapMeasurementConfig(
                    map_padding=3,
                    map_resolution=1024,
                    draw_source=True,
                    draw_border=True,
                    draw_shortest_path=True,
                    draw_view_points=True,
                    draw_goal_positions=True,
                    draw_goal_aabbs=True,
                    fog_of_war=FogOfWarConfig(draw=True, visibility_dist=5.0, fov=90),
                ),
                "collisions": CollisionsMeasurementConfig(),
            }
        )
    return config


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp-config", type=str, required=True)
    parser.add_argument("--num-episodes", type=int, default=2, help="how many distinct target episodes to roll out")
    parser.add_argument("--pass-k", type=int, default=4, help="rollouts per target episode (GRPO group size)")
    parser.add_argument("--forward-distance", type=int, default=25)
    parser.add_argument("--turn-angle", type=int, default=15)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=str, required=True, help="output parquet path")
    args = parser.parse_args()

    seed_all(args.seed)
    api_key = os.environ.get("OPENAI_API_KEY") or "EMPTY"
    base_url = os.environ.get("OPENAI_API_BASE")
    assert base_url is not None, "Set OPENAI_API_BASE to the vLLM endpoint"

    config = patch_config(get_config(args.exp_config))
    dataset = habitat.datasets.make_dataset(
        id_dataset=config.habitat.dataset.type, config=config.habitat.dataset
    )

    target_episodes = list(dataset.episodes)[: args.num_episodes]
    print(f"Rolling out {len(target_episodes)} episodes × pass_k={args.pass_k} = {len(target_episodes) * args.pass_k} trajectories")

    env = Env(config=config, dataset=dataset)
    agent = HabitatRolloutAgent(
        api_key=api_key,
        base_url=base_url,
        forward_distance=args.forward_distance,
        turn_angle=args.turn_angle,
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_tokens,
    )

    all_rows = []
    t0 = time.time()
    with tqdm(total=len(target_episodes) * args.pass_k, desc="rollout") as pbar:
        for target_ep in target_episodes:
            for trial_id in range(args.pass_k):
                traj = run_episode(env, agent, target_ep, trial_id)
                all_rows.extend(trajectory_to_rows(traj))
                pbar.set_postfix(success=traj.success, spl=f"{traj.spl:.2f}", ne=f"{traj.ne:.2f}", turns=len(traj.turns))
                pbar.update(1)
    env.close()

    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    df = pd.DataFrame(all_rows)
    df.to_parquet(args.out, index=False)
    elapsed = time.time() - t0

    summary = {
        "trajectories": len(target_episodes) * args.pass_k,
        "rows": len(df),
        "unique_groups": df["group_id"].nunique() if len(df) else 0,
        "mean_success": float(df.drop_duplicates(["group_id", "trial_id"])["success"].mean()) if len(df) else 0.0,
        "pass_at_k": float(df.groupby("group_id")["success"].max().mean()) if len(df) else 0.0,
        "wall_time_s": elapsed,
        "out": args.out,
    }
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
