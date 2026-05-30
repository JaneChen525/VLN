"""GRPO grouping + advantage from rollout parquet (pure, no verl/GPU).

Contract with rollout_client v1 schema:
- one row per turn; reward_success is the terminal SR broadcast to every row
  of a trajectory.
- group_id = (split, episode_id, variant_hash)  — trial_id NOT included.
- trajectory_id = group_id | trial_<k>  — the GRPO sample index inside a group.

GRPO outcome advantage: within each group, A_i = (R_i - mean_R) / (std_R + eps)
over the G trajectories; the scalar A_i is then broadcast to every turn row of
trajectory i (and downstream to every assistant action token of that turn).
"""
from dataclasses import dataclass

import numpy as np
import pandas as pd

EPS = 1e-6


class MixedPolicyError(RuntimeError):
    pass


def assert_single_policy_version(df: pd.DataFrame) -> str:
    """Fail-fast: a train step must consume rows from exactly one policy_version."""
    versions = df["policy_version"].unique()
    if len(versions) != 1:
        raise MixedPolicyError(f"expected 1 policy_version per step, got {sorted(versions)}")
    return str(versions[0])


@dataclass
class GroupStats:
    n_groups: int
    group_size_mean: float
    all_zero_ratio: float
    all_one_ratio: float
    mixed_ratio: float
    effective_trajectories: int  # trajectories in non-degenerate (mixed) groups
    reward_mean: float
    adv_std: float


def compute_grpo_advantages(df: pd.DataFrame) -> tuple[pd.DataFrame, GroupStats]:
    """Add an `advantage` column (per-row, broadcast from trajectory scalar).

    Returns (df_with_advantage, group_stats). Does not mutate the input.
    """
    df = df.copy()

    # One scalar reward per trajectory (reward_success is already broadcast).
    traj = df.drop_duplicates("trajectory_id")[["trajectory_id", "group_id", "reward_success"]]

    # Within-group mean/std over trajectories.
    g = traj.groupby("group_id")["reward_success"]
    traj = traj.join(g.transform("mean").rename("_mean"))
    traj = traj.join(g.transform("std").rename("_std"))  # ddof=1; NaN if group size 1
    traj["_std"] = traj["_std"].fillna(0.0)
    traj["advantage"] = (traj["reward_success"] - traj["_mean"]) / (traj["_std"] + EPS)

    adv_map = dict(zip(traj["trajectory_id"], traj["advantage"]))
    df["advantage"] = df["trajectory_id"].map(adv_map)

    stats = _group_stats(traj)
    return df, stats


def _group_stats(traj: pd.DataFrame) -> GroupStats:
    sizes = traj.groupby("group_id").size()
    rmax = traj.groupby("group_id")["reward_success"].max()
    rmin = traj.groupby("group_id")["reward_success"].min()
    all_zero = (rmax == 0)
    all_one = (rmin == 1)
    mixed = ~(all_zero | all_one)
    n = len(sizes)
    mixed_groups = mixed[mixed].index
    eff = int(traj[traj["group_id"].isin(mixed_groups)].shape[0])
    return GroupStats(
        n_groups=n,
        group_size_mean=float(sizes.mean()) if n else 0.0,
        all_zero_ratio=float(all_zero.mean()) if n else 0.0,
        all_one_ratio=float(all_one.mean()) if n else 0.0,
        mixed_ratio=float(mixed.mean()) if n else 0.0,
        effective_trajectories=eff,
        reward_mean=float(traj["reward_success"].mean()) if len(traj) else 0.0,
        adv_std=float(traj["advantage"].std(ddof=0)) if len(traj) else 0.0,
    )
