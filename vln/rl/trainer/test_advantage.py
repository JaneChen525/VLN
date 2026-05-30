"""Pure unit tests for GRPO grouping/advantage. Run: pytest -q (no verl/GPU)."""
import numpy as np
import pandas as pd
import pytest

from vln.rl.trainer.advantage import (
    MixedPolicyError,
    assert_single_policy_version,
    compute_grpo_advantages,
)


def _rows(spec, policy_version="p0"):
    """spec: list of (group_id, trial, reward, n_turns) -> turn-level rows."""
    out = []
    for gid, trial, reward, n_turns in spec:
        tid = f"{gid}|trial_{trial}"
        for t in range(n_turns):
            out.append({
                "group_id": gid, "trajectory_id": tid, "trial_id": trial,
                "turn_idx": t, "reward_success": reward, "policy_version": policy_version,
            })
    return pd.DataFrame(out)


def test_grpo_grouping():
    # one group, rewards [1,0,0,1] -> mean .5, std(ddof1)=.577; winners +, losers -
    df = _rows([("g", 0, 1.0, 2), ("g", 1, 0.0, 2), ("g", 2, 0.0, 2), ("g", 3, 1.0, 2)])
    out, stats = compute_grpo_advantages(df)
    adv = out.drop_duplicates("trajectory_id").set_index("trial_id")["advantage"]
    assert adv[0] > 0 and adv[3] > 0
    assert adv[1] < 0 and adv[2] < 0
    assert adv[0] == pytest.approx(adv[3])      # symmetric winners
    assert adv[1] == pytest.approx(adv[2])      # symmetric losers
    assert stats.n_groups == 1 and stats.mixed_ratio == 1.0


def test_all_zero_group():
    df = _rows([("g", k, 0.0, 3) for k in range(4)])
    out, stats = compute_grpo_advantages(df)
    assert (out["advantage"].abs() < 1e-9).all()      # all-zero -> zero adv
    assert not out["advantage"].isna().any()
    assert stats.all_zero_ratio == 1.0 and stats.mixed_ratio == 0.0
    assert stats.effective_trajectories == 0


def test_all_one_group():
    df = _rows([("g", k, 1.0, 2) for k in range(4)])
    out, stats = compute_grpo_advantages(df)
    assert (out["advantage"].abs() < 1e-9).all()
    assert stats.all_one_ratio == 1.0


def test_trajectory_advantage_broadcast():
    # 8-turn trajectory: every turn row carries the same trajectory advantage.
    df = _rows([("g", 0, 1.0, 8), ("g", 1, 0.0, 4)])
    out, _ = compute_grpo_advantages(df)
    t0 = out[out["trajectory_id"] == "g|trial_0"]["advantage"]
    assert t0.nunique() == 1 and len(t0) == 8 and t0.iloc[0] > 0


def test_policy_version_assert():
    df = _rows([("g", 0, 1.0, 2)], policy_version="p0")
    df2 = _rows([("g", 1, 0.0, 2)], policy_version="p1")
    mixed = pd.concat([df, df2], ignore_index=True)
    with pytest.raises(MixedPolicyError):
        assert_single_policy_version(mixed)
    assert assert_single_policy_version(df) == "p0"


def test_two_groups_independent():
    df = _rows([("ga", 0, 1.0, 2), ("ga", 1, 0.0, 2),
                ("gb", 0, 1.0, 2), ("gb", 1, 1.0, 2)])  # gb all-one
    out, stats = compute_grpo_advantages(df)
    assert stats.n_groups == 2
    gb = out[out["group_id"] == "gb"]["advantage"]
    assert (gb.abs() < 1e-9).all()                     # gb degenerate -> 0
    ga = out[out["group_id"] == "ga"]["advantage"]
    assert ga.abs().max() > 0                          # ga has signal
