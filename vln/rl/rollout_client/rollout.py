import hashlib
import json
import time
from dataclasses import dataclass, field

from vln.rl.env_server.client import AsyncEnvClient
from vln.rl.messages import PROMPT_TEMPLATE
from vln.rl.rollout_client.agent import AgentState, AsyncRolloutAgent, TurnRecord
from vln.rl.rollout_client.config import RolloutConfig, config_hash

SCHEMA_VERSION = "v1"
PROMPT_TEMPLATE_HASH = hashlib.sha256(PROMPT_TEMPLATE.encode()).hexdigest()[:16]


@dataclass
class TrajectoryRecord:
    episode_id: str
    trial_id: int
    scene_id: str
    instruction: str
    turns: list[TurnRecord] = field(default_factory=list)
    success: float = 0.0
    spl: float = 0.0
    oracle_success: float = 0.0
    ne: float = 0.0
    env_steps: int = 0
    duration_s: float = 0.0
    vllm_s: float = 0.0
    env_s: float = 0.0
    done_reason: str = ""
    aborted: bool = False


def trajectory_seed(base_seed: int, episode_id: str, trial_id: int) -> int:
    # Deterministic per-trajectory seed, independent of concurrent completion order.
    h = int(hashlib.sha256(f"{base_seed}:{episode_id}:{trial_id}".encode()).hexdigest()[:8], 16)
    return h


async def run_episode(
    env: AsyncEnvClient,
    agent: AsyncRolloutAgent,
    episode_id: str,
    trial_id: int,
    cfg: RolloutConfig,
    seed: int,
) -> TrajectoryRecord:
    t0 = time.time()
    r = await env.reset(episode_id, trial_id)
    env_id = r["env_id"]
    state = agent.new_state(seed)
    public_obs = r["obs"]
    last_obs = {"success": 0.0, "spl": 0.0, "oracle_success": 0.0, "ne": 0.0, "step_count": 0}
    instruction = public_obs["instruction"]
    done = False
    done_reason = ""
    rotation_count = 0
    last_ne = 999.0
    env_s = 0.0

    try:
        while not done:
            cur_ne = last_obs.get("ne", 999.0)
            if cur_ne != last_ne:
                last_ne = cur_ne
                rotation_count = 0
            else:
                rotation_count += 1

            if rotation_count > cfg.early_stop_rotation:
                action = 0
                done_reason = "early_stop_rotation"
            elif last_obs["step_count"] >= cfg.early_stop_steps:
                action = 0
                done_reason = "early_stop_steps"
            else:
                action = await agent.act(state, public_obs)

            _t = time.perf_counter()
            step = await env.step(env_id, action)
            env_s += time.perf_counter() - _t
            last_obs = {
                "success": step["success"], "spl": step["spl"],
                "oracle_success": step["oracle_success"], "ne": step["ne"],
                "step_count": step["step_count"],
            }
            public_obs = step["obs"]
            done = step["done"]
            if done and not done_reason:
                done_reason = "stop_action" if action == 0 else "env_done"
    finally:
        await env.delete_env(env_id)

    return TrajectoryRecord(
        episode_id=episode_id,
        trial_id=trial_id,
        scene_id=r["scene_id"],
        instruction=instruction,
        turns=state.turns,
        success=float(last_obs["success"]),
        spl=float(last_obs["spl"]),
        oracle_success=float(last_obs["oracle_success"]),
        ne=float(last_obs["ne"]),
        env_steps=int(last_obs["step_count"]),
        duration_s=time.time() - t0,
        vllm_s=state.vllm_s,
        env_s=env_s,
        done_reason=done_reason,
    )


def trajectory_to_rows(
    traj: TrajectoryRecord,
    cfg: RolloutConfig,
    *,
    split: str,
    policy_version: str,
    rl_step: int = 0,
) -> list[dict]:
    cfg_hash = config_hash(cfg)
    variant_hash = cfg_hash[:8]
    group_id = f"{split}:{traj.episode_id}:{variant_hash}"
    trajectory_id = f"{group_id}|trial_{traj.trial_id}"

    rows = []
    for t in traj.turns:
        rows.append({
            "schema_version": SCHEMA_VERSION,
            "rl_step": rl_step,
            "policy_version": policy_version,
            "group_id": group_id,
            "trajectory_id": trajectory_id,
            "episode_id": traj.episode_id,
            "trial_id": traj.trial_id,
            "scene_id": traj.scene_id,
            "turn_idx": t.turn_idx,
            "variant_hash": variant_hash,
            "prompt_template_hash": PROMPT_TEMPLATE_HASH,
            "rollout_sampling_config_hash": cfg_hash,
            "instruction": traj.instruction,
            "prompt_text_rendered": t.prompt_text_rendered,
            "response_text": t.response_text,
            "parsed_actions": json.dumps(t.parsed_actions),
            "parse_ok": t.parse_ok,
            "selected_frame_ids": json.dumps(t.selected_frame_ids),
            "image_manifest": json.dumps(t.image_manifest),
            "reward_success": traj.success,
            "success": traj.success,
            "spl": traj.spl,
            "oracle_success": traj.oracle_success,
            "ne": traj.ne,
            "env_steps": traj.env_steps,
            "duration_s": traj.duration_s,
            "done_reason": traj.done_reason,
            "aborted": traj.aborted,
        })
    return rows
