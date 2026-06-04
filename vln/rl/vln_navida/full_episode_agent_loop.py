"""Full-episode rollout core (016 §8.3).

run_episode() drives ONE Habitat trajectory to terminal/cap, recording every NaVIDA
decision. The only backend-specific piece is `decide(instruction, rgb_history)`:
- standalone (13.4): build b64 messages -> vLLM OpenAI API -> action_text.
- verl (13.5+): build markers -> apply_chat_template -> server_manager.generate ->
  also fills prompt_ids/response_ids/response_mask/images for training.
So the env loop, parsing, frame accumulation and reward are written once and shared.
"""
from dataclasses import dataclass, field
from typing import Any, Callable, Awaitable, Optional

from vln.rl.vln_navida.action import parse_navida_action, to_atomic_chunk
from vln.rl.vln_navida.reward import compute_trajectory_reward

# eval_vllm.sh runs --max-action-history 200 (NOT the argparse default 10): the buffer
# stays effectively uncapped so uniform_sample_with_ends(buffer[:-1], 8) spreads the 8
# history frames across the WHOLE episode (global view). Capping at 10 gives only a recent
# local view -> the model loses long-range context and never recognizes goal -> never stops.
MAX_ACTION_HISTORY = 200  # eval default (eval_vllm.sh)


@dataclass
class DecisionGen:
    """What a backend returns for one decision. Training fields are None in rollout-only."""
    action_text: str
    prompt_ids: Optional[list[int]] = None
    response_ids: Optional[list[int]] = None
    response_logprobs: Optional[list[float]] = None
    images: Optional[list[Any]] = None
    mm_processor_kwargs: Optional[dict] = None
    selected_frame_ids: Optional[list[int]] = None


@dataclass
class DecisionRecord:
    turn_id: int
    action_text: str
    parsed_actions: list
    atomic_chunk: list[int]
    env_step_before: int
    env_step_after: int
    is_stop_action: bool
    gen: DecisionGen


@dataclass
class TrajectoryRecord:
    group_uid: str
    trajectory_uid: str
    scene_id: str
    episode_id: str
    instruction: str
    reward: float
    metrics: dict
    decisions: list[DecisionRecord] = field(default_factory=list)


async def run_episode(
    env,
    extra_info: dict,
    decide: Callable[[str, list], Awaitable[DecisionGen]],
    *,
    group_uid: str,
    trajectory_uid: str,
    max_decisions: int = 64,
    max_env_steps: int = 100,
    max_action_history: int = MAX_ACTION_HISTORY,
    progress_coef: float = 0.0,
) -> TrajectoryRecord:
    # Frame buffer holds the raw env_server JPEG b64 strings, forwarded to the
    # model with NO decode/re-encode (avoids double-JPEG; byte-identical to eval).
    await env.reset(extra_info)
    frame_buffer = [env.current_jpeg_b64()]
    decisions: list[DecisionRecord] = []
    env_steps = 0
    done = False

    while not done and len(decisions) < max_decisions and env_steps < max_env_steps:
        gen = await decide(env.instruction, frame_buffer)
        parsed = parse_navida_action(gen.action_text)
        chunk = to_atomic_chunk(parsed)
        step_before = env_steps
        for atomic in chunk:
            s = await env.step(atomic)
            frame_buffer.append(env.current_jpeg_b64())
            if len(frame_buffer) > max_action_history:
                frame_buffer = frame_buffer[1:]
            env_steps += 1
            done = bool(s["done"])
            if done:
                break
        decisions.append(DecisionRecord(
            turn_id=len(decisions),
            action_text=gen.action_text,
            parsed_actions=parsed,
            atomic_chunk=chunk,
            env_step_before=step_before,
            env_step_after=env_steps,
            is_stop_action=(0 in chunk),
            gen=gen,
        ))
        if not chunk:  # model emitted no valid action -> stop the episode (record invalid)
            break

    metrics = env.metrics()
    reward = compute_trajectory_reward(metrics, progress_coef=progress_coef)
    return TrajectoryRecord(
        group_uid=group_uid,
        trajectory_uid=trajectory_uid,
        scene_id=str(extra_info.get("scene_id", "")),
        episode_id=str(extra_info["episode_id"]),
        instruction=env.instruction,
        reward=reward,
        metrics={**metrics, "num_decisions": len(decisions), "env_steps": env_steps},
        decisions=decisions,
    )
