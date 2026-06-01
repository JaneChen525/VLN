"""HabitatAgentLoop: verl-native agent loop driving the habitat env_server.

13.2 skeleton = single LLM turn. run() resets one episode on the env_server,
builds the NaVIDA prompt with the live frame(s), generates once, steps the
parsed atomic actions, and returns an AgentLoopOutput carrying the trajectory
tokens + a (sparse) reward. verl owns rollout/advantage/FSDP/ref/weight-sync.

Multi-turn frame accumulation + real SR reward land in 13.3.
"""
import asyncio
import os
from typing import Any
from uuid import uuid4

import httpx
from PIL import Image

from verl.experimental.agent_loop.agent_loop import AgentLoopBase, AgentLoopOutput, register
from verl.utils.profiler import simple_timer
from verl.utils.rollout_trace import rollout_trace_op
from verl.workers.rollout.replica import TokenOutput

from vln.rl.env_server.client import AsyncEnvClient, decode_jpeg_b64
from vln.rl.messages import (
    NAVIDA_BRIDGE,
    NAVIDA_INTRO,
    SYSTEM_PROMPT,
    expand_to_atomic,
    navida_tail,
    parse_actions,
    uniform_sample_with_ends,
)


def build_navida_content(instruction: str, rgb_history: list, k_history: int = 8):
    """Return (verl-format content, ordered images) for the NaVIDA prompt.

    Mirrors messages.build_messages exactly (same text constants + image order
    [*history, current]) but emits verl multimodal markers ({"type": "image"})
    with PIL frames passed separately to apply_chat_template.
    """
    current = rgb_history[-1]
    historic = uniform_sample_with_ends(rgb_history[:-1], k_history) if len(rgb_history) > 1 else [current]
    images = list(historic) + [current]
    content = [{"type": "text", "text": NAVIDA_INTRO}]
    content.extend({"type": "image"} for _ in historic)
    content.append({"type": "text", "text": NAVIDA_BRIDGE})
    content.append({"type": "image"})  # current observation
    content.append({"type": "text", "text": navida_tail(instruction)})
    return content, images


@register("habitat")
class HabitatAgentLoop(AgentLoopBase):
    def __init__(self, *args, env_server_url: str = "http://127.0.0.1:8002", **kwargs):
        # env_server_url comes from habitat_agent.yaml (hydra passes extra yaml
        # fields as __init__ kwargs). VLN_ENV_SERVER_URL env var overrides it.
        # NOTE: do NOT rely on the env var alone — verl's Ray runtime_env uses a
        # fixed env_vars whitelist, so driver-set vars don't reach the workers.
        super().__init__(*args, **kwargs)
        self.prompt_length = self.rollout_config.prompt_length
        self.response_length = self.rollout_config.response_length
        self.env_server_url = os.environ.get("VLN_ENV_SERVER_URL", env_server_url)
        # NaVIDA nav hyperparams (match eval defaults)
        self.forward_distance = 25
        self.turn_angle = 15
        self.actions_per_turn = 2
        self.k_history = 8
        # env_server pool is always smaller than total rollout concurrency, so a
        # /reset can hit 503 "no idle worker" until a slot frees. Bounded wait.
        self.reset_retry_s = 2.0
        self.reset_timeout_s = 600.0

    async def _reset_with_retry(self, env: AsyncEnvClient, episode_id: str, trial_id: int) -> dict:
        deadline = asyncio.get_event_loop().time() + self.reset_timeout_s
        while True:
            try:
                return await env.reset(episode_id, trial_id)
            except httpx.HTTPStatusError as e:
                if e.response.status_code == 503 and asyncio.get_event_loop().time() < deadline:
                    await asyncio.sleep(self.reset_retry_s)
                    continue
                raise

    @rollout_trace_op
    async def run(self, sampling_params: dict[str, Any], **kwargs) -> AgentLoopOutput:
        extra = kwargs.get("extra_info", {}) or {}
        episode_id = str(extra.get("episode_id", kwargs.get("episode_id", "")))
        trial_id = int(extra.get("trial_id", 0))

        env = AsyncEnvClient(self.env_server_url)
        metrics: dict = {}
        success = 0.0
        try:
            reset = await self._reset_with_retry(env, episode_id, trial_id)
            env_id = reset["env_id"]
            instruction = reset["obs"]["instruction"]
            frame = Image.fromarray(decode_jpeg_b64(reset["obs"]["rgb_jpeg_b64"]))
            frames = [frame]  # single-turn skeleton: only the current observation

            content, images = build_navida_content(instruction, frames, self.k_history)
            messages = [
                {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
                {"role": "user", "content": content},
            ]
            prompt_ids = await self.apply_chat_template(messages, images=images)

            with simple_timer("generate_sequences", metrics):
                output: TokenOutput = await self.server_manager.generate(
                    request_id=uuid4().hex,
                    prompt_ids=prompt_ids,
                    sampling_params=sampling_params,
                    image_data=images,
                )
            response_ids = output.token_ids
            text = self.tokenizer.decode(response_ids)

            # Drive the env with the parsed atomic actions; reward = env success flag.
            for action_id, numeric in parse_actions(text, self.forward_distance, self.turn_angle, self.actions_per_turn):
                if action_id is None:
                    continue
                done = False
                for atomic in expand_to_atomic(action_id, numeric, self.forward_distance, self.turn_angle):
                    step = await env.step(env_id, atomic)
                    success = step["success"]
                    done = step["done"]
                    if done:
                        break
                if done:
                    break
            await env.delete_env(env_id)
        finally:
            await env.aclose()

        response_mask = [1] * len(response_ids)
        return AgentLoopOutput(
            prompt_ids=prompt_ids,
            response_ids=response_ids[: self.response_length],
            response_mask=response_mask[: self.response_length],
            response_logprobs=output.log_probs[: self.response_length] if output.log_probs else None,
            multi_modal_data={"images": images},
            mm_processor_kwargs=self._get_mm_processor_kwargs(None),
            reward_score=float(success),
            num_turns=2,
            metrics=metrics,
            extra_fields={},
        )
