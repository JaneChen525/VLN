import hashlib
import random
from dataclasses import dataclass, field
from typing import Optional

import httpx

from vln.rl.messages import PROMPT_TEMPLATE, SYSTEM_PROMPT, expand_to_atomic, parse_actions
from vln.rl.rollout_client.config import RolloutConfig
from vln.rl.rollout_client.selectors import FrameSelector


@dataclass
class TurnRecord:
    turn_idx: int
    selected_frame_ids: list[int]
    image_manifest: list[dict]
    prompt_text_rendered: str
    response_text: str
    parsed_actions: list
    parse_ok: bool


@dataclass
class AgentState:
    jpeg_b64_buffer: list[str] = field(default_factory=list)
    pending_actions: list[int] = field(default_factory=list)
    turns: list[TurnRecord] = field(default_factory=list)
    turn_idx: int = 0
    parse_fail_count: int = 0


def _build_messages(jpeg_b64_list: list[str], current_jpeg_b64: str, instruction: str) -> tuple[list, str]:
    """Return (openai_messages, prompt_text_rendered).

    NaVIDA prompt: text intro + N history images + text + 1 current image + text tail.
    Images are forwarded as the JPEG b64 strings the env server returned —
    no decode + re-encode — so the model sees byte-identical bytes vs a direct
    habitat pipeline that JPEG-encodes obs["rgb"] once.
    """
    intro = "Imagine you are a robot programmed for navigation tasks. You have been given a video of historical observations"
    bridge = "and an image of the current observation"
    tail = PROMPT_TEMPLATE.format(instruction).split("current observation")[1]

    content = [{"type": "text", "text": intro}]
    content.extend(
        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b}"}} for b in jpeg_b64_list
    )
    content.append({"type": "text", "text": bridge})
    content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{current_jpeg_b64}"}})
    content.append({"type": "text", "text": tail})

    messages = [
        {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
        {"role": "user", "content": content},
    ]
    rendered = f"{intro}\n[<{len(jpeg_b64_list)} history images>]\n{bridge}\n[<current image>]\n{tail}"
    return messages, rendered


class AsyncRolloutAgent:
    """Async vLLM-driven agent. Stores JPEG b64 strings as frame buffer (no decode)."""

    def __init__(
        self,
        vllm_client: httpx.AsyncClient,
        model_name: str,
        cfg: RolloutConfig,
        selector: FrameSelector,
        rng: Optional[random.Random] = None,
    ):
        self.vllm = vllm_client
        self.model = model_name
        self.cfg = cfg
        self.selector = selector
        self.rng = rng or random.Random()

    def new_state(self) -> AgentState:
        return AgentState()

    async def _generate(self, messages: list) -> str:
        body = {
            "model": self.model,
            "messages": messages,
            "max_completion_tokens": self.cfg.max_tokens,
            "temperature": self.cfg.temperature,
            "top_p": self.cfg.top_p,
        }
        r = await self.vllm.post("/chat/completions", json=body)
        r.raise_for_status()
        return r.json()["choices"][0]["message"]["content"].strip()

    async def act(self, state: AgentState, public_obs: dict) -> int:
        state.jpeg_b64_buffer.append(public_obs["rgb_jpeg_b64"])

        if state.pending_actions:
            return state.pending_actions.pop(0)

        idxs = self.selector.select_indices(len(state.jpeg_b64_buffer), state.turn_idx)
        history_b64 = [state.jpeg_b64_buffer[i] for i in idxs]
        current_b64 = state.jpeg_b64_buffer[-1]
        current_idx = len(state.jpeg_b64_buffer) - 1

        messages, rendered = _build_messages(history_b64, current_b64, public_obs["instruction"])
        response = await self._generate(messages)
        parsed = parse_actions(
            response,
            self.cfg.forward_distance,
            self.cfg.turn_angle,
            n=self.cfg.action_packing.actions_per_turn,
        )
        parse_ok = any(a is not None for a, _ in parsed)
        if not parse_ok:
            state.parse_fail_count += 1

        manifest = [
            {"idx": i, "sha256": hashlib.sha256(state.jpeg_b64_buffer[i].encode()).hexdigest()[:16],
             "bytes": (len(state.jpeg_b64_buffer[i]) * 3) // 4}
            for i in idxs + [current_idx]
        ]
        state.turns.append(
            TurnRecord(
                turn_idx=state.turn_idx,
                selected_frame_ids=idxs + [current_idx],
                image_manifest=manifest,
                prompt_text_rendered=rendered,
                response_text=response,
                parsed_actions=[[a, n] for (a, n) in parsed],
                parse_ok=parse_ok,
            )
        )
        state.turn_idx += 1

        atoms: list[int] = []
        for action_id, numeric in parsed:
            if action_id is None:
                action_id = self.rng.randint(1, 3)
                numeric = self.cfg.forward_distance if action_id == 1 else self.cfg.turn_angle
            atoms.extend(
                expand_to_atomic(action_id, numeric, self.cfg.forward_distance, self.cfg.turn_angle)
            )
        if not atoms:
            atoms.append(self.rng.randint(1, 3))
        state.pending_actions.extend(atoms)
        return state.pending_actions.pop(0)
