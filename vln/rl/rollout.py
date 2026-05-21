import json
import os
import random
import time
from dataclasses import dataclass, field
from typing import List, Optional

import cv2
import numpy as np
from PIL import Image
from openai import OpenAI

from vln.rl.messages import (
    SYSTEM_PROMPT,
    build_messages,
    expand_to_atomic,
    parse_actions,
)


@dataclass
class TurnRecord:
    turn_index: int
    messages: list
    response_text: str
    parsed_actions: list


@dataclass
class TrajectoryRecord:
    episode_id: str
    scene_id: str
    trial_id: int
    group_id: str  # = f"{scene_id}|{episode_id}"
    instruction: str
    turns: List[TurnRecord] = field(default_factory=list)
    success: float = 0.0
    spl: float = 0.0
    oracle_success: float = 0.0
    ne: float = 0.0
    env_steps: int = 0
    duration_s: float = 0.0


class HabitatRolloutAgent:
    """NaVIDA-style rollout agent that records (messages, response) per model call."""

    def __init__(
        self,
        api_key: str,
        base_url: str,
        forward_distance: int = 25,
        turn_angle: int = 15,
        resolution_ratio: float = 1.0,
        temperature: float = 0.7,
        top_p: float = 0.95,
        max_tokens: int = 512,
    ):
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.model = self.client.models.list().data[0].id
        self.forward_distance = forward_distance
        self.turn_angle = turn_angle
        self.resolution_ratio = resolution_ratio
        self.temperature = temperature
        self.top_p = top_p
        self.max_tokens = max_tokens

        self.rgb_history: List[Image.Image] = []
        self.pending_actions: List[int] = []
        self.turns: List[TurnRecord] = []
        self.turn_index = 0

    def reset(self):
        self.rgb_history = []
        self.pending_actions = []
        self.turns = []
        self.turn_index = 0

    def _generate(self, messages: list) -> str:
        resp = self.client.chat.completions.create(
            messages=messages,
            model=self.model,
            max_completion_tokens=self.max_tokens,
            temperature=self.temperature,
            top_p=self.top_p,
        )
        return resp.choices[0].message.content.strip()

    def act(self, observations, instruction: str) -> int:
        rgb = observations["rgb"]
        if self.resolution_ratio < 1:
            rgb = cv2.resize(rgb, (0, 0), fx=self.resolution_ratio, fy=self.resolution_ratio)
        self.rgb_history.append(Image.fromarray(rgb.astype("uint8")).convert("RGB"))

        if self.pending_actions:
            return self.pending_actions.pop(0)

        messages = build_messages(instruction, self.rgb_history)
        response = self._generate(messages)
        parsed = parse_actions(response, self.forward_distance, self.turn_angle)

        self.turns.append(
            TurnRecord(
                turn_index=self.turn_index,
                messages=messages,
                response_text=response,
                parsed_actions=[[a, n] for (a, n) in parsed],
            )
        )
        self.turn_index += 1

        for action_id, numeric in parsed:
            if action_id is None:
                action_id = random.randint(1, 3)
                numeric = self.forward_distance if action_id == 1 else self.turn_angle
            self.pending_actions.extend(
                expand_to_atomic(action_id, numeric, self.forward_distance, self.turn_angle)
            )

        if not self.pending_actions:
            self.pending_actions.append(random.randint(1, 3))

        return self.pending_actions.pop(0)


def run_episode(
    env,
    agent: HabitatRolloutAgent,
    target_ep,
    trial_id: int,
    early_stop_rotation: int = 25,
    early_stop_steps: int = 400,
) -> TrajectoryRecord:
    env.current_episode = target_ep
    obs = env.reset()
    agent.reset()

    instruction = obs["instruction"]["text"]
    episode_id = str(env.current_episode.episode_id)
    scene_id = os.path.basename(env.current_episode.scene_id).split(".")[0]

    rotation_count = 0
    last_dtg = 999.0
    iter_step = 0
    t0 = time.time()

    while not env.episode_over:
        info = env.get_metrics()
        cur_dtg = info["distance_to_goal"]
        if cur_dtg != last_dtg:
            last_dtg = cur_dtg
            rotation_count = 0
        else:
            rotation_count += 1

        if rotation_count > early_stop_rotation or iter_step > early_stop_steps:
            action = 0  # stop
        else:
            action = agent.act(obs, instruction)

        obs = env.step({"action": action})
        iter_step += 1

    info = env.get_metrics()
    return TrajectoryRecord(
        episode_id=episode_id,
        scene_id=scene_id,
        trial_id=trial_id,
        group_id=f"{scene_id}|{episode_id}",
        instruction=instruction,
        turns=agent.turns,
        success=float(info["success"]),
        spl=float(info["spl"]),
        oracle_success=float(info["oracle_success"]),
        ne=float(info["distance_to_goal"]),
        env_steps=iter_step,
        duration_s=time.time() - t0,
    )


def trajectory_to_rows(traj: TrajectoryRecord) -> list:
    # One row per model call (turn). Reward = terminal SR broadcast.
    rows = []
    for t in traj.turns:
        rows.append(
            {
                "episode_id": traj.episode_id,
                "scene_id": traj.scene_id,
                "trial_id": traj.trial_id,
                "group_id": traj.group_id,
                "turn_index": t.turn_index,
                "instruction": traj.instruction,
                "messages": json.dumps(t.messages),
                "response": t.response_text,
                "parsed_actions": json.dumps(t.parsed_actions),
                "reward": traj.success,
                "success": traj.success,
                "spl": traj.spl,
                "oracle_success": traj.oracle_success,
                "ne": traj.ne,
                "env_steps": traj.env_steps,
                "duration_s": traj.duration_s,
            }
        )
    return rows
