"""
Task 14 — Episode difficulty evaluation for RL data engineering.

Runs pass@K rollouts on each episode (train split) to measure per-episode
success rate. Episodes with pass_rate=0 or pass_rate=1 provide zero GRPO
gradient signal; the "boundary" episodes (0 < pass_rate < 1) are what
matters for RL training.

Based on eval_vllm_navida.py — identical Agent logic, with additions:
  --temperature   sampling temperature (default 0.6 for diversity)
  --max-episode   cap number of episodes (random subset, seed=42)
  --output-csv    per-episode difficulty CSV path

Usage (on cluster, after vLLM is serving):
  OPENAI_API_KEY=EMPTY OPENAI_API_BASE=http://localhost:8001/v1 \
  python -u vln/difficulty_eval.py \
    --exp-config config/vln_r2r.yaml --split-num 24 \
    --pass-k 8 --temperature 0.6 --max-episode 200 \
    --result-path results/difficulty \
    --output-csv results/difficulty/episode_difficulty.csv
"""

import json
import csv
import numpy as np
from habitat import Env
from habitat.core.agent import Agent
from tqdm import tqdm
import os
import re
import cv2
import imageio
from habitat.utils.visualizations import maps
import random
import argparse
import habitat
from habitat_extensions import measures, task
from habitat.config.default import get_config
from habitat.config.default_structured_configs import (
    CollisionsMeasurementConfig,
    FogOfWarConfig,
    TopDownMapMeasurementConfig,
)
from PIL import Image, ImageFont, ImageDraw
import multiprocessing as mp
import time
import math
from openai import OpenAI
import base64
from io import BytesIO


SYSTEM_PROMPT = "You are a helpful assistant."


def encode_image_base64(image):
    buffer = BytesIO()
    image.save(buffer, format="JPEG")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


def seed_all(seed=42):
    np.random.seed(seed)
    random.seed(seed)


def canonical_episode_key(scene_id, episode_id):
    """Return the stable key shared by Habitat episodes and manifest rows."""
    scene = os.path.splitext(os.path.basename(str(scene_id)))[0]
    return scene, str(episode_id)


def load_manifest_episode_metadata(manifest_path):
    """Load exact episode membership and prior pass rate from a VLN parquet."""
    import pyarrow.parquet as pq

    rows = pq.read_table(manifest_path, columns=["extra_info"]).to_pylist()
    metadata = {}
    for row in rows:
        info = row["extra_info"]
        key = canonical_episode_key(info["scene_id"], info["episode_id"])
        if key in metadata:
            raise ValueError(f"Duplicate episode in manifest: {key}")
        metadata[key] = {
            "manifest_pass_rate": info.get("pass_rate"),
            "manifest_instruction": info.get("instruction"),
        }
    return metadata


def filter_dataset_by_manifest(dataset, manifest_metadata):
    """Filter and order Habitat episodes exactly as the manifest."""
    by_key = {
        canonical_episode_key(episode.scene_id, episode.episode_id): episode
        for episode in dataset.episodes
    }
    missing = [key for key in manifest_metadata if key not in by_key]
    if missing:
        raise ValueError(
            f"Manifest has {len(missing)} episodes absent from Habitat data; "
            f"first missing keys: {missing[:5]}"
        )
    dataset.episodes = [by_key[key] for key in manifest_metadata]
    return dataset


def load_done_result_keys(result_path, require_trajectory=False):
    result_file = os.path.join(result_path, "result.json")
    done = set()
    if not os.path.exists(result_file):
        return done
    with open(result_file, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                continue
            if "scene_id" in item and "episode_id" in item and "episode_instruction" in item:
                if require_trajectory:
                    base_map = item.get("trajectory_base_map")
                    if (
                        not item.get("trajectory_points")
                        or not base_map
                        or not os.path.isfile(os.path.join(result_path, base_map))
                    ):
                        continue
                trial_id = int(item.get("trial_id", 0))
                trial_total = int(item.get("trial_total", 1))
                done.add(
                    (
                        item["scene_id"],
                        str(item["episode_id"]),
                        item["episode_instruction"],
                        trial_id,
                        trial_total,
                    )
                )
    return done


def trajectory_episode_name(scene_id, episode_id):
    scene = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(scene_id))
    episode = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(episode_id))
    return f"{scene}_ep{episode}"


def save_trajectory_base_map(result_path, scene_id, episode_id, top_down_map):
    trajectory_dir = os.path.join(result_path, "trajectories")
    os.makedirs(trajectory_dir, exist_ok=True)
    filename = f".{trajectory_episode_name(scene_id, episode_id)}_base.png"
    output_path = os.path.join(trajectory_dir, filename)
    if not os.path.isfile(output_path):
        imageio.imwrite(output_path, maps.colorize_topdown_map(top_down_map))
    return os.path.relpath(output_path, result_path)


def get_agent_map_point(top_down_map_info):
    """Return the first agent coordinate as OpenCV (x, y)."""
    coordinates = top_down_map_info.get("agent_map_coord") or []
    if not coordinates:
        return None
    row, column = coordinates[0]
    return [int(column), int(row)]


def evaluate_agent(result_queue, api_key, base_url, config, dataset, result_path,
                   num_generations, forward_distance, turn_angle, max_action_history,
                   resolution_ratio, save_video, pass_k, temperature,
                   render_gpu_id=0, manifest_metadata=None,
                   save_trajectory_map=False):

    if render_gpu_id >= 0:
        from omegaconf import read_write
        with read_write(config):
            config.habitat.simulator.habitat_sim_v0.gpu_device_id = render_gpu_id
    env = Env(config.habitat, dataset)

    agent = NaVIDA_Agent(
        api_key,
        base_url,
        result_path,
        forward_distance,
        turn_angle,
        max_action_history,
        resolution_ratio,
        num_generations,
        save_video=save_video,
        temperature=temperature,
    )

    num_episodes = len(env.episodes)
    done_result_keys = load_done_result_keys(
        result_path, require_trajectory=save_trajectory_map
    )
    episodes_snapshot = list(env.episodes)

    EARLY_STOP_ROTATION = 25
    EARLY_STOP_STEPS = 200

    for target_ep in episodes_snapshot:
        scene_id = target_ep.scene_id.split('/')[-2]
        episode_id = target_ep.episode_id
        episode_instruction = target_ep.instruction.instruction_text
        for trial_id in range(pass_k):
            episode_start_time = time.time()

            if (scene_id, str(episode_id), episode_instruction, trial_id, pass_k) in done_result_keys:
                t_dict = {"t_episode": 0, "skipped": 1}
                result_queue.put(t_dict)
                continue

            env.current_episode = target_ep
            obs = env.reset()
            iter_step = 0
            agent.reset()

            t_dict = {"t_episode": 0}

            continuse_rotation_count = 0
            last_dtg = 999
            trajectory_points = []
            trajectory_base_map = None
            while not env.episode_over:
                info = env.get_metrics()

                if save_trajectory_map:
                    if trajectory_base_map is None:
                        trajectory_base_map = save_trajectory_base_map(
                            result_path,
                            scene_id,
                            episode_id,
                            info["top_down_map"]["map"],
                        )
                    point = get_agent_map_point(info["top_down_map"])
                    if point is not None:
                        if not trajectory_points or point != trajectory_points[-1]:
                            trajectory_points.append(point)

                if info["distance_to_goal"] != last_dtg:
                    last_dtg = info["distance_to_goal"]
                    continuse_rotation_count = 0
                else:
                    continuse_rotation_count += 1

                action = agent.act(obs, info, env.current_episode.episode_id)

                if continuse_rotation_count > EARLY_STOP_ROTATION or iter_step > EARLY_STOP_STEPS:
                    action = {"action": 0}

                iter_step += 1
                obs = env.step(action)

            info = env.get_metrics()
            if save_trajectory_map:
                point = get_agent_map_point(info["top_down_map"])
                if point is not None:
                    if not trajectory_points or point != trajectory_points[-1]:
                        trajectory_points.append(point)
            result = {
                "scene_id": scene_id,
                "episode_id": int(episode_id) if str(episode_id).isdigit() else episode_id,
                "trial_id": trial_id,
                "trial_total": pass_k,
                "success": info["success"],
                "spl": info["spl"],
                "os": info["oracle_success"],
                "ne": info["distance_to_goal"],
                "steps": iter_step,
                "episode_instruction": episode_instruction,
            }
            metadata = (manifest_metadata or {}).get(
                canonical_episode_key(scene_id, episode_id), {}
            )
            result.update(metadata)
            if save_trajectory_map:
                result["trajectory_points"] = trajectory_points
                result["trajectory_base_map"] = trajectory_base_map
            with open(os.path.join(result_path, "result.json"), "a") as f:
                f.write(json.dumps(result) + "\n")
            done_result_keys.add((scene_id, str(episode_id), episode_instruction, trial_id, pass_k))

            t_dict["t_episode"] = time.time() - episode_start_time
            result_queue.put(t_dict)


# ---------------------------------------------------------------------------
# NaVIDA_Agent — identical to eval_vllm_navida.py, only temperature is
# parameterized (constructor kwarg instead of hardcoded 0.3).
# ---------------------------------------------------------------------------

class NaVIDA_Agent(Agent):
    def __init__(self, api_key, base_url, result_path, forward_distance,
                 turn_angle, max_action_history, resolution_ratio, num_generations=1,
                 save_video=False, temperature=0.6):

        print("Initialize NaVIDA (difficulty eval)")

        self.result_path = result_path
        self.save_video = save_video
        self.forward_distance = forward_distance
        self.turn_angle = turn_angle
        self.resolution_ratio = resolution_ratio
        self.max_action_history = max_action_history
        self.num_generations = num_generations
        os.makedirs(self.result_path, exist_ok=True)
        if self.save_video:
            os.makedirs(os.path.join(self.result_path, "video"), exist_ok=True)

        self.client = OpenAI(
            api_key=api_key,
            base_url=base_url,
        )
        self.model = self.client.models.list().data[0].id

        self.temperature = temperature
        self.top_p = 0.95
        self.max_tokens = 512

        self.promt_template = (
            "Imagine you are a robot programmed for navigation tasks. "
            "You have been given a video of historical observations and an image of the current observation. "
            "Your assigned task is: '{}'. Analyze this series of images to decide your next move, "
            "which could involve turning left or right by a specific degree or moving forward a certain distance."
        )
        self.history_rgb_tensor = None

        self.rgb_list = []
        self.topdown_map_list = []
        self.conversations = []
        self.conversations.append({
            "role": "system",
            "content": [{"type": "text", "text": SYSTEM_PROMPT}]
        })

        self.reset()

    def uniform_sample_with_ends(self, data, n):
        if len(data) <= n:
            return data
        indices = [round(i * (len(data) - 1) / (n - 1)) for i in range(n)]
        return [data[i] for i in indices]

    def predict_inference(self):
        outputs = self.client.chat.completions.create(
            messages=self.conversations,
            model=self.model,
            max_completion_tokens=self.max_tokens,
            temperature=self.temperature,
            top_p=self.top_p,
        )
        output_text = outputs.choices[0].message.content
        output_text = output_text.strip()
        return output_text

    def extract_multi_result(self, output):
        sub_actions = output.split(', ')
        result = []
        for sub_action in sub_actions:
            action_index, numeric = self.extract_result(sub_action)
            result.append([action_index, numeric])
        return result

    def extract_result(self, output):
        output_match = re.search(r'<answer>(.*?)</answer>', output)
        output = output_match.group(1).strip() if output_match else output.strip()

        output = output.lower()
        if "stop" in output:
            return 0, None
        elif "forward" in output:
            match = re.search(r'-?\d+', output)
            if match is None:
                return 1, self.forward_distance
            match = match.group()
            return 1, float(match)
        elif "left" in output:
            match = re.search(r'-?\d+', output)
            if match is None:
                return 2, self.turn_angle
            match = match.group()
            return 2, float(match)
        elif "right" in output:
            match = re.search(r'-?\d+', output)
            if match is None:
                return 3, self.turn_angle
            match = match.group()
            return 3, float(match)
        return None, None

    def addtext(self, image, instuction, navigation):
        h, w = image.shape[:2]
        new_height = h + 150
        new_image = np.zeros((new_height, w, 3), np.uint8)
        new_image.fill(255)
        new_image[:h, :w] = image

        font = cv2.FONT_HERSHEY_SIMPLEX
        textsize = cv2.getTextSize(instuction, font, 0.5, 2)[0]
        textY = h + (50 + textsize[1]) // 2

        y_line = textY + 0 * textsize[1]

        words = instuction.split(' ')
        x = 10
        line = ""

        for word in words:
            test_line = line + ' ' + word if line else word
            test_line_size, _ = cv2.getTextSize(test_line, font, 0.5, 2)

            if test_line_size[0] > image.shape[1] - x:
                cv2.putText(new_image, line, (x, y_line), font, 0.5, (0, 0, 0), 2)
                line = word
                y_line += textsize[1] + 5
            else:
                line = test_line

        if line:
            cv2.putText(new_image, line, (x, y_line), font, 0.5, (0, 0, 0), 2)
        y_line = y_line + 1 * textsize[1] + 10
        new_image = cv2.putText(new_image, navigation, (x, y_line), font, 0.5, (0, 0, 0), 2)

        return new_image

    def action_id_to_str(self, action_id):
        if action_id == 0:
            return "stop"
        elif action_id == 1:
            return "forward"
        elif action_id == 2:
            return "turn left"
        elif action_id == 3:
            return "turn right"
        else:
            raise ValueError(f"Invalid action ID: {action_id}")

    def reset(self):
        if self.save_video:
            if len(self.topdown_map_list) != 0:
                output_video_path = os.path.join(
                    self.result_path, "video", "{}.gif".format(self.episode_id)
                )
                imageio.mimsave(output_video_path, self.topdown_map_list)

        self.topdown_map_list = []
        self.pending_action_list = []
        self.rgb_list = []

        self.conversations = []
        self.conversations.append({
            "role": "system",
            "content": [{"type": "text", "text": SYSTEM_PROMPT}]
        })

    def act(self, observations, info, episode_id):

        self.episode_id = episode_id
        rgb = observations["rgb"]
        if self.resolution_ratio < 1:
            rgb = cv2.resize(rgb, (0, 0), fx=self.resolution_ratio, fy=self.resolution_ratio)
        rgb_ = Image.fromarray(rgb.astype('uint8')).convert('RGB')
        self.rgb_list.append(rgb_)
        if len(self.rgb_list) > self.max_action_history:
            self.rgb_list = self.rgb_list[1:]

        if self.save_video:
            top_down_map = maps.colorize_draw_agent_and_fit_to_height(info["top_down_map"], rgb.shape[0])
            output_im = np.concatenate((rgb, top_down_map), axis=1)

        if len(self.pending_action_list) != 0:
            temp_action = self.pending_action_list.pop(0)

            if self.save_video:
                img = self.addtext(output_im, observations["instruction"]["text"], "Pending action: {}".format(temp_action))
                self.topdown_map_list.append(img)
            return {"action": temp_action}

        # single-turn: clear conversation history, send all history frames + current frame
        self.conversations = self.conversations[:1]
        content = []

        content.append({"type": "text", "text": 'Imagine you are a robot programmed for navigation tasks. You have been given a video of historical observations'})
        if len(self.rgb_list) > 1:
            content.extend([{"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{encode_image_base64(item)}"}} for item in self.uniform_sample_with_ends(self.rgb_list[:-1], 8)])
        else:
            content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{encode_image_base64(self.rgb_list[-1])}"}})
        content.append({"type": "text", "text": 'and an image of the current observation'})
        content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{encode_image_base64(self.rgb_list[-1])}"}})
        item = self.promt_template.format(observations["instruction"]["text"]).split('current observation')
        content.append({"type": "text", "text": item[1]})

        self.conversations.append({
            "role": "user",
            "content": content
        })

        navigation = self.predict_inference()

        if self.save_video:
            img = self.addtext(output_im, observations["instruction"]["text"], navigation)
            self.topdown_map_list.append(img)

        result = self.extract_multi_result(navigation)

        select_action_idx = 2

        result = result[:select_action_idx]
        for action_index, numeric in result:

            if action_index == 0:
                self.pending_action_list.append(0)
            elif action_index == 1:
                for _ in range(min(3, round(numeric / self.forward_distance))):
                    self.pending_action_list.append(1)

            elif action_index == 2:
                for _ in range(min(3, round(numeric / self.turn_angle))):
                    self.pending_action_list.append(2)

            elif action_index == 3:
                for _ in range(min(3, round(numeric / self.turn_angle))):
                    self.pending_action_list.append(3)

            if action_index is None or len(self.pending_action_list) == 0:
                print('random select an action')
                action_index = random.randint(1, 3)
                navigation = self.action_id_to_str(action_index)
                self.pending_action_list.append(action_index)

        return {"action": self.pending_action_list.pop(0)}


# ---------------------------------------------------------------------------
# Difficulty analysis: per-episode pass_rate CSV
# ---------------------------------------------------------------------------

def compute_difficulty_csv(result_path, output_csv, pass_k):
    """Read result.json and compute per-episode pass_rate → CSV."""
    result_file = os.path.join(result_path, "result.json")
    trials_by_key = {}
    with open(result_file, "r") as f:
        for line in f:
            try:
                item = json.loads(line)
            except (json.JSONDecodeError, TypeError):
                continue
            if not all(key in item for key in ("scene_id", "episode_id", "trial_id")):
                continue
            key = (
                *canonical_episode_key(item["scene_id"], item["episode_id"]),
                int(item["trial_id"]),
            )
            trials_by_key[key] = item

    episodes = {}
    for item in trials_by_key.values():
        key = canonical_episode_key(item["scene_id"], item["episode_id"])
        if key not in episodes:
            episodes[key] = {
                "scene_id": item["scene_id"],
                "episode_id": item["episode_id"],
                "manifest_pass_rate": item.get("manifest_pass_rate"),
                "successes": 0,
                "trials": 0,
                "ne_sum": 0.0,
                "spl_sum": 0.0,
                "steps_sum": 0,
            }
        episodes[key]["trials"] += 1
        episodes[key]["successes"] += int(item["success"])
        episodes[key]["ne_sum"] += float(item["ne"])
        episodes[key]["spl_sum"] += float(item["spl"])
        episodes[key]["steps_sum"] += int(item["steps"])

    rows = []
    for key, ep in sorted(episodes.items()):
        k = ep["trials"]
        rows.append({
            "scene_id": ep["scene_id"],
            "episode_id": ep["episode_id"],
            "manifest_pass_rate": ep["manifest_pass_rate"],
            "pass_k": k,
            "success_count": ep["successes"],
            "pass_rate": ep["successes"] / k if k > 0 else 0.0,
            "avg_ne": ep["ne_sum"] / k if k > 0 else 0.0,
            "avg_spl": ep["spl_sum"] / k if k > 0 else 0.0,
            "avg_steps": ep["steps_sum"] / k if k > 0 else 0,
        })

    os.makedirs(os.path.dirname(output_csv) or ".", exist_ok=True)
    with open(output_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "scene_id", "episode_id", "manifest_pass_rate", "pass_k", "success_count",
            "pass_rate", "avg_ne", "avg_spl", "avg_steps",
        ])
        writer.writeheader()
        writer.writerows(rows)

    n_total = len(rows)
    n_easy = sum(1 for r in rows if r["pass_rate"] == 1.0)
    n_hard = sum(1 for r in rows if r["pass_rate"] == 0.0)
    n_boundary = n_total - n_easy - n_hard
    avg_pass_rate = np.mean([r["pass_rate"] for r in rows]) if rows else 0
    print(f"\n{'='*60}")
    print(f"Difficulty Analysis: {output_csv}")
    print(f"{'='*60}")
    print(f"Total episodes:   {n_total}")
    denominator = max(1, n_total)
    print(f"Easy (rate=1.0):  {n_easy} ({100*n_easy/denominator:.1f}%)")
    print(f"Hard (rate=0.0):  {n_hard} ({100*n_hard/denominator:.1f}%)")
    print(f"Boundary (0<r<1): {n_boundary} ({100*n_boundary/denominator:.1f}%)")
    print(f"Avg pass_rate:    {avg_pass_rate:.3f}")
    print(f"{'='*60}\n")

    return rows


def _draw_wrapped_text(image, text, origin, max_width, scale=0.55, thickness=1):
    x, y = origin
    line = ""
    for word in str(text).split():
        candidate = f"{line} {word}".strip()
        width = cv2.getTextSize(candidate, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness)[0][0]
        if line and width > max_width:
            cv2.putText(
                image, line, (x, y), cv2.FONT_HERSHEY_SIMPLEX,
                scale, (25, 25, 25), thickness, cv2.LINE_AA,
            )
            y += 24
            line = word
        else:
            line = candidate
    if line:
        cv2.putText(
            image, line, (x, y), cv2.FONT_HERSHEY_SIMPLEX,
            scale, (25, 25, 25), thickness, cv2.LINE_AA,
        )
    return y


def build_episode_trajectory_maps(result_path, pass_k):
    """Create one PNG per episode with all pass@K paths overlaid."""
    result_file = os.path.join(result_path, "result.json")
    deduped = {}
    with open(result_file) as handle:
        for line in handle:
            try:
                item = json.loads(line)
            except (json.JSONDecodeError, TypeError):
                continue
            if not item.get("trajectory_points") or not item.get("trajectory_base_map"):
                continue
            key = (
                *canonical_episode_key(item.get("scene_id"), item.get("episode_id")),
                int(item.get("trial_id", 0)),
            )
            deduped[key] = item

    episodes = {}
    for item in deduped.values():
        key = canonical_episode_key(item["scene_id"], item["episode_id"])
        episodes.setdefault(key, []).append(item)

    success_colors = [
        (0, 105, 45), (0, 135, 60), (0, 165, 80), (0, 190, 105),
        (25, 150, 120), (40, 175, 140), (55, 195, 155), (70, 215, 170),
    ]
    failure_colors = [
        (170, 0, 25), (195, 20, 35), (220, 35, 45), (235, 55, 35),
        (190, 0, 90), (215, 25, 110), (235, 50, 130), (245, 75, 145),
    ]
    output_dir = os.path.join(result_path, "trajectory_maps")
    os.makedirs(output_dir, exist_ok=True)

    generated = 0
    for (scene, episode), trials in episodes.items():
        trials.sort(key=lambda item: int(item["trial_id"]))
        success_count = sum(int(item["success"]) for item in trials)
        base_path = os.path.join(result_path, trials[0]["trajectory_base_map"])
        base = imageio.imread(base_path)
        if base.ndim == 2:
            base = np.repeat(base[:, :, None], 3, axis=2)
        base = np.ascontiguousarray(base[:, :, :3].copy())
        map_height, map_width = base.shape[:2]
        header_height = 130
        sidebar_width = 430
        canvas = np.full(
            (map_height + header_height, map_width + sidebar_width, 3),
            255, dtype=np.uint8,
        )
        canvas[header_height:, :map_width] = base

        instruction = trials[0].get("episode_instruction", "")
        prior = trials[0].get("manifest_pass_rate")
        prior_text = "n/a" if prior is None else f"{float(prior):.3f}"
        title = (
            f"scene={scene}  episode={episode}  success={success_count}/{pass_k}  "
            f"prior_pass_rate={prior_text}"
        )
        cv2.putText(
            canvas, title, (16, 30), cv2.FONT_HERSHEY_SIMPLEX,
            0.68, (20, 20, 20), 2, cv2.LINE_AA,
        )
        _draw_wrapped_text(canvas, instruction, (16, 60), map_width + sidebar_width - 32)

        for item in trials:
            trial_id = int(item["trial_id"])
            succeeded = bool(item["success"])
            palette = success_colors if succeeded else failure_colors
            color = palette[trial_id % len(palette)]
            points = np.asarray(item["trajectory_points"], dtype=np.int32)
            points[:, 1] += header_height
            cv2.polylines(
                canvas, [points.reshape(-1, 1, 2)], False,
                color, 4, cv2.LINE_AA,
            )
            start = tuple(int(value) for value in points[0])
            end = tuple(int(value) for value in points[-1])
            cv2.circle(canvas, start, 6, (30, 90, 220), -1, cv2.LINE_AA)
            cv2.circle(canvas, end, 7, color, -1, cv2.LINE_AA)
            cv2.putText(
                canvas, f"t{trial_id}", (end[0] + 7, end[1] - 7),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA,
            )

            legend_y = header_height + 34 + trial_id * 58
            legend_x = map_width + 24
            cv2.line(
                canvas, (legend_x, legend_y), (legend_x + 48, legend_y),
                color, 5, cv2.LINE_AA,
            )
            status = "SUCCESS" if succeeded else "FAILURE"
            cv2.putText(
                canvas, f"trial {trial_id}: {status}",
                (legend_x + 62, legend_y + 5), cv2.FONT_HERSHEY_SIMPLEX,
                0.58, color, 2, cv2.LINE_AA,
            )
            cv2.putText(
                canvas,
                f"NE={float(item['ne']):.2f}m  SPL={float(item['spl']):.3f}  steps={int(item['steps'])}",
                (legend_x, legend_y + 27), cv2.FONT_HERSHEY_SIMPLEX,
                0.47, (45, 45, 45), 1, cv2.LINE_AA,
            )

        output_path = os.path.join(
            output_dir, f"{trajectory_episode_name(scene, episode)}_8rollouts.png"
        )
        imageio.imwrite(output_path, canvas)
        generated += 1

    print(f"Trajectory maps: {generated} PNGs in {output_dir}")
    return generated


def main():
    seed_all(42)
    parser = argparse.ArgumentParser(description="Episode difficulty evaluation (pass@K per-episode)")

    parser.add_argument("--exp-config", type=str, required=True)
    parser.add_argument("--split-num", type=int, required=True)
    parser.add_argument("--resolution-ratio", type=float, default=1.0)
    parser.add_argument("--result-path", type=str, required=True)
    parser.add_argument("--forward-distance", type=int, default=25)
    parser.add_argument("--turn-angle", type=int, default=15)
    parser.add_argument("--max-action-history", type=int, default=10)
    parser.add_argument("--num-generations", type=int, default=1)
    parser.add_argument("--pass-k", type=int, default=8)
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--max-episode", type=int, default=0,
                        help="Max episodes to evaluate (0 = all). Random subset with seed=42.")
    parser.add_argument("--shard-id", type=int, default=0,
                        help="Shard index for multi-node parallel (0-indexed)")
    parser.add_argument("--num-shards", type=int, default=1,
                        help="Total number of shards (1 = no sharding)")
    parser.add_argument("--num-render-gpus", type=int, default=1,
                        help="Number of GPUs for habitat rendering (workers round-robin)")
    parser.add_argument("--render-gpu-offset", type=int, default=0,
                        help="First physical GPU id used for Habitat rendering")
    parser.add_argument("--episode-manifest", type=str, default="",
                        help="Optional VLN parquet whose episodes are evaluated exactly")
    parser.add_argument("--output-csv", type=str, default="",
                        help="Per-episode difficulty CSV path (default: <result-path>/episode_difficulty.csv)")
    parser.add_argument("--save_vedio", action="store_true")
    parser.add_argument("--save-trajectory-map", action="store_true",
                        help="Save one PNG overlaying all pass@K trajectories per episode")
    args = parser.parse_args()

    if not args.output_csv:
        args.output_csv = os.path.join(args.result_path, "episode_difficulty.csv")

    api_key = os.environ.get("OPENAI_API_KEY")
    base_url = os.environ.get("OPENAI_API_BASE")
    assert api_key is not None and base_url is not None

    config = get_config(args.exp_config)
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
                    fog_of_war=FogOfWarConfig(
                        draw=True,
                        visibility_dist=5.0,
                        fov=90,
                    ),
                ),
                "collisions": CollisionsMeasurementConfig(),
            }
        )

    dataset = habitat.datasets.make_dataset(
        id_dataset=config.habitat.dataset.type, config=config.habitat.dataset
    )

    manifest_metadata = {}
    if args.episode_manifest:
        manifest_metadata = load_manifest_episode_metadata(args.episode_manifest)
        filter_dataset_by_manifest(dataset, manifest_metadata)
        print(
            f"Manifest filter: {len(dataset.episodes)} episodes from "
            f"{args.episode_manifest}"
        )

    total_episodes = len(dataset.episodes)
    if args.max_episode > 0 and args.max_episode < total_episodes:
        random.seed(42)
        random.shuffle(dataset.episodes)
        dataset.episodes = dataset.episodes[:args.max_episode]
        print(f"Subsampled {args.max_episode} / {total_episodes} episodes (seed=42)")

    if args.num_shards > 1:
        dataset.episodes.sort(key=lambda e: e.episode_id)
        all_eps = dataset.episodes
        dataset.episodes = all_eps[args.shard_id::args.num_shards]
        print(f"Shard {args.shard_id}/{args.num_shards}: {len(dataset.episodes)} / {len(all_eps)} episodes")

    dataset_splits = dataset.get_splits(args.split_num, allow_uneven_splits=True)
    num_episodes = len(dataset.episodes)

    print(f"Episodes: {num_episodes}, pass_k: {args.pass_k}, temperature: {args.temperature}")
    print(f"Total trials: {num_episodes * args.pass_k}")

    manager = mp.Manager()
    result_queue = manager.Queue()
    processes = []
    num_render_gpus = args.num_render_gpus
    for i in range(args.split_num):
        render_gpu_id = (
            args.render_gpu_offset + i % num_render_gpus
            if num_render_gpus > 0 else -1
        )
        worker_args = (
            result_queue,
            api_key,
            base_url,
            config,
            dataset_splits[i],
            args.result_path,
            args.num_generations,
            args.forward_distance,
            args.turn_angle,
            args.max_action_history,
            args.resolution_ratio,
            args.save_vedio,
            args.pass_k,
            args.temperature,
            render_gpu_id,
            manifest_metadata,
            args.save_trajectory_map,
        )
        p = mp.Process(target=evaluate_agent, args=worker_args, daemon=True)
        p.start()
        processes.append(p)

    with tqdm(total=num_episodes * args.pass_k, desc="Difficulty Eval") as pbar:
        for _ in range(num_episodes * args.pass_k):
            result = result_queue.get()
            pbar.update(1)
            pbar.set_postfix(**result)

    for p in processes:
        p.join()

    rows = compute_difficulty_csv(args.result_path, args.output_csv, args.pass_k)
    if args.save_trajectory_map:
        build_episode_trajectory_maps(args.result_path, args.pass_k)


if __name__ == "__main__":
    main()
