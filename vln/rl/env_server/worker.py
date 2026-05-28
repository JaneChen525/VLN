"""Habitat env worker — runs in its own process, holds one habitat.Env + EGL context.

IPC: parent sends dict commands via cmd_queue, worker writes dict responses to resp_queue.
Commands: {"op": "reset", "episode_id": ...} / {"op": "step", "action": ...} /
         {"op": "info"} / {"op": "close"}.
"""
import base64
import io
import multiprocessing as mp
import os
import time
import traceback
from typing import Any

import cv2
import numpy as np
from PIL import Image


def _encode_jpeg_b64(rgb: np.ndarray) -> str:
    """Encode RGB ndarray to base64 JPEG with PIL default quality (75).

    Quality MUST match `vln/rl/messages.py:encode_image_base64` and
    `vln/eval_vllm_navida.py:encode_image_base64` so the model receives
    byte-identical JPEG whether the obs flows through env_server or direct.
    """
    img = Image.fromarray(rgb.astype("uint8")).convert("RGB")
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def _load_env(exp_config_path: str):
    """Imported lazily inside the worker process so the parent doesn't need habitat."""
    import habitat
    from habitat import Env
    from habitat.config.default import get_config
    from habitat.config.default_structured_configs import (
        CollisionsMeasurementConfig,
        FogOfWarConfig,
        TopDownMapMeasurementConfig,
    )

    from habitat_extensions import measures, task  # noqa: F401

    config = get_config(exp_config_path)
    with habitat.config.read_write(config):
        config.habitat.task.measurements.update(
            {
                "top_down_map": TopDownMapMeasurementConfig(
                    map_padding=3, map_resolution=1024,
                    draw_source=True, draw_border=True, draw_shortest_path=True,
                    draw_view_points=True, draw_goal_positions=True, draw_goal_aabbs=True,
                    fog_of_war=FogOfWarConfig(draw=True, visibility_dist=5.0, fov=90),
                ),
                "collisions": CollisionsMeasurementConfig(),
            }
        )
    dataset = habitat.datasets.make_dataset(
        id_dataset=config.habitat.dataset.type, config=config.habitat.dataset
    )
    env = Env(config=config, dataset=dataset)
    episodes_by_id = {str(ep.episode_id): ep for ep in dataset.episodes}
    return env, episodes_by_id


def worker_loop(
    worker_id: int,
    exp_config_path: str,
    cmd_queue: mp.Queue,
    resp_queue: mp.Queue,
    heartbeat_value,  # mp.Value('d', 0.0), worker periodically writes time.time()
):
    """Main loop for a habitat env worker process. Exits on {"op": "close"}."""
    try:
        env, episodes_by_id = _load_env(exp_config_path)
        resp_queue.put({"ok": True, "msg": f"worker {worker_id} ready ({len(episodes_by_id)} episodes)"})
    except Exception:
        resp_queue.put({"ok": False, "error": traceback.format_exc()})
        return

    current_obs = None
    current_episode = None
    step_count = 0
    last_collisions = 0

    while True:
        heartbeat_value.value = time.time()
        cmd = cmd_queue.get()
        try:
            op = cmd.get("op")
            if op == "close":
                env.close()
                resp_queue.put({"ok": True, "msg": "closed"})
                return

            if op == "reset":
                ep_id = str(cmd["episode_id"])
                if ep_id not in episodes_by_id:
                    resp_queue.put({"ok": False, "error": f"episode_id {ep_id} not in dataset"})
                    continue
                ep = episodes_by_id[ep_id]
                env.current_episode = ep
                obs = env.reset()
                current_obs = obs
                current_episode = ep
                step_count = 0
                last_collisions = 0
                scene_id = os.path.basename(ep.scene_id).split(".")[0]
                resp_queue.put({
                    "ok": True,
                    "rgb_jpeg_b64": _encode_jpeg_b64(obs["rgb"]),
                    "instruction": obs["instruction"]["text"],
                    "scene_id": scene_id,
                    "episode_id": ep_id,
                })
                continue

            if op == "step":
                action = int(cmd["action"])
                obs = env.step({"action": action})
                current_obs = obs
                step_count += 1
                done = env.episode_over
                info = env.get_metrics()
                collisions = int(info.get("collisions", {}).get("count", 0)) if isinstance(info.get("collisions"), dict) else 0
                resp_queue.put({
                    "ok": True,
                    "rgb_jpeg_b64": _encode_jpeg_b64(obs["rgb"]),
                    "instruction": obs["instruction"]["text"],
                    "done": done,
                    "success": float(info.get("success", 0.0)),
                    "spl": float(info.get("spl", 0.0)),
                    "oracle_success": float(info.get("oracle_success", 0.0)),
                    "ne": float(info.get("distance_to_goal", 0.0)),
                    "step_count": step_count,
                    "collisions": collisions,
                })
                continue

            if op == "private":
                if current_episode is None:
                    resp_queue.put({"ok": False, "error": "no active episode"})
                    continue
                info = env.get_metrics() if current_obs is not None else {}
                resp_queue.put({
                    "ok": True,
                    "episode_id": str(current_episode.episode_id),
                    "scene_id": os.path.basename(current_episode.scene_id).split(".")[0],
                    "gt_path_len": float(getattr(current_episode, "info", {}).get("geodesic_distance", 0.0)) if hasattr(current_episode, "info") else 0.0,
                    "oracle_success": float(info.get("oracle_success", 0.0)),
                    "final_ne": float(info.get("distance_to_goal", 0.0)),
                })
                continue

            resp_queue.put({"ok": False, "error": f"unknown op: {op}"})

        except Exception:
            resp_queue.put({"ok": False, "error": traceback.format_exc()})
