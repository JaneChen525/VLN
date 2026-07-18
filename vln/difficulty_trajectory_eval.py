"""Pass@K difficulty evaluation with one static trajectory map per episode.

This script keeps ``difficulty_eval.py`` unchanged and reuses the authoritative
NaVIDA agent from ``eval_vllm_navida.py``. Each output PNG overlays all K
rollouts: successful routes use green shades and failed routes use red shades.
"""

import argparse
import csv
import json
import multiprocessing as mp
import os
import queue
import random
import re
import time

import cv2
import habitat
import imageio.v2 as imageio
import numpy as np
from habitat import Env
from habitat.config.default import get_config
from habitat.config.default_structured_configs import (
    CollisionsMeasurementConfig,
    FogOfWarConfig,
    TopDownMapMeasurementConfig,
)
from habitat.utils.visualizations import maps
from tqdm import tqdm

import eval_vllm_navida as standard_eval


def episode_key(scene_id, episode_id):
    scene = os.path.splitext(os.path.basename(str(scene_id)))[0]
    return scene, str(episode_id)


def load_manifest(path):
    import pyarrow.parquet as pq

    metadata = {}
    rows = pq.read_table(path, columns=["extra_info"]).to_pylist()
    for row in rows:
        info = row["extra_info"]
        key = episode_key(info["scene_id"], info["episode_id"])
        if key in metadata:
            raise ValueError(f"Duplicate episode in manifest: {key}")
        metadata[key] = {
            "manifest_pass_rate": info.get("pass_rate"),
            "manifest_instruction": info.get("instruction"),
        }
    return metadata


def filter_dataset(dataset, manifest, selected_ids):
    habitat_episodes = {
        episode_key(ep.scene_id, ep.episode_id): ep for ep in dataset.episodes
    }
    selected_ids = {str(value) for value in selected_ids}
    if manifest:
        missing = [key for key in manifest if key not in habitat_episodes]
        if missing:
            raise ValueError(
                f"{len(missing)} manifest episodes are absent from Habitat; "
                f"first missing: {missing[:5]}"
            )
        keys = [key for key in manifest if not selected_ids or key[1] in selected_ids]
    else:
        keys = [
            key for key in habitat_episodes
            if not selected_ids or key[1] in selected_ids
        ]
    if selected_ids:
        found = {key[1] for key in keys}
        absent = sorted(selected_ids - found)
        if absent:
            raise ValueError(f"Episode ids not found in manifest: {absent}")
    dataset.episodes = [habitat_episodes[key] for key in keys]


def output_name(scene_id, episode_id):
    clean_scene = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(scene_id))
    clean_episode = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(episode_id))
    return f"{clean_scene}_ep{clean_episode}"


def load_completed_trials(result_path, pass_k):
    result_file = os.path.join(result_path, "result.json")
    completed = set()
    if not os.path.isfile(result_file):
        return completed
    with open(result_file) as handle:
        for line in handle:
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                continue
            base_map = item.get("trajectory_base_map")
            if (
                item.get("trial_total") == pass_k
                and item.get("trajectory_points")
                and base_map
                and os.path.isfile(os.path.join(result_path, base_map))
            ):
                completed.add(
                    (
                        item["scene_id"],
                        str(item["episode_id"]),
                        item["episode_instruction"],
                        int(item["trial_id"]),
                        pass_k,
                    )
                )
    return completed


def agent_xy(top_down_map):
    coordinates = top_down_map.get("agent_map_coord") or []
    if not coordinates:
        return None
    row, column = coordinates[0]
    return [int(column), int(row)]


def save_base_map(result_path, scene_id, episode_id, raw_map):
    base_dir = os.path.join(result_path, "trajectory_maps", ".base")
    os.makedirs(base_dir, exist_ok=True)
    path = os.path.join(base_dir, f"{output_name(scene_id, episode_id)}.png")
    if not os.path.isfile(path):
        imageio.imwrite(path, maps.colorize_topdown_map(raw_map))
    return os.path.relpath(path, result_path)


def evaluate_worker(
    result_queue,
    api_key,
    base_url,
    config,
    dataset,
    result_path,
    num_generations,
    forward_distance,
    turn_angle,
    max_action_history,
    resolution_ratio,
    pass_k,
    temperature,
    render_gpu_id,
    manifest,
):
    if render_gpu_id >= 0:
        from omegaconf import read_write

        with read_write(config):
            config.habitat.simulator.habitat_sim_v0.gpu_device_id = render_gpu_id

    env = Env(config.habitat, dataset)
    agent = standard_eval.NaVIDA_Agent(
        api_key,
        base_url,
        result_path,
        forward_distance,
        turn_angle,
        max_action_history,
        resolution_ratio,
        num_generations,
        save_video=False,
    )
    agent.temperature = temperature
    completed = load_completed_trials(result_path, pass_k)

    for target_episode in list(env.episodes):
        scene_id = target_episode.scene_id.split("/")[-2]
        episode_id = target_episode.episode_id
        instruction = target_episode.instruction.instruction_text
        metadata = manifest.get(episode_key(scene_id, episode_id), {})

        for trial_id in range(pass_k):
            completed_key = (
                scene_id,
                str(episode_id),
                instruction,
                trial_id,
                pass_k,
            )
            if completed_key in completed:
                result_queue.put({"t_episode": 0, "skipped": 1})
                continue

            start_time = time.time()
            env.current_episode = target_episode
            observations = env.reset()
            agent.reset()
            steps = 0
            unchanged_distance_steps = 0
            last_distance = 999
            points = []
            base_map = None

            while not env.episode_over:
                metrics = env.get_metrics()
                top_down_map = metrics["top_down_map"]
                if base_map is None:
                    base_map = save_base_map(
                        result_path, scene_id, episode_id, top_down_map["map"]
                    )
                point = agent_xy(top_down_map)
                if point is not None and (not points or point != points[-1]):
                    points.append(point)

                if metrics["distance_to_goal"] != last_distance:
                    last_distance = metrics["distance_to_goal"]
                    unchanged_distance_steps = 0
                else:
                    unchanged_distance_steps += 1

                action = agent.act(observations, metrics, episode_id)
                if unchanged_distance_steps > 25 or steps > 200:
                    action = {"action": 0}
                steps += 1
                observations = env.step(action)

            metrics = env.get_metrics()
            point = agent_xy(metrics["top_down_map"])
            if point is not None and (not points or point != points[-1]):
                points.append(point)

            result = {
                "scene_id": scene_id,
                "episode_id": int(episode_id) if str(episode_id).isdigit() else episode_id,
                "trial_id": trial_id,
                "trial_total": pass_k,
                "success": metrics["success"],
                "spl": metrics["spl"],
                "os": metrics["oracle_success"],
                "ne": metrics["distance_to_goal"],
                "steps": steps,
                "episode_instruction": instruction,
                "trajectory_points": points,
                "trajectory_base_map": base_map,
                **metadata,
            }
            with open(os.path.join(result_path, "result.json"), "a") as handle:
                handle.write(json.dumps(result) + "\n")
            completed.add(completed_key)
            result_queue.put({"t_episode": time.time() - start_time})


def draw_wrapped_text(image, text, x, y, max_width):
    line = ""
    for word in str(text).split():
        candidate = f"{line} {word}".strip()
        width = cv2.getTextSize(
            candidate, cv2.FONT_HERSHEY_SIMPLEX, 0.52, 1
        )[0][0]
        if line and width > max_width:
            cv2.putText(
                image,
                line,
                (x, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.52,
                (25, 25, 25),
                1,
                cv2.LINE_AA,
            )
            line = word
            y += 23
        else:
            line = candidate
    if line:
        cv2.putText(
            image,
            line,
            (x, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.52,
            (25, 25, 25),
            1,
            cv2.LINE_AA,
        )


def build_maps(result_path, pass_k):
    trials = {}
    with open(os.path.join(result_path, "result.json")) as handle:
        for line in handle:
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                continue
            if item.get("trajectory_points") and item.get("trajectory_base_map"):
                key = (
                    *episode_key(item["scene_id"], item["episode_id"]),
                    int(item["trial_id"]),
                )
                trials[key] = item

    episodes = {}
    for item in trials.values():
        episodes.setdefault(
            episode_key(item["scene_id"], item["episode_id"]), []
        ).append(item)

    green = [
        (0, 100, 35), (0, 125, 50), (0, 150, 65), (0, 175, 80),
        (20, 135, 95), (35, 160, 110), (50, 185, 125), (65, 210, 140),
    ]
    red = [
        (160, 0, 20), (185, 15, 30), (210, 30, 40), (235, 45, 50),
        (175, 0, 80), (200, 20, 95), (225, 40, 110), (245, 60, 125),
    ]
    output_dir = os.path.join(result_path, "trajectory_maps")
    os.makedirs(output_dir, exist_ok=True)
    generated = 0

    for (scene_id, episode_id), episode_trials in sorted(episodes.items()):
        episode_trials.sort(key=lambda item: int(item["trial_id"]))
        if len(episode_trials) != pass_k:
            print(
                f"Skip incomplete episode {scene_id}/{episode_id}: "
                f"{len(episode_trials)}/{pass_k}"
            )
            continue

        base_map = imageio.imread(
            os.path.join(result_path, episode_trials[0]["trajectory_base_map"])
        )[:, :, :3]
        map_height, map_width = base_map.shape[:2]
        header_height, legend_width = 125, 410
        canvas = np.full(
            (map_height + header_height, map_width + legend_width, 3),
            255,
            dtype=np.uint8,
        )
        canvas[header_height:, :map_width] = base_map
        success_count = sum(int(item["success"]) for item in episode_trials)
        prior_rate = episode_trials[0].get("manifest_pass_rate")
        prior_text = "n/a" if prior_rate is None else f"{float(prior_rate):.3f}"
        cv2.putText(
            canvas,
            f"scene={scene_id} episode={episode_id} success={success_count}/{pass_k} prior={prior_text}",
            (14, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (20, 20, 20),
            2,
            cv2.LINE_AA,
        )
        draw_wrapped_text(
            canvas,
            episode_trials[0]["episode_instruction"],
            14,
            58,
            map_width + legend_width - 28,
        )

        for item in episode_trials:
            trial_id = int(item["trial_id"])
            succeeded = bool(item["success"])
            color = (green if succeeded else red)[trial_id % 8]
            points = np.asarray(item["trajectory_points"], dtype=np.int32)
            points[:, 1] += header_height
            if len(points) > 1:
                cv2.polylines(
                    canvas,
                    [points.reshape(-1, 1, 2)],
                    False,
                    color,
                    4,
                    cv2.LINE_AA,
                )
            start = tuple(int(value) for value in points[0])
            end = tuple(int(value) for value in points[-1])
            cv2.circle(canvas, start, 6, (25, 90, 220), -1, cv2.LINE_AA)
            cv2.circle(canvas, end, 7, color, -1, cv2.LINE_AA)
            cv2.putText(
                canvas,
                f"t{trial_id}",
                (end[0] + 7, end[1] - 7),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.52,
                color,
                2,
                cv2.LINE_AA,
            )

            x, y = map_width + 20, header_height + 32 + trial_id * 56
            cv2.line(canvas, (x, y), (x + 44, y), color, 5, cv2.LINE_AA)
            status = "SUCCESS" if succeeded else "FAILURE"
            cv2.putText(
                canvas,
                f"trial {trial_id}: {status}",
                (x + 56, y + 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                color,
                2,
                cv2.LINE_AA,
            )
            cv2.putText(
                canvas,
                f"NE={float(item['ne']):.2f}m SPL={float(item['spl']):.3f} steps={int(item['steps'])}",
                (x, y + 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (45, 45, 45),
                1,
                cv2.LINE_AA,
            )

        output = os.path.join(
            output_dir, f"{output_name(scene_id, episode_id)}_{pass_k}rollouts.png"
        )
        imageio.imwrite(output, canvas)
        generated += 1

    print(f"Trajectory maps: {generated} PNGs in {output_dir}")


def compute_difficulty_csv(result_path, output_csv):
    deduplicated = {}
    with open(os.path.join(result_path, "result.json")) as handle:
        for line in handle:
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                continue
            if all(key in item for key in ("scene_id", "episode_id", "trial_id")):
                key = (
                    *episode_key(item["scene_id"], item["episode_id"]),
                    int(item["trial_id"]),
                )
                deduplicated[key] = item

    episodes = {}
    for item in deduplicated.values():
        key = episode_key(item["scene_id"], item["episode_id"])
        episodes.setdefault(key, []).append(item)

    rows = []
    for key, trials in sorted(episodes.items()):
        count = len(trials)
        successes = sum(int(item["success"]) for item in trials)
        rows.append(
            {
                "scene_id": key[0],
                "episode_id": key[1],
                "pass_k": count,
                "success_count": successes,
                "pass_rate": successes / count,
                "avg_ne": sum(float(item["ne"]) for item in trials) / count,
                "avg_spl": sum(float(item["spl"]) for item in trials) / count,
                "avg_steps": sum(int(item["steps"]) for item in trials) / count,
            }
        )

    os.makedirs(os.path.dirname(output_csv) or ".", exist_ok=True)
    with open(output_csv, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]) if rows else [
            "scene_id", "episode_id", "pass_k", "success_count", "pass_rate",
            "avg_ne", "avg_spl", "avg_steps",
        ])
        writer.writeheader()
        writer.writerows(rows)
    print(f"Difficulty CSV: {len(rows)} episodes in {output_csv}")


def main():
    np.random.seed(42)
    random.seed(42)
    parser = argparse.ArgumentParser(
        description="Pass@K difficulty eval with one static trajectory map per episode"
    )
    parser.add_argument("--exp-config", required=True)
    parser.add_argument(
        "--episode-manifest", default=os.environ.get("EPISODE_MANIFEST", "")
    )
    parser.add_argument("--result-path", required=True)
    parser.add_argument("--split-num", type=int, default=24)
    parser.add_argument("--pass-k", type=int, default=8)
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--max-episode", type=int, default=0)
    parser.add_argument("--episode-id", action="append", default=[])
    parser.add_argument("--forward-distance", type=int, default=25)
    parser.add_argument("--turn-angle", type=int, default=15)
    parser.add_argument("--max-action-history", type=int, default=200)
    parser.add_argument("--num-generations", type=int, default=1)
    parser.add_argument("--resolution-ratio", type=float, default=1.0)
    parser.add_argument("--num-render-gpus", type=int, default=0)
    parser.add_argument("--shard-id", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--output-csv", default="")
    args = parser.parse_args()

    api_key = os.environ.get("OPENAI_API_KEY")
    base_url = os.environ.get("OPENAI_API_BASE")
    if not api_key or not base_url:
        raise ValueError("OPENAI_API_KEY and OPENAI_API_BASE are required")
    os.makedirs(args.result_path, exist_ok=True)

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
                        draw=True, visibility_dist=5.0, fov=90
                    ),
                ),
                "collisions": CollisionsMeasurementConfig(),
            }
        )

    dataset = habitat.datasets.make_dataset(
        id_dataset=config.habitat.dataset.type, config=config.habitat.dataset
    )
    manifest = load_manifest(args.episode_manifest) if args.episode_manifest else {}
    filter_dataset(dataset, manifest, args.episode_id)

    if args.max_episode > 0 and args.max_episode < len(dataset.episodes):
        random.seed(42)
        random.shuffle(dataset.episodes)
        dataset.episodes = dataset.episodes[: args.max_episode]

    if args.num_shards > 1:
        if not 0 <= args.shard_id < args.num_shards:
            raise ValueError("shard-id must be in [0, num-shards)")
        dataset.episodes.sort(key=lambda episode: str(episode.episode_id))
        dataset.episodes = dataset.episodes[args.shard_id :: args.num_shards]

    num_episodes = len(dataset.episodes)
    if num_episodes == 0:
        raise ValueError("No episodes selected")
    worker_count = min(args.split_num, num_episodes)
    dataset_splits = dataset.get_splits(worker_count, allow_uneven_splits=True)
    print(
        f"Episodes={num_episodes} pass_k={args.pass_k} "
        f"temperature={args.temperature} workers={worker_count}"
    )

    manager = mp.Manager()
    result_queue = manager.Queue()
    processes = []
    for worker_id, dataset_split in enumerate(dataset_splits):
        render_gpu_id = (
            worker_id % args.num_render_gpus if args.num_render_gpus > 0 else -1
        )
        process = mp.Process(
            target=evaluate_worker,
            args=(
                result_queue,
                api_key,
                base_url,
                config,
                dataset_split,
                args.result_path,
                args.num_generations,
                args.forward_distance,
                args.turn_angle,
                args.max_action_history,
                args.resolution_ratio,
                args.pass_k,
                args.temperature,
                render_gpu_id,
                manifest,
            ),
            daemon=True,
        )
        process.start()
        processes.append(process)

    total_trials = num_episodes * args.pass_k
    completed_trials = 0
    with tqdm(total=total_trials, desc="Trajectory Difficulty Eval") as bar:
        while completed_trials < total_trials:
            try:
                result = result_queue.get(timeout=30)
            except queue.Empty:
                failed = [
                    process.exitcode
                    for process in processes
                    if process.exitcode not in (None, 0)
                ]
                if failed:
                    raise RuntimeError(f"Evaluation workers failed: {failed}")
                continue
            completed_trials += 1
            bar.update(1)
            bar.set_postfix(**result)

    for process in processes:
        process.join()
        if process.exitcode != 0:
            raise RuntimeError(f"Evaluation worker failed with exit code {process.exitcode}")

    output_csv = args.output_csv or os.path.join(
        args.result_path, "episode_difficulty.csv"
    )
    compute_difficulty_csv(args.result_path, output_csv)
    build_maps(args.result_path, args.pass_k)


if __name__ == "__main__":
    main()
