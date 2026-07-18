"""Compute pass@1, pass@4, SPL, OSR, and NE from evaluation results."""

import glob
import json
import os
import sys
from collections import defaultdict


def load_results(result_dir):
    files = [os.path.join(result_dir, "result.json")]
    files.extend(glob.glob(os.path.join(result_dir, "result_w*.json")))
    seen = {}
    for path in files:
        if not os.path.exists(path):
            continue
        with open(path) as file:
            for line in file:
                try:
                    result = json.loads(line.strip())
                except (json.JSONDecodeError, ValueError):
                    continue
                key = (
                    result.get("scene_id"),
                    result.get("episode_id"),
                    result.get("trial_id"),
                )
                required = ("scene_id", "episode_id", "trial_id", "success")
                if all(result.get(name) is not None for name in required):
                    seen[key] = result
    return list(seen.values())


def pass_k_metrics(records, pass_k=4):
    by_episode = defaultdict(list)
    for record in records:
        by_episode[(record["scene_id"], record["episode_id"])].append(record)

    n_episodes = 0
    n_complete_pass_k = 0
    n_trials_total = 0
    n_trials_success = 0
    sum_pass_1 = 0.0
    sum_pass_k = 0.0
    sum_spl = 0.0
    sum_osr = 0.0
    sum_ne = 0.0
    sum_steps = 0.0

    for trials in by_episode.values():
        trials.sort(key=lambda item: item["trial_id"])
        n_episodes += 1
        n_trials_total += len(trials)
        successes = [float(item["success"]) for item in trials]
        spls = [float(item["spl"]) for item in trials]
        oracle_successes = [float(item["os"]) for item in trials]
        navigation_errors = [float(item["ne"]) for item in trials]
        steps = [float(item.get("steps", 0)) for item in trials]

        sum_pass_1 += sum(successes) / max(1, len(successes))
        n_trials_success += int(sum(successes))
        selected = slice(0, pass_k)
        if len(trials) >= pass_k:
            n_complete_pass_k += 1
        sum_pass_k += max(successes[selected]) if successes else 0.0
        sum_spl += sum(spls[selected]) / max(1, min(pass_k, len(spls)))
        sum_osr += max(oracle_successes[selected]) if oracle_successes else 0.0
        sum_ne += min(navigation_errors[selected]) if navigation_errors else 999.0
        sum_steps += sum(steps) / max(1, len(steps))

    denominator = max(1, n_episodes)
    return {
        "n_episodes": n_episodes,
        "n_complete_pass_k": n_complete_pass_k,
        "n_trials_total": n_trials_total,
        "n_trials_success": n_trials_success,
        "pass@1": sum_pass_1 / denominator,
        "pass@4_or_partial": sum_pass_k / denominator,
        "spl": sum_spl / denominator,
        "osr": sum_osr / denominator,
        "ne": sum_ne / denominator,
        "steps_avg": sum_steps / denominator,
    }


if __name__ == "__main__":
    records = load_results(sys.argv[1])
    metrics = pass_k_metrics(records)
    print(f"=== {sys.argv[1]} ===")
    for name, value in metrics.items():
        if isinstance(value, float):
            print(f"  {name:25s}: {value:.4f}")
        else:
            print(f"  {name:25s}: {value}")
