"""Generate filtered episode manifest parquet based on difficulty eval results.

Filters episodes to pass_rate range [lo, hi] for GRPO training
(boundary episodes where model sometimes succeeds, sometimes fails).

Usage (on env1):
    conda activate vln
    cd /var/data0/sandbox/janec/WorldModel
    python vln/reinforcement_learning/recipe/vln_navida/make_filtered_manifest.py \
        --filtered-json data/task14/filtered_episodes.json \
        --data-dir data/vln_eval_datasets/r2r \
        --split train \
        --out /root/data/vln_r2r_train_filtered.parquet
"""
import argparse
import gzip
import json
from pathlib import Path

import numpy as np
import pandas as pd


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--filtered-json", type=Path, required=True,
                        help="JSON with {episodes: {episode_id: pass_rate}}")
    parser.add_argument("--data-dir", type=Path,
                        default=Path("data/vln_eval_datasets/r2r"))
    parser.add_argument("--split", default="train")
    parser.add_argument("--out", type=Path,
                        default=Path("/root/data/vln_r2r_train_filtered.parquet"))
    parser.add_argument("--lo", type=float, default=0.2)
    parser.add_argument("--hi", type=float, default=0.8)
    args = parser.parse_args()

    with open(args.filtered_json) as f:
        fdata = json.load(f)

    filtered_ids = set()
    for eid, pr in fdata["episodes"].items():
        if args.lo <= pr <= args.hi:
            filtered_ids.add(str(eid))

    gz_path = args.data_dir / args.split / f"{args.split}.json.gz"
    with gzip.open(gz_path, "rt") as f:
        data = json.load(f)

    rows = []
    for ep in data["episodes"]:
        eid = str(ep["episode_id"])
        if eid not in filtered_ids:
            continue
        rows.append({
            "data_source": "vln",
            "prompt": [{"role": "user", "content": "placeholder"}],
            "agent_name": "vln_full_episode_agent",
            "extra_info": {
                "episode_id": eid,
                "scene_id": str(ep.get("scene_id", "")),
                "instruction": str(ep.get("instruction", {}).get("instruction_text", ""))
                    if isinstance(ep.get("instruction"), dict)
                    else str(ep.get("instruction", "")),
                "config_path": "config/vln_r2r.yaml",
                "pass_rate": fdata["episodes"].get(eid, -1),
            },
        })

    df = pd.DataFrame(rows)
    df["prompt"] = df["prompt"].apply(lambda x: np.array(x, dtype=object))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(args.out, index=False)
    print(f"Filtered: {len(df)} / {len(data['episodes'])} episodes "
          f"(pass_rate [{args.lo}, {args.hi}])")
    print(f"Output: {args.out}")


if __name__ == "__main__":
    main()
