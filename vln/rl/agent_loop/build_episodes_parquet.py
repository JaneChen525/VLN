"""Build a tiny habitat-episode parquet to drive verl's dataloader.

Each row = one episode. Frames are NOT stored here — HabitatAgentLoop resets the
env_server at rollout time and builds the real prompt from live frames. The
`prompt` column is a text-only placeholder so verl can tokenize/batch; run()
ignores it. Episode ids are pulled from the env_server /episodes endpoint.

Usage (inside the verl container, env_server reachable):
    python -m vln.rl.agent_loop.build_episodes_parquet \
        --env-server-url http://127.0.0.1:8000 --n 8 \
        --out ~/data/habitat/train.parquet
"""
import argparse
import os

import pandas as pd

from vln.rl.env_server.client import EnvClient

PLACEHOLDER = "Habitat VLN episode. Frames are provided by the env_server at rollout time."


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--env-server-url", default="http://127.0.0.1:8000")
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--split", default="train")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    client = EnvClient(args.env_server_url)
    episode_ids = client.episodes(limit=args.n)
    client.close()
    assert episode_ids, "env_server returned no episodes"

    rows = []
    for idx, ep in enumerate(episode_ids[: args.n]):
        rows.append(
            {
                "data_source": "habitat/r2r",
                "prompt": [{"role": "user", "content": PLACEHOLDER}],
                "agent_name": "habitat",
                "ability": "vln",
                "reward_model": {"style": "rule", "ground_truth": ""},
                "extra_info": {"split": args.split, "index": idx, "episode_id": str(ep), "trial_id": 0},
            }
        )

    df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(os.path.expanduser(args.out)), exist_ok=True)
    df.to_parquet(os.path.expanduser(args.out))
    print(f"wrote {len(df)} rows -> {args.out}")
    print(df[["data_source", "agent_name"]].head())


if __name__ == "__main__":
    main()
