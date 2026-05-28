"""Entrypoint: python -m vln.rl.env_server.launch --exp-config config/vln_r2r.yaml --port 8002

Pool size = 1 for 7.3a sanity. Bump --pool-size in 7.3c.
"""
import argparse

import uvicorn

from vln.rl.env_server.server import app, set_config


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp-config", type=str, required=True)
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8002)
    parser.add_argument("--pool-size", type=int, default=1)
    args = parser.parse_args()

    set_config(args.exp_config, pool_size=args.pool_size)
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
