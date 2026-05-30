"""Quick check: trainer can resolve + byte-verify selected frames from sidecar.

    python -m vln.rl.trainer.verify_frames /tmp/rl/frames_test.parquet /tmp/rl/frames
"""
import base64
import hashlib
import json
import os
import sys

import pandas as pd
from PIL import Image


def main(parquet_path, frames_dir):
    df = pd.read_parquet(parquet_path)
    total = ok = 0
    last_size = None
    for _, row in df.iterrows():
        for m in json.loads(row["image_manifest"]):
            total += 1
            p = os.path.join(frames_dir, m["sha256"] + ".jpg")
            if not os.path.exists(p):
                print("MISSING", p)
                continue
            raw = open(p, "rb").read()
            sha = hashlib.sha256(base64.b64encode(raw)).hexdigest()[:16]
            img = Image.open(p)
            img.load()
            last_size = img.size
            if sha == m["sha256"]:
                ok += 1
    print(f"manifest frame refs: {total}, resolved+sha-verified: {ok}, img size: {last_size}")
    assert ok == total, "some frames missing or sha mismatch"
    print("OK: all manifest frames resolvable + byte-identical")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
