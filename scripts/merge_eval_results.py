"""Merge resumable evaluation shards into one deduplicated result.json."""

import argparse
import json
import os
import tempfile


REQUIRED_FIELDS = (
    "scene_id",
    "episode_id",
    "episode_instruction",
    "trial_id",
    "trial_total",
    "success",
    "spl",
    "os",
    "ne",
    "steps",
)


def result_key(item):
    return (
        str(item["scene_id"]),
        str(item["episode_id"]),
        int(item["trial_id"]),
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("inputs", nargs="+")
    args = parser.parse_args()

    records = {}
    for path in args.inputs:
        if not os.path.exists(path):
            continue
        with open(path) as file:
            for line in file:
                try:
                    item = json.loads(line)
                except (json.JSONDecodeError, ValueError):
                    continue
                if not all(field in item for field in REQUIRED_FIELDS):
                    continue
                key = result_key(item)
                if key in records and records[key] != item:
                    raise ValueError(f"conflicting duplicate result for {key}")
                records[key] = item

    output_dir = os.path.dirname(os.path.abspath(args.output))
    os.makedirs(output_dir, exist_ok=True)
    fd, temporary_path = tempfile.mkstemp(prefix="result.", suffix=".tmp", dir=output_dir)
    try:
        with os.fdopen(fd, "w") as file:
            for key in sorted(records):
                file.write(json.dumps(records[key], ensure_ascii=False) + "\n")
        os.replace(temporary_path, args.output)
    except BaseException:
        if os.path.exists(temporary_path):
            os.unlink(temporary_path)
        raise

    print(f"Merged {len(records)} unique trials into {args.output}")


if __name__ == "__main__":
    main()
