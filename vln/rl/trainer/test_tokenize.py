"""Token-alignment golden test for build_training_example.

Needs the Qwen3VLProcessor + a real rollout parquet + frame sidecar, so it runs
in the verl-dev docker, not in plain CI:

  docker exec verl-dev bash -lc "cd /workspace/WorldModel && PYTHONPATH=. \
    VLN_TEST_CKPT=/workspace/WorldModel/checkpoints/Qwen3-VL-4B-vln-r2r-merged \
    VLN_TEST_PARQUET=/tmp/rl/frames_test.parquet \
    VLN_TEST_FRAMES=/tmp/rl/frames \
    python -m vln.rl.trainer.test_tokenize"

Checks per row:
  - loss_mask is 0 over the prompt span, 1 over the response span
  - response tokens decode back to response_text (round-trip, no drift)
  - #image placeholders / grids == #frames in the manifest
  - pixel_values present when the row has images
"""
import json
import os
import sys


def _load():
    from transformers import AutoProcessor
    ckpt = os.environ["VLN_TEST_CKPT"]
    proc = AutoProcessor.from_pretrained(ckpt, trust_remote_code=True)
    return proc


def main():
    import pandas as pd
    from vln.rl.trainer.tokenize import build_training_example, reconstruct_messages

    proc = _load()
    parquet = os.environ["VLN_TEST_PARQUET"]
    frames = os.environ["VLN_TEST_FRAMES"]
    df = pd.read_parquet(parquet)

    rows = [df.iloc[0], df.iloc[len(df) // 2], df.iloc[-1]]
    for i, row in enumerate(rows):
        ex = build_training_example(row, frames, proc)
        n_frames = len(json.loads(row["image_manifest"]))

        # 1) masking
        lm = ex["loss_mask"]
        plen, rlen = ex["prompt_len"], ex["response_len"]
        assert lm[:plen].sum().item() == 0, f"row{i}: prompt region not masked"
        assert lm[plen:].sum().item() == rlen, f"row{i}: response region not all 1"
        assert lm.shape[0] == ex["input_ids"].shape[0]

        # 2) response round-trip (re-tokenized response decodes back unchanged)
        resp_ids = ex["input_ids"][plen:plen + rlen - 1]  # drop appended eos
        decoded = proc.tokenizer.decode(resp_ids, skip_special_tokens=True)
        assert decoded.strip() == row["response_text"].strip(), \
            f"row{i}: response round-trip mismatch:\n  got={decoded!r}\n  exp={row['response_text']!r}"

        # 3) image count
        if "image_grid_thw" in ex:
            assert ex["image_grid_thw"].shape[0] == n_frames, \
                f"row{i}: grid count {ex['image_grid_thw'].shape[0]} != manifest {n_frames}"
            assert "pixel_values" in ex
        print(f"row{i}: OK prompt_len={plen} response_len={rlen} frames={n_frames} "
              f"total_tokens={ex['input_ids'].shape[0]}")

    print("token-alignment golden test PASSED")


if __name__ == "__main__":
    sys.exit(main())
