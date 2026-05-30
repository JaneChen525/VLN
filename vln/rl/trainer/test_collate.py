"""3a shape/sanity test. Runs in verl-dev docker:

  docker exec verl-dev bash -lc "cd /workspace/WorldModel && PYTHONPATH=. \
    VLN_TEST_CKPT=/workspace/WorldModel/checkpoints/Qwen3-VL-4B-vln-r2r-merged \
    VLN_TEST_PARQUET=/workspace/rl_test/frames_test.parquet \
    VLN_TEST_FRAMES=/workspace/rl_test/frames \
    python -m vln.rl.trainer.test_collate"
"""
import os


def main():
    import pandas as pd
    from transformers import AutoProcessor

    from vln.rl.trainer.advantage import compute_grpo_advantages
    from vln.rl.trainer.collate import collate_batch, make_example

    proc = AutoProcessor.from_pretrained(os.environ["VLN_TEST_CKPT"], trust_remote_code=True)
    df = pd.read_parquet(os.environ["VLN_TEST_PARQUET"])
    frames = os.environ["VLN_TEST_FRAMES"]

    df, stats = compute_grpo_advantages(df)
    print("group stats:", stats)

    # take 4 rows spanning short/long prompts
    idxs = [0, len(df) // 3, 2 * len(df) // 3, len(df) - 1]
    examples = [make_example(df.iloc[i], frames, proc, float(df.iloc[i]["advantage"])) for i in idxs]

    pad_id = proc.tokenizer.pad_token_id or proc.tokenizer.eos_token_id
    batch = collate_batch(examples, pad_id)

    B = len(examples)
    L = batch["input_ids"].shape[1]
    print(f"batch B={B} L={L}")
    assert batch["input_ids"].shape == (B, L)
    assert batch["attention_mask"].shape == (B, L)
    assert batch["response_mask"].shape == (B, L)
    assert batch["advantages"].shape == (B, L)
    assert batch["position_ids"].shape == (B, 3, L), batch["position_ids"].shape
    assert batch["multi_modal_inputs"].shape == (B,)

    # advantages must be 0 where response_mask is 0, and == scalar where 1
    for i, ix in enumerate(idxs):
        a = float(df.iloc[ix]["advantage"])
        rm = batch["response_mask"][i].bool()
        adv = batch["advantages"][i]
        assert (adv[~rm] == 0).all(), f"row{i}: advantage leaked into prompt/pad"
        if rm.any():
            assert (adv[rm] == a).all(), f"row{i}: response advantage != {a}"
    # per-sample multimodal patches present
    for i in range(B):
        assert "pixel_values" in batch["multi_modal_inputs"][i]
        assert "image_grid_thw" in batch["multi_modal_inputs"][i]
    print("collate shape/sanity test PASSED")


if __name__ == "__main__":
    main()
