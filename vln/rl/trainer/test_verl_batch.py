"""3b-2: verify to_verl_batch produces verl's prompt-left-pad / response-
right-pad layout from real parquet, and that the split is loss-mask-faithful.

  docker exec verl-dev bash -lc "cd /workspace/WorldModel && PYTHONPATH=. \
    VLN_TEST_CKPT=/workspace/WorldModel/checkpoints/Qwen3-VL-4B-vln-r2r-merged \
    VLN_TEST_PARQUET=/workspace/rl_test/frames_test.parquet \
    VLN_TEST_FRAMES=/workspace/rl_test/frames \
    python -m vln.rl.trainer.test_verl_batch"
"""
import os


def main():
    import pandas as pd
    import torch
    from transformers import AutoProcessor

    from vln.rl.trainer.advantage import compute_grpo_advantages
    from vln.rl.trainer.collate import make_example, to_verl_batch

    proc = AutoProcessor.from_pretrained(os.environ["VLN_TEST_CKPT"], trust_remote_code=True)
    df = pd.read_parquet(os.environ["VLN_TEST_PARQUET"])
    frames = os.environ["VLN_TEST_FRAMES"]
    df, _ = compute_grpo_advantages(df)

    idxs = [0, len(df) // 3, 2 * len(df) // 3, len(df) - 1]
    examples = [make_example(df.iloc[i], frames, proc, float(df.iloc[i]["advantage"])) for i in idxs]
    pad_id = proc.tokenizer.pad_token_id or proc.tokenizer.eos_token_id

    batch = to_verl_batch(examples, pad_id, proc)
    B = len(examples)
    P = batch["prompts"].shape[1]
    R = batch["responses"].shape[1]
    L = batch["input_ids"].shape[1]
    print(f"B={B} P={P} R={R} L={L}")

    assert L == P + R
    assert batch["responses"].shape == (B, R)
    assert batch["response_mask"].shape == (B, R)
    assert batch["advantages"].shape == (B, R)
    assert batch["position_ids"].shape == (B, 3, L), batch["position_ids"].shape

    for i, ix in enumerate(idxs):
        e = examples[i]
        pl, rl = e["prompt_len"], e["response_len"]
        orig = e["input_ids"]
        # response real tokens (mask==1) must equal original response span
        rm = batch["response_mask"][i].bool()
        assert rm.sum().item() == rl, f"row{i}: response_mask sum {rm.sum()} != {rl}"
        got_resp = batch["responses"][i][rm]
        assert torch.equal(got_resp, orig[pl:pl + rl]), f"row{i}: response tokens mismatch"
        # prompt real tokens (right side after left-pad) == original prompt
        got_prompt = batch["prompts"][i][P - pl:]
        assert torch.equal(got_prompt, orig[:pl]), f"row{i}: prompt tokens mismatch"
        # advantage 0 off-response, scalar on-response
        a = float(df.iloc[ix]["advantage"])
        adv = batch["advantages"][i]
        assert (adv[~rm] == 0).all() and (adv[rm] == a).all(), f"row{i}: advantage layout wrong"
        # full input_ids = [left-pad prompt | right-pad response]
        assert torch.equal(batch["input_ids"][i][:P], batch["prompts"][i])
        assert torch.equal(batch["input_ids"][i][P:], batch["responses"][i])

    print("3b-2 verl-layout test PASSED")


if __name__ == "__main__":
    main()
