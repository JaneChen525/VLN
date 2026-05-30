"""3a: turn rollout parquet rows into a padded batch dict for the verl FSDP
engine. Runs in verl-dev docker (needs torch, processor, verl qwen3_vl rope).

Per sample we extend the Phase-2 tokenization with:
  - position_ids: 3-D Qwen3-VL mrope ids via verl's get_rope_index (reused, not
    reimplemented).
  - response_mask: == loss_mask (assistant token span).
  - token_advantages: the trajectory scalar advantage (Phase 1) broadcast to
    every response token; 0 over the prompt.
Multimodal patch tensors stay per-sample in `multi_modal_inputs` (object array),
matching verl's collate convention.
"""
import numpy as np
import torch

from vln.rl.trainer.tokenize import build_training_example


def make_example(row, frames_dir: str, processor, advantage: float) -> dict:
    from verl.models.transformers.qwen3_vl import get_rope_index

    ex = build_training_example(row, frames_dir, processor)
    input_ids = ex["input_ids"]
    attention_mask = ex["attention_mask"]
    loss_mask = ex["loss_mask"]

    grid = ex.get("image_grid_thw")
    position_ids = get_rope_index(
        processor, input_ids=input_ids, image_grid_thw=grid, attention_mask=attention_mask
    )  # [3, seq]

    token_adv = loss_mask.to(torch.float32) * float(advantage)  # broadcast scalar to response tokens

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "position_ids": position_ids,
        "response_mask": loss_mask.clone(),
        "advantages": token_adv,
        "multi_modal_inputs": {
            "pixel_values": ex["pixel_values"],
            "image_grid_thw": grid,
        },
        "prompt_len": ex["prompt_len"],
        "response_len": ex["response_len"],
    }


def _pad_1d(t, length, value):
    if t.shape[-1] >= length:
        return t[..., :length]
    pad = length - t.shape[-1]
    return torch.nn.functional.pad(t, (0, pad), value=value)


def collate_batch(examples: list[dict], pad_token_id: int) -> dict:
    """Right-pad to the batch max length. Tensor fields stacked [B, L]
    (position_ids [B, 3, L]); multi_modal_inputs kept as object array [B]."""
    max_len = max(e["input_ids"].shape[-1] for e in examples)

    input_ids, attn, resp_mask, adv, pos = [], [], [], [], []
    for e in examples:
        input_ids.append(_pad_1d(e["input_ids"], max_len, pad_token_id))
        attn.append(_pad_1d(e["attention_mask"], max_len, 0))
        resp_mask.append(_pad_1d(e["response_mask"], max_len, 0))
        adv.append(_pad_1d(e["advantages"], max_len, 0.0))
        pos.append(_pad_1d(e["position_ids"], max_len, 1))  # [3, L]

    mm = np.empty(len(examples), dtype=object)
    for i, e in enumerate(examples):
        mm[i] = e["multi_modal_inputs"]

    return {
        "input_ids": torch.stack(input_ids),            # [B, L]
        "attention_mask": torch.stack(attn),            # [B, L]
        "response_mask": torch.stack(resp_mask),        # [B, L]
        "advantages": torch.stack(adv),                 # [B, L]
        "position_ids": torch.stack(pos),               # [B, 3, L]
        "multi_modal_inputs": mm,                        # object[B]
    }
