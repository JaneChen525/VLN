"""Turn rollout parquet rows into a verl DataProto batch. Runs in verl-dev
docker (torch, processor, verl qwen3_vl rope).

make_example: per-sample fields (input_ids, prompt/response lengths, scalar
advantage, multimodal patches) from Phase-2 tokenization.
to_verl_batch: verl RL layout — prompts LEFT-padded, responses RIGHT-padded,
3-D Qwen3-VL mrope position_ids via verl's get_rope_index (reused). Multimodal
patches stay per-sample in `multi_modal_inputs` (object array).
"""
import numpy as np
import torch

from vln.rl.trainer.tokenize import build_training_example


def make_example(row, frames_dir: str, processor, advantage: float) -> dict:
    """Per-sample fields consumed by to_verl_batch. position_ids / response_mask
    are not built here — to_verl_batch recomputes them after padding."""
    ex = build_training_example(row, frames_dir, processor)
    return {
        "input_ids": ex["input_ids"],
        "prompt_len": ex["prompt_len"],
        "response_len": ex["response_len"],
        "advantage": float(advantage),
        "multi_modal_inputs": {"pixel_values": ex["pixel_values"], "image_grid_thw": ex.get("image_grid_thw")},
    }


def _pad_1d(t, length, value, left=False):
    pad = length - t.shape[-1]
    return torch.nn.functional.pad(t, (pad, 0) if left else (0, pad), value=value)


def to_verl_batch(examples: list[dict], pad_token_id: int, processor) -> dict:
    """verl RL layout: prompts LEFT-padded to max_prompt_len, responses
    RIGHT-padded to max_response_len, concatenated. Matches the DataProto
    `from_single_dict({input_ids, prompts, attention_mask, position_ids,
    responses, response_mask})` convention used by the engine.

    `responses`/`response_mask`/`advantages` are response-shaped [B, R]
    (not full-sequence), as verl's ppo_loss expects.
    """
    from verl.models.transformers.qwen3_vl import get_rope_index

    P = max(e["prompt_len"] for e in examples)
    R = max(e["response_len"] for e in examples)

    prompts, responses, resp_mask, adv, input_ids, attn, pos = [], [], [], [], [], [], []
    mm = np.empty(len(examples), dtype=object)

    for i, e in enumerate(examples):
        pl, rl = e["prompt_len"], e["response_len"]
        ids = e["input_ids"]
        p_ids, r_ids = ids[:pl], ids[pl:pl + rl]
        a = e["advantage"]  # per-trajectory scalar

        p_pad = _pad_1d(p_ids, P, pad_token_id, left=True)
        r_pad = _pad_1d(r_ids, R, pad_token_id, left=False)
        full = torch.cat([p_pad, r_pad])
        p_attn = _pad_1d(torch.ones(pl, dtype=torch.long), P, 0, left=True)
        r_attn = _pad_1d(torch.ones(rl, dtype=torch.long), R, 0, left=False)

        prompts.append(p_pad)
        responses.append(r_pad)
        resp_mask.append(r_attn.clone())
        adv.append(r_attn.to(torch.float32) * a)
        input_ids.append(full)
        full_attn = torch.cat([p_attn, r_attn])
        attn.append(full_attn)
        pos.append(get_rope_index(
            processor=processor, input_ids=full,
            image_grid_thw=e["multi_modal_inputs"].get("image_grid_thw"),
            attention_mask=full_attn,
        ))
        mm[i] = e["multi_modal_inputs"]

    return {
        "input_ids": torch.stack(input_ids),     # [B, P+R]
        "attention_mask": torch.stack(attn),      # [B, P+R]
        "position_ids": torch.stack(pos),         # [B, 3, P+R]
        "prompts": torch.stack(prompts),          # [B, P]
        "responses": torch.stack(responses),      # [B, R]
        "response_mask": torch.stack(resp_mask),  # [B, R]
        "advantages": torch.stack(adv),           # [B, R]
        "multi_modal_inputs": mm,                  # object[B]
    }
