"""Reconstruct the exact multimodal prompt from a rollout parquet row + frame
sidecar, then build (input_ids, pixel_values, image_grid_thw, loss_mask) for the
verl actor update. Runs in the verl-dev docker (Qwen3VLProcessor, transformers).

Message order matches the rollout agent exactly (shared text constants in
vln.rl.messages): [INTRO, *history_images, BRIDGE, current_image, tail].
The trainer loads frames as PIL from <frames_dir>/<sha256>.jpg (manifest order:
history first, current last).
"""
import json
import os

from PIL import Image

from vln.rl.messages import NAVIDA_BRIDGE, NAVIDA_INTRO, SYSTEM_PROMPT, navida_tail


def reconstruct_messages(row, frames_dir: str) -> list:
    """Rebuild OpenAI-style chat messages with PIL images for the processor.

    image_manifest is ordered [*history, current]; selected_frame_ids matches.
    """
    manifest = json.loads(row["image_manifest"])
    imgs = [Image.open(os.path.join(frames_dir, m["sha256"] + ".jpg")).convert("RGB") for m in manifest]
    history, current = imgs[:-1], imgs[-1]

    content = [{"type": "text", "text": NAVIDA_INTRO}]
    content += [{"type": "image", "image": im} for im in history]
    content.append({"type": "text", "text": NAVIDA_BRIDGE})
    content.append({"type": "image", "image": current})
    content.append({"type": "text", "text": navida_tail(row["instruction"])})

    return [
        {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
        {"role": "user", "content": content},
    ]


def build_training_example(row, frames_dir: str, processor) -> dict:
    """Return dict with input_ids, attention_mask, pixel_values, image_grid_thw,
    loss_mask (1 over response tokens incl. eos, 0 over prompt). Tensors are 1-D
    / unbatched so the collator can pad a batch.
    """
    import torch

    messages = reconstruct_messages(row, frames_dir)

    # Prompt = templated chat up to (and including) the generation prompt.
    prompt = processor.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=True,
        return_dict=True, return_tensors="pt",
    )
    prompt_ids = prompt["input_ids"][0]

    # Response = the sampled assistant text + eos.
    tok = processor.tokenizer
    resp_ids = tok(row["response_text"], add_special_tokens=False, return_tensors="pt")["input_ids"][0]
    eos = torch.tensor([tok.eos_token_id], dtype=resp_ids.dtype)
    resp_ids = torch.cat([resp_ids, eos])

    input_ids = torch.cat([prompt_ids, resp_ids])
    attention_mask = torch.ones_like(input_ids)
    loss_mask = torch.cat([torch.zeros_like(prompt_ids), torch.ones_like(resp_ids)])

    out = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "loss_mask": loss_mask,
        "prompt_len": int(prompt_ids.shape[0]),
        "response_len": int(resp_ids.shape[0]),
    }
    if "pixel_values" in prompt:
        out["pixel_values"] = prompt["pixel_values"]
    if "image_grid_thw" in prompt:
        out["image_grid_thw"] = prompt["image_grid_thw"]
    return out
