"""NaVIDA stateless prompt for ONE decision (016 §9, matches eval_vllm_navida.py).

Each decision rebuilds a fresh [system, user(8 sampled history frames + current
frame + instruction)] — the model never sees its own past replies, so the train
prompt == the rollout prompt. Returns verl multimodal markers ({"type":"image"})
plus the ordered PIL frames to pass to apply_chat_template(images=...).
"""
from vln.rl.messages import (
    NAVIDA_BRIDGE,
    NAVIDA_INTRO,
    SYSTEM_PROMPT,
    navida_tail,
    uniform_sample_with_ends,
)

K_HISTORY = 8


def build_navida_messages(instruction: str, rgb_history: list, k_history: int = K_HISTORY):
    """rgb_history[-1] is the current frame; rgb_history[:-1] is past observations.

    Returns (messages, images). Image order = [*sampled_history, current], matching
    eval (uniform_sample_with_ends over history, current frame always last).
    """
    current = rgb_history[-1]
    historic = uniform_sample_with_ends(rgb_history[:-1], k_history) if len(rgb_history) > 1 else [current]
    images = list(historic) + [current]
    content = [{"type": "text", "text": NAVIDA_INTRO}]
    content.extend({"type": "image"} for _ in historic)
    content.append({"type": "text", "text": NAVIDA_BRIDGE})
    content.append({"type": "image"})  # current observation
    content.append({"type": "text", "text": navida_tail(instruction)})
    messages = [
        {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
        {"role": "user", "content": content},
    ]
    return messages, images


def build_navida_messages_b64(instruction: str, b64_buffer: list, k_history: int = K_HISTORY):
    """Standalone (b64-API) NaVIDA prompt — forwards env_server JPEG b64 DIRECTLY,
    no decode/re-encode (matches V1 agent.py + eval byte-for-byte; avoids double-JPEG).

    b64_buffer[-1] = current frame, b64_buffer[:-1] = history. Returns (messages, selected_ids).
    """
    current = b64_buffer[-1]
    hist = b64_buffer[:-1]
    if len(b64_buffer) > 1:
        if len(hist) <= k_history:
            idxs = list(range(len(hist)))
        else:
            idxs = [round(i * (len(hist) - 1) / (k_history - 1)) for i in range(k_history)]
    else:
        idxs = []
        hist = []
    historic = [hist[i] for i in idxs]
    content = [{"type": "text", "text": NAVIDA_INTRO}]
    content.extend({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b}"}} for b in historic)
    content.append({"type": "text", "text": NAVIDA_BRIDGE})
    content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{current}"}})
    content.append({"type": "text", "text": navida_tail(instruction)})
    messages = [
        {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
        {"role": "user", "content": content},
    ]
    return messages, idxs + [len(b64_buffer) - 1]
