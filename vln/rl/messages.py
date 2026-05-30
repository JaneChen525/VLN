import base64
import re
from io import BytesIO

from PIL import Image


SYSTEM_PROMPT = "You are a helpful assistant."

PROMPT_TEMPLATE = (
    "Imagine you are a robot programmed for navigation tasks. "
    "You have been given a video of historical observations and an image of the current observation. "
    "Your assigned task is: '{}'. Analyze this series of images to decide your next move, "
    "which could involve turning left or right by a specific degree or moving forward a certain distance."
)

# NaVIDA prompt text pieces. The content order is fixed:
#   [INTRO, *history_images, BRIDGE, current_image, navida_tail(instruction)]
# Shared by the rollout agent (b64 image urls) and the trainer reconstruction
# (PIL images) so the two cannot drift apart.
NAVIDA_INTRO = "Imagine you are a robot programmed for navigation tasks. You have been given a video of historical observations"
NAVIDA_BRIDGE = "and an image of the current observation"


def navida_tail(instruction: str) -> str:
    return PROMPT_TEMPLATE.format(instruction).split("current observation")[1]


def encode_image_base64(image: Image.Image) -> str:
    buf = BytesIO()
    image.save(buf, format="JPEG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def uniform_sample_with_ends(data: list, n: int) -> list:
    if len(data) <= n:
        return data
    indices = [round(i * (len(data) - 1) / (n - 1)) for i in range(n)]
    return [data[i] for i in indices]


def build_messages(instruction: str, rgb_history: list, k_history: int = 8) -> list:
    # rgb_history: list[PIL.Image], full episode history including current frame at [-1]
    current = rgb_history[-1]
    historic = uniform_sample_with_ends(rgb_history[:-1], k_history) if len(rgb_history) > 1 else [current]
    template_tail = PROMPT_TEMPLATE.format(instruction).split("current observation")[1]

    content = [{"type": "text", "text": "Imagine you are a robot programmed for navigation tasks. You have been given a video of historical observations"}]
    content.extend(
        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{encode_image_base64(im)}"}}
        for im in historic
    )
    content.append({"type": "text", "text": "and an image of the current observation"})
    content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{encode_image_base64(current)}"}})
    content.append({"type": "text", "text": template_tail})

    return [
        {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
        {"role": "user", "content": content},
    ]


def parse_actions(output_text: str, forward_distance: int = 25, turn_angle: int = 15, n: int = 2):
    # Return list[(action_id, numeric)] for the first up to n sub-actions.
    # action_id: 0=stop, 1=forward, 2=turn left, 3=turn right.
    # n=2 mirrors NaVIDA baseline + m1/m4 variants (select_action_idx=2 / [:2]).
    match = re.search(r"<answer>(.*?)</answer>", output_text)
    body = match.group(1).strip() if match else output_text.strip()
    sub_actions = body.split(", ")
    results = []
    for sa in sub_actions[:n]:
        results.append(_parse_one(sa, forward_distance, turn_angle))
    return results


def _parse_one(text: str, forward_distance: int, turn_angle: int):
    text = text.lower()
    if "stop" in text:
        return (0, None)
    if "forward" in text:
        m = re.search(r"-?\d+", text)
        return (1, float(m.group()) if m else float(forward_distance))
    if "left" in text:
        m = re.search(r"-?\d+", text)
        return (2, float(m.group()) if m else float(turn_angle))
    if "right" in text:
        m = re.search(r"-?\d+", text)
        return (3, float(m.group()) if m else float(turn_angle))
    return (None, None)


def expand_to_atomic(action_id: int, numeric, forward_distance: int, turn_angle: int, max_repeat: int = 3) -> list:
    # Convert one parsed (action_id, numeric) into the atomic step actions Habitat consumes.
    if action_id == 0:
        return [0]
    if action_id == 1:
        return [1] * min(max_repeat, round(numeric / forward_distance))
    if action_id == 2:
        return [2] * min(max_repeat, round(numeric / turn_angle))
    if action_id == 3:
        return [3] * min(max_repeat, round(numeric / turn_angle))
    return []
