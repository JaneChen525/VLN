import hashlib
import json
from dataclasses import asdict, dataclass, field


@dataclass
class FrameSelectionConfig:
    strategy: str = "uniform_sample_with_ends"
    max_frames: int = 8


@dataclass
class FrameCombineConfig:
    mode: str = "separate"
    grid_cols: int = 4


@dataclass
class ActionPackingConfig:
    # NaVIDA baseline + m1/m2/m4 variants all hardcode 2; see eval_vllm_navida*.py.
    actions_per_turn: int = 2


@dataclass
class RolloutConfig:
    frame_selection: FrameSelectionConfig = field(default_factory=FrameSelectionConfig)
    frame_combine: FrameCombineConfig = field(default_factory=FrameCombineConfig)
    action_packing: ActionPackingConfig = field(default_factory=ActionPackingConfig)
    forward_distance: int = 25
    turn_angle: int = 15
    temperature: float = 0.7
    top_p: float = 0.95
    max_tokens: int = 512
    early_stop_rotation: int = 25
    early_stop_steps: int = 400


def config_hash(cfg: RolloutConfig) -> str:
    payload = json.dumps(asdict(cfg), sort_keys=True).encode()
    return hashlib.sha256(payload).hexdigest()[:16]
