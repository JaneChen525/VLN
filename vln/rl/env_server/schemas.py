from typing import Optional

from pydantic import BaseModel


class ResetRequest(BaseModel):
    episode_id: str
    trial_id: int = 0


class PublicObs(BaseModel):
    rgb_jpeg_b64: str
    instruction: str


class ResetResponse(BaseModel):
    env_id: str
    scene_id: str
    obs: PublicObs


class StepRequest(BaseModel):
    env_id: str
    action: int  # 0=stop, 1=forward, 2=turn_left, 3=turn_right


class StepResponse(BaseModel):
    obs: PublicObs
    done: bool
    success: float
    spl: float
    oracle_success: float
    ne: float
    step_count: int
    collisions: int


class PrivateResponse(BaseModel):
    """Eval metadata, must NOT be exposed to prompt builder."""
    env_id: str
    episode_id: str
    scene_id: str
    gt_path_len: float
    oracle_success: float
    final_ne: Optional[float] = None


class WorkerStatus(BaseModel):
    worker_id: int
    state: str  # "idle" | "busy" | "dead"
    env_id: Optional[str] = None
    episode_id: Optional[str] = None
    last_heartbeat_s: float


class HealthzResponse(BaseModel):
    ok: bool
    pool_size: int
    workers: list[WorkerStatus]


class EpisodesResponse(BaseModel):
    episode_ids: list[str]
