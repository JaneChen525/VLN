from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class FrameSelectionConfig:
    strategy: str = "uniform_sample_with_ends"
    max_frames: int = 8


class FrameSelector(ABC):
    @abstractmethod
    def select_indices(self, buffer_len: int, turn_idx: int) -> list[int]:
        """Return history indices into buffer[:-1]. Caller appends buffer[-1] as current."""


class UniformSampleWithEnds(FrameSelector):
    """NaVIDA default. Step-0 fallback (buffer_len=1) returns [0]."""
    def __init__(self, n: int = 8):
        self.n = n

    def select_indices(self, buffer_len: int, turn_idx: int) -> list[int]:
        if buffer_len <= 1:
            return [buffer_len - 1]
        hist_len = buffer_len - 1
        if hist_len <= self.n:
            return list(range(hist_len))
        return [round(i * (hist_len - 1) / (self.n - 1)) for i in range(self.n)]


class LastWindow(FrameSelector):
    def __init__(self, n: int = 8):
        self.n = n

    def select_indices(self, buffer_len: int, turn_idx: int) -> list[int]:
        raise NotImplementedError("LastWindow: ablation stub")


class AllFrames(FrameSelector):
    def select_indices(self, buffer_len: int, turn_idx: int) -> list[int]:
        raise NotImplementedError("AllFrames: ablation stub")


def build_selector(cfg: FrameSelectionConfig) -> FrameSelector:
    if cfg.strategy == "uniform_sample_with_ends":
        return UniformSampleWithEnds(n=cfg.max_frames)
    if cfg.strategy == "last_window":
        return LastWindow(n=cfg.max_frames)
    if cfg.strategy == "all":
        return AllFrames()
    raise ValueError(f"unknown frame_selection.strategy: {cfg.strategy}")
