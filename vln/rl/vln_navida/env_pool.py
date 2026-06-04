"""VLNEnv: a per-trajectory Habitat env handle (016 §8.2 VLNEnvPool role).

Thin async client over the env_server HTTP backend (reused from vln/rl/env_server):
reset() claims a server worker (env_id), step() advances it, close() releases it.
Habitat stays in the host conda env; the verl rollout worker only talks HTTP, so no
habitat install is needed inside the vLLM container.

One VLNEnv per trajectory. The env_server's own worker pool handles concurrency;
reset() can return 503 ("no idle worker") under load, so it retries with a bound.
"""
import asyncio

import httpx
import numpy as np

from vln.rl.env_server.client import AsyncEnvClient, decode_jpeg_b64


class VLNEnv:
    def __init__(self, base_url: str, reset_timeout_s: float = 600.0, reset_retry_s: float = 2.0,
                 timeout: float = 120.0):
        self._client = AsyncEnvClient(base_url, timeout=timeout)
        self._reset_timeout_s = reset_timeout_s
        self._reset_retry_s = reset_retry_s
        self.env_id: str | None = None
        self.instruction: str | None = None
        self._cur_b64: str | None = None   # raw env_server JPEG b64 (forward as-is, no re-encode)
        self._cur_rgb: np.ndarray | None = None
        self._last: dict | None = None  # last /step response

    async def reset(self, extra_info: dict) -> np.ndarray:
        """Start one trajectory. Returns the first RGB observation (HxWx3 uint8)."""
        episode_id = str(extra_info["episode_id"])
        trial_id = int(extra_info.get("trial_id", 0))
        deadline = asyncio.get_event_loop().time() + self._reset_timeout_s
        while True:
            try:
                r = await self._client.reset(episode_id, trial_id)
                break
            except httpx.HTTPStatusError as e:
                if e.response.status_code == 503 and asyncio.get_event_loop().time() < deadline:
                    await asyncio.sleep(self._reset_retry_s)
                    continue
                raise
        self.env_id = r["env_id"]
        self.instruction = r["obs"]["instruction"]
        self._cur_b64 = r["obs"]["rgb_jpeg_b64"]
        self._cur_rgb = None
        self._last = None
        return self._cur_b64

    async def step(self, action_id: int) -> dict:
        """Advance one atomic action. Returns the raw /step dict and updates current obs."""
        s = await self._client.step(self.env_id, action_id)
        self._cur_b64 = s["obs"]["rgb_jpeg_b64"]
        self._cur_rgb = None
        self._last = s
        return s

    def current_jpeg_b64(self) -> str:
        """Raw env_server JPEG b64 — forward directly to vLLM (no decode/re-encode)."""
        return self._cur_b64

    def current_rgb(self) -> np.ndarray:
        """Decoded RGB ndarray (lazy; only if a backend needs pixels, e.g. verl PIL path)."""
        if self._cur_rgb is None and self._cur_b64 is not None:
            self._cur_rgb = decode_jpeg_b64(self._cur_b64)
        return self._cur_rgb

    @property
    def done(self) -> bool:
        return bool(self._last["done"]) if self._last else False

    def metrics(self) -> dict:
        """Terminal/episode metrics from the last step (success/spl/ne/...)."""
        s = self._last or {}
        return {
            "success": float(s.get("success", 0.0)),
            "spl": float(s.get("spl", 0.0)),
            "ne": float(s.get("ne", 0.0)),
            "oracle_success": float(s.get("oracle_success", 0.0)),
            "step_count": int(s.get("step_count", 0)),
            "collisions": int(s.get("collisions", 0)),
        }

    async def close(self) -> None:
        try:
            if self.env_id is not None:
                await self._client.delete_env(self.env_id)
        finally:
            await self._client.aclose()
