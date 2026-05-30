"""Lightweight HTTP client used by rollout and parity tests."""
import base64
import io
from typing import Optional

import httpx
import numpy as np
from PIL import Image


def decode_jpeg_b64(b64: str) -> np.ndarray:
    img = Image.open(io.BytesIO(base64.b64decode(b64)))
    return np.array(img.convert("RGB"))


class EnvClient:
    def __init__(self, base_url: str, timeout: float = 120.0):
        self.base_url = base_url.rstrip("/")
        self.client = httpx.Client(base_url=self.base_url, timeout=timeout)

    def reset(self, episode_id: str, trial_id: int = 0) -> dict:
        r = self.client.post("/reset", json={"episode_id": episode_id, "trial_id": trial_id})
        r.raise_for_status()
        return r.json()

    def step(self, env_id: str, action: int) -> dict:
        r = self.client.post("/step", json={"env_id": env_id, "action": action})
        r.raise_for_status()
        return r.json()

    def private(self, env_id: str) -> dict:
        r = self.client.get(f"/private/{env_id}")
        r.raise_for_status()
        return r.json()

    def delete_env(self, env_id: str) -> None:
        self.client.delete(f"/env/{env_id}")

    def episodes(self, limit: int = 100) -> list[str]:
        r = self.client.get("/episodes", params={"limit": limit})
        r.raise_for_status()
        return r.json()["episode_ids"]

    def healthz(self) -> dict:
        r = self.client.get("/healthz")
        r.raise_for_status()
        return r.json()

    def close(self):
        self.client.close()


class AsyncEnvClient:
    def __init__(self, base_url: str, timeout: float = 120.0):
        self.client = httpx.AsyncClient(base_url=base_url.rstrip("/"), timeout=timeout)

    async def reset(self, episode_id: str, trial_id: int = 0) -> dict:
        r = await self.client.post("/reset", json={"episode_id": episode_id, "trial_id": trial_id})
        r.raise_for_status()
        return r.json()

    async def step(self, env_id: str, action: int) -> dict:
        r = await self.client.post("/step", json={"env_id": env_id, "action": action})
        r.raise_for_status()
        return r.json()

    async def delete_env(self, env_id: str) -> None:
        await self.client.delete(f"/env/{env_id}")

    async def episodes(self, limit: int = 100) -> list[str]:
        r = await self.client.get("/episodes", params={"limit": limit})
        r.raise_for_status()
        return r.json()["episode_ids"]

    async def aclose(self):
        await self.client.aclose()
