"""FastAPI front + worker process pool for habitat env.

Each worker is one process holding a habitat.Env. Concurrent requests for
distinct env_ids hit distinct workers (sync endpoints run in a threadpool);
pool state is guarded by a lock. Set pool size via launch --pool-size.

Endpoints:
  POST   /reset    -> ResetResponse        (public obs only)
  POST   /step     -> StepResponse
  GET    /private/{env_id}  -> PrivateResponse  (gt info, separated from prompt path)
  DELETE /env/{env_id}      -> {"ok": True}     (idempotent)
  GET    /healthz           -> HealthzResponse
"""
import multiprocessing as mp
import threading
import time
import uuid
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, HTTPException

from vln.rl.env_server.schemas import (
    EpisodesResponse,
    HealthzResponse,
    PrivateResponse,
    PublicObs,
    ResetRequest,
    ResetResponse,
    StepRequest,
    StepResponse,
    WorkerStatus,
)
from vln.rl.env_server.worker import worker_loop


class WorkerHandle:
    def __init__(self, worker_id: int, exp_config_path: str):
        self.worker_id = worker_id
        ctx = mp.get_context("spawn")
        self.cmd_q: mp.Queue = ctx.Queue()
        self.resp_q: mp.Queue = ctx.Queue()
        self.heartbeat = ctx.Value("d", 0.0)
        self.proc = ctx.Process(
            target=worker_loop,
            args=(worker_id, exp_config_path, self.cmd_q, self.resp_q, self.heartbeat),
            daemon=True,
        )
        self.proc.start()
        # Wait for ready ack
        ready = self.resp_q.get(timeout=300)
        if not ready.get("ok"):
            raise RuntimeError(f"worker {worker_id} failed to start: {ready.get('error')}")
        self.state: str = "idle"  # idle | busy | dead
        self.env_id: Optional[str] = None
        self.episode_id: Optional[str] = None

    def call(self, cmd: dict, timeout: float = 60.0) -> dict:
        self.cmd_q.put(cmd)
        return self.resp_q.get(timeout=timeout)

    def close(self):
        try:
            self.cmd_q.put({"op": "close"})
            self.proc.join(timeout=10)
        except Exception:
            pass
        if self.proc.is_alive():
            self.proc.terminate()

    def is_alive(self) -> bool:
        return self.proc.is_alive()


class WorkerPool:
    def __init__(self, exp_config_path: str, pool_size: int = 1):
        self.exp_config_path = exp_config_path
        self.pool_size = pool_size
        self.workers: list[WorkerHandle] = []
        for i in range(pool_size):
            self.workers.append(WorkerHandle(i, exp_config_path))
        self.env_id_to_worker: dict[str, WorkerHandle] = {}
        self._lock = threading.Lock()  # sync endpoints run in a threadpool; guard pool state

    def claim(self, env_id: str, episode_id: str) -> Optional[WorkerHandle]:
        """Atomically pick an idle worker, mark it busy, register env_id."""
        with self._lock:
            for w in self.workers:
                if w.state == "idle" and w.is_alive():
                    w.state = "busy"
                    w.env_id = env_id
                    w.episode_id = episode_id
                    self.env_id_to_worker[env_id] = w
                    return w
            return None

    def get_by_env(self, env_id: str) -> Optional[WorkerHandle]:
        return self.env_id_to_worker.get(env_id)

    def release(self, env_id: str):
        with self._lock:
            w = self.env_id_to_worker.pop(env_id, None)
            if w is not None:
                w.state = "idle"
                w.env_id = None
                w.episode_id = None

    def close(self):
        for w in self.workers:
            w.close()


# Application state — set in lifespan
_pool: Optional[WorkerPool] = None
_exp_config_path: Optional[str] = None
_pool_size: int = 1


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _pool
    assert _exp_config_path is not None, "set _exp_config_path before starting uvicorn"
    _pool = WorkerPool(_exp_config_path, _pool_size)
    yield
    if _pool is not None:
        _pool.close()


app = FastAPI(lifespan=lifespan)


@app.post("/reset", response_model=ResetResponse)
def reset(req: ResetRequest):
    assert _pool is not None
    env_id = f"env_{uuid.uuid4().hex[:12]}"
    w = _pool.claim(env_id, req.episode_id)
    if w is None:
        raise HTTPException(status_code=503, detail="no idle worker")
    try:
        resp = w.call({"op": "reset", "episode_id": req.episode_id}, timeout=120)
    except Exception:
        _pool.release(env_id)
        raise
    if not resp.get("ok"):
        _pool.release(env_id)
        raise HTTPException(status_code=400, detail=resp.get("error", "reset failed"))
    return ResetResponse(
        env_id=env_id,
        scene_id=resp["scene_id"],
        obs=PublicObs(rgb_jpeg_b64=resp["rgb_jpeg_b64"], instruction=resp["instruction"]),
    )


@app.post("/step", response_model=StepResponse)
def step(req: StepRequest):
    assert _pool is not None
    w = _pool.get_by_env(req.env_id)
    if w is None:
        raise HTTPException(status_code=404, detail=f"env_id {req.env_id} not found")
    resp = w.call({"op": "step", "action": req.action}, timeout=60)
    if not resp.get("ok"):
        raise HTTPException(status_code=500, detail=resp.get("error", "step failed"))
    return StepResponse(
        obs=PublicObs(rgb_jpeg_b64=resp["rgb_jpeg_b64"], instruction=resp["instruction"]),
        done=resp["done"],
        success=resp["success"],
        spl=resp["spl"],
        oracle_success=resp["oracle_success"],
        ne=resp["ne"],
        step_count=resp["step_count"],
        collisions=resp["collisions"],
    )


@app.get("/private/{env_id}", response_model=PrivateResponse)
def private(env_id: str):
    assert _pool is not None
    w = _pool.get_by_env(env_id)
    if w is None:
        raise HTTPException(status_code=404, detail=f"env_id {env_id} not found")
    resp = w.call({"op": "private"}, timeout=10)
    if not resp.get("ok"):
        raise HTTPException(status_code=500, detail=resp.get("error", "private failed"))
    return PrivateResponse(
        env_id=env_id,
        episode_id=resp["episode_id"],
        scene_id=resp["scene_id"],
        gt_path_len=resp["gt_path_len"],
        oracle_success=resp["oracle_success"],
        final_ne=resp.get("final_ne"),
    )


@app.delete("/env/{env_id}")
def delete_env(env_id: str):
    """Idempotent — releases worker slot, env state cleared on next /reset."""
    assert _pool is not None
    _pool.release(env_id)
    return {"ok": True}


@app.get("/episodes", response_model=EpisodesResponse)
def episodes(limit: int = 100):
    assert _pool is not None
    resp = _pool.workers[0].call({"op": "episodes", "limit": limit}, timeout=10)
    if not resp.get("ok"):
        raise HTTPException(status_code=500, detail=resp.get("error", "episodes failed"))
    return EpisodesResponse(episode_ids=resp["episode_ids"])


@app.get("/healthz", response_model=HealthzResponse)
def healthz():
    assert _pool is not None
    now = time.time()
    workers = []
    all_ok = True
    for w in _pool.workers:
        alive = w.is_alive()
        state = w.state if alive else "dead"
        if not alive:
            all_ok = False
        workers.append(
            WorkerStatus(
                worker_id=w.worker_id,
                state=state,
                env_id=w.env_id,
                episode_id=w.episode_id,
                last_heartbeat_s=now - w.heartbeat.value if w.heartbeat.value else -1.0,
            )
        )
    return HealthzResponse(ok=all_ok, pool_size=_pool.pool_size, workers=workers)


def set_config(exp_config_path: str, pool_size: int = 1):
    """Call before running uvicorn to inject habitat config + pool size."""
    global _exp_config_path, _pool_size
    _exp_config_path = exp_config_path
    _pool_size = pool_size
