# env_server

Habitat env as an HTTP service. Decouples env from the rollout process so vLLM /
trainer / habitat run in separate CUDA contexts and can be scaled / restarted
independently.

Architecture in [report/007 §架构总览](../../../../report/007-rl-framework.md). Status
tracking in [report/011 §7.3a](../../../../report/011-rl-dev-progress.md).

## Files

| File | Role |
|---|---|
| `schemas.py` | Pydantic request/response models. `PublicObs` (RGB+instruction) and `PrivateResponse` (gt_path etc.) are deliberately separate — prompt builder must never see private fields. |
| `worker.py` | Subprocess worker holding one `habitat.Env` + EGL context. IPC: `mp.Queue` for commands and responses. Operations: `reset` / `step` / `private` / `close`. Heartbeat via `mp.Value`. |
| `server.py` | FastAPI app + `WorkerPool` (size = 1 in 7.3a, ≥4 in 7.3c) + `WorkerHandle`. Lifespan-managed pool startup/teardown. |
| `launch.py` | uvicorn entrypoint. |
| `client.py` | `EnvClient` httpx wrapper used by rollout + parity tests. |
| `parity_test.py` | Random-action parity (env behavior + JPEG bytes). |
| `parity_test_expert.py` | Expert-action parity using DAgger/SFT GT trajectories. Stronger: verifies goal-reaching trajectories byte-identical end-to-end. |

## HTTP API

| Method | Path | Returns |
|---|---|---|
| POST | `/reset` | `ResetResponse` — `env_id`, `scene_id`, public obs (RGB JPEG b64 + instruction). **No GT info.** |
| POST | `/step` | `StepResponse` — public obs + `done`, `success`, `spl`, `ne`, `oracle_success`, `step_count`, `collisions`. |
| GET | `/private/{env_id}` | `PrivateResponse` — `gt_path_len`, `oracle_success`, `final_ne`. **Eval metadata only, not for prompt.** |
| DELETE | `/env/{env_id}` | Idempotent release of worker slot. |
| GET | `/healthz` | Pool size + per-worker state + last heartbeat seconds. |

Schemas in [schemas.py](schemas.py).

## Image contract (critical for model-input parity)

- **Sensor**: 640×480 RGB, hfov from `config/vln_r2r.yaml` (default 79).
- **Encoding**: `PIL.Image.save(format="JPEG")` with default quality (75). Must
  match [`vln/rl/messages.py:encode_image_base64`](../messages.py) and
  [`vln/eval_vllm_navida.py:encode_image_base64`](../../eval_vllm_navida.py) so
  the model sees byte-identical JPEG whether obs comes from env_server or a
  direct `habitat.Env`. **Do not pass `quality=` kwarg in [worker.py](worker.py).**
- **Scene cache**: worker keeps `habitat.Env` alive across episodes; habitat
  internally reuses the loaded GLB scene. `pass_k=4` reset of the same episode
  costs no scene reload.

## Launch (env1, GPU 6)

```bash
cd /var/data0/sandbox/janec/WorldModel
source /var/data0/sandbox/janec/miniconda3/bin/activate vln
CUDA_VISIBLE_DEVICES=6 \
  PYTHONPATH=/var/data0/sandbox/janec/WorldModel:/var/data0/sandbox/janec/WorldModel/vln \
  python -m vln.rl.env_server.launch \
    --exp-config /var/data0/sandbox/janec/WorldModel/config/vln_r2r.yaml \
    --port 8002 --pool-size 1
```

Healthcheck: `curl localhost:8002/healthz`.

## Parity tests (M1 verification)

### Random actions (sanity)

```bash
CUDA_VISIBLE_DEVICES=7 PYTHONPATH=... python -m vln.rl.env_server.parity_test \
  --exp-config .../config/vln_r2r.yaml \
  --server-url http://localhost:8002 \
  --num-episodes 2 --num-steps 30
```

Expects: `success_eq`, `spl_abs_diff < 1e-4`, `ne_abs_diff < 1e-3`,
`jpeg_all_eq=True` per episode.

### Expert (SFT/DAgger) actions

```bash
CUDA_VISIBLE_DEVICES=7 PYTHONPATH=... python -m vln.rl.env_server.parity_test_expert \
  --exp-config .../config/vln_r2r.yaml \
  --gt-path .../data/vln_eval_datasets/r2r/val_unseen_sample33/val_unseen_sample33_gt.json.gz \
  --server-url http://localhost:8002 \
  --num-episodes 5
```

Expects: `direct_success=1.0` (expert reaches goal) + same metric/JPEG byte
equality as random test, over real DAgger atomic action sequences (typical
length 29–73 actions per episode).

GT format: `gt[trajectory_id]` → `{"actions": [int…], "locations": […], "forward_steps": int}`.
Episode → trajectory_id via `ep.trajectory_id`.

## Known limitations / TODO

- Pool size = 1. Multi-worker pool + async client = 7.3c (target ≥ 2.5× throughput).
- No worker auto-restart on crash. With pool=1 a crash returns 5xx; client retry
  policy TBD in 7.3c.
- No env lease TTL / heartbeat enforcement on client side. Server tracks
  `last_heartbeat_s` in healthz but doesn't reclaim stale slots yet.
- `/private` returns `gt_path_len` from `episode.info.geodesic_distance`. If
  more eval metadata is needed (action_history, oracle path), extend
  `PrivateResponse` + worker `op=private`.

## Tested results (2026-05-28, env1 A100-40G GPU 6)

| Test | Episodes | Steps total | JPEG SHA256 eq | env metrics eq |
|---|---|---|---|---|
| Random 30-step | 2 | 64 | 64/64 | ✅ |
| Expert DAgger | 5 | 291 | 291/291 | ✅ |

All expert trajectories reach goal (5/5 success). End-to-end parity with
direct `habitat.Env` verified for both random and expert action regimes.
