# weight_sync (7.3d-0 spike)

How to push a new full-FT checkpoint from the trainer to the rollout vLLM.
Spike findings that inform the 7.3d trainer bridge design.

Design context: [report/007 §权重同步](../../../../report/007-rl-framework.md).
Progress: [report/011 §7.3d-0](../../../../report/011-rl-dev-progress.md).

## Findings (vLLM 0.16.0, env1)

Probe with `python -m vln.rl.weight_sync.probe_vllm`.

1. **standalone `vllm serve` exposes NO weight-update HTTP endpoint** — only
   `/load` (LoRA adapter). No `/update_weights`, no dev-mode weight routes.
   (Matches vLLM issue #12774 "load weights from disk via HTTP" → closed
   not-planned.)
2. The weight-transfer primitives exist only on the **in-process `LLM`
   object**: `init_weight_transfer_engine`, `update_weights`, `collective_rpc`,
   `sleep`, `wake_up`. These are built for a **colocated trainer pushing over
   NCCL/IPC**, not for a separate HTTP server.

## Working path for the decoupled architecture: restart

Kill `vllm serve`, relaunch from the new ckpt dir.

- Measured cold-start: **~130s** (TP=2, Qwen3-VL-4B, includes torch.compile +
  cudagraph capture).
- Effect verified: greedy `temperature=0` on a fixed prompt diverges before vs
  after restarting from a different ckpt (base Instruct vs vln-merged) → the
  server really serves the new weights.

**Operational caveat**: `pkill -f "vllm serve"` does NOT kill the EngineCore /
Worker_TP subprocesses; they leak GPU memory. Kill the worker PIDs from
`nvidia-smi --query-compute-apps=gpu_uuid,pid,used_memory` before relaunching,
or the next launch fails with "Free memory ... less than desired".

## Path decision for 7.3d (open)

| Path | Sync cost | Trade-off |
|---|---|---|
| A. standalone serve + restart | ~130s/step (maybe ~40-60s with `--enforce-eager`, untested) | simple, reuses 7.3a-c rollout_client unchanged |
| B. verl native vLLM hybrid engine (colocated NCCL) | <1s/step | rollout must target verl's vLLM, not our standalone serve |

130s/step is heavy relative to a 100-300s train step. Recommend evaluating B
first in 7.3d (verl already implements NCCL weight transfer for its colocated
vLLM); fall back to A if colocation conflicts with the decoupled rollout design.

## Files

- `probe_vllm.py` — prints vLLM version, server routes, and in-process
  weight-sync method names.
