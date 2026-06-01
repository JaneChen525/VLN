# trainer

Turns rollout parquet (+ frame sidecar) into a verl GRPO update. Decoupled
design: the rollout side produces data, this side trains. Reuses verl's FSDP
engine + ppo_loss; we own grouping/advantage, tokenization, and the data layout.

Architecture: [report/007 §7.3d](../../../../report/007-rl-framework.md).
Progress / phase breakdown: [report/011 §7.3d](../../../../report/011-rl-dev-progress.md).

Runs in the **verl-dev docker** on env1 (verl 0.8.0.dev / vllm 0.11 / torch 2.8,
transformers 4.57). Plain pandas/numpy tests (advantage) run anywhere.

## Files

| File | Role |
|---|---|
| `advantage.py` | GRPO grouping + advantage from parquet. `compute_grpo_advantages` (group by `group_id`, NOT trial_id; within-group `(R-mean)/(std+eps)`; broadcast to turns) + `assert_single_policy_version` (`MixedPolicyError`) + `GroupStats`. Pure pandas/numpy. |
| `tokenize.py` | `reconstruct_messages` (rebuild agent's chat with PIL frames from `<frames_dir>/<sha>.jpg`) + `build_training_example` (input_ids / loss_mask / pixel_values / image_grid_thw via Qwen3VLProcessor). |
| `collate.py` | `make_example` (per-sample fields) + `to_verl_batch` (verl RL layout: prompts LEFT-pad, responses RIGHT-pad, response-shaped advantages/response_mask, 3-D mrope position_ids via verl `get_rope_index`; multimodal as per-sample object array). |
| `smoke_engine.py` | `build_config(ckpt, fsdp_size)` + standalone Ray `TrainingWorker` smoke (random text): `infer_batch` → `train_batch`. Proves Ray/FSDP/config link. |
| `smoke_mm.py` | Full GRPO step on a real multimodal parquet batch: infer (old_log_probs) → inject advantages → ppo_loss train_batch + memory smoke. |
| `verify_frames.py` | Confirm every manifest frame ref resolves + is byte-identical. |
| `test_advantage.py` | 6 pure unit tests (pytest). |
| `test_tokenize.py` | token-alignment golden test (loss_mask 0/1, response round-trip, image count). |
| `test_verl_batch.py` | verl-layout test (response==loss_mask span, advantage layout). |

## Data contract (from rollout_client)

- parquet rows (turn-level) with v1 schema: `group_id` / `trajectory_id` /
  `policy_version` / `reward_success` / `prompt_text_rendered` / `response_text` /
  `image_manifest` (sha256 per frame) ...
- frame sidecar `<out>/frames/<sha256>.jpg` (dedup); trainer loads frames by
  manifest sha.

## Pipeline

```
parquet ─► compute_grpo_advantages         (advantage.py, per-trajectory scalar)
        ─► make_example + build_training_example  (collate.py + tokenize.py)
        ─► to_verl_batch                    (verl prompts/responses/response_mask/
                                             advantages + mrope position_ids + mm)
        ─► DataProto ─► infer_batch (old_log_probs) ─► train_batch(ppo_loss)
```

GRPO group = `(split, episode_id, variant_hash)`, G = pass_k trajectories.
Advantage (scalar per trajectory) is response-shaped and fed to verl `ppo_loss`
(vanilla PPO clip). KL off (`use_kl_loss=False`) for sanity.

## Run (env1, verl-dev docker, 4 GPU)

```bash
# free GPUs first; container NVML can drop -> docker restart verl-dev
docker exec verl-dev bash -lc "cd /workspace/WorldModel && PYTHONPATH=. \
  CUDA_VISIBLE_DEVICES=0,1,2,3 VLN_FSDP_SIZE=4 \
  VLN_CKPT=/workspace/WorldModel/checkpoints/Qwen3-VL-4B-vln-r2r-merged \
  VLN_PARQUET=/workspace/rl_test/frames_test.parquet \
  VLN_FRAMES=/workspace/rl_test/frames \
  python -m vln.rl.trainer.smoke_mm"

# pure advantage tests (any python with pandas+pytest)
PYTHONPATH=. python -m pytest vln/rl/trainer/test_advantage.py -q
# token / layout tests (docker, needs processor + frames)
... python -m vln.rl.trainer.test_tokenize     # VLN_TEST_CKPT/PARQUET/FRAMES
... python -m vln.rl.trainer.test_verl_batch
```

## Config compatibility (env1 multi-GPU / env2 single-GPU)

`build_config(ckpt, fsdp_size)` is driven by `VLN_FSDP_SIZE`:
- **env1 (A100-40G): fsdp_size=4 required** for full-FT 4B. fsdp 1 and 2 OOM
  (params+grads+AdamW fp32 ~64GB); fsdp=4 peaks ~20.8GB/GPU.
- **env2 (GB200/300 big HBM): fsdp_size=1** works.
- `ulysses_sequence_parallel_size=1` both sides (4B needs no seq parallel).
- `max_token_len_per_gpu` must be ≥ real VL sequence length (9-frame prompts
  ~2.3k+ tokens; set 8192).
- batch size must be divisible by fsdp_size (dp ranks split the batch).

## Known limitations / TODO

- old_log_probs via pre-forward (`infer_batch`) — no vLLM logprobs stored.
- ckpt save + weight sync (scheme A: restart vLLM from ckpt) is Phase 4, not here.
- accuracy-band filter (`GroupStats` exposes the ratios) not wired into batch
  selection yet.
- single train step only; multi-step loop is 7.4.

## Tested (env1 4x A100-40G, fsdp=4, 2026-05-31)

| step | result |
|---|---|
| 3b-1 text smoke | loss=-0.40, peak 20.8GB/GPU |
| 3b-3 multimodal forward | log_probs (4,24) |
| 3b-4 full GRPO step | loss=0.71, grad_norm=9.6 |
| 3b-5 memory | peak 20.77GB/GPU < 40G |
