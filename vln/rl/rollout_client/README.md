# rollout_client

Async rollout orchestrator. Drives `env_server` (habitat) + vLLM, packs frames
per a configurable `FrameSelector`, writes per-turn trajectory rows to parquet
for the verl trainer.

Architecture: [report/007 §7.3b](../../../../report/007-rl-framework.md). Status:
[report/011 §7.3b](../../../../report/011-rl-dev-progress.md).

## Files

| File | Role |
|---|---|
| `config.py` | `RolloutConfig` dataclasses (frame_selection / frame_combine / action_packing + sampling params) + `config_hash`. |
| `selectors.py` | `FrameSelector` ABC + `UniformSampleWithEnds` (NaVIDA default) + `LastWindow` / `AllFrames` stubs for future ablation. `build_selector(cfg)` dispatches. |
| `agent.py` | `AsyncRolloutAgent` + `AgentState`. Forwards env_server's JPEG b64 directly to vLLM (no decode/re-encode) so model input is byte-identical to direct habitat path. Builds NaVIDA prompt inline. |
| `rollout.py` | `run_episode` (async) + `trajectory_to_rows` (v1 schema). |
| `run_rollout.py` | CLI. Resolves vLLM model id, enumerates episodes via `/episodes`, loops episodes × pass_k, writes parquet. |

## Parquet schema (v1)

| Column | Why |
|---|---|
| `schema_version`, `rl_step`, `policy_version` | trainer demuxes by step + asserts single policy_version per batch (report/007 §策略版本化) |
| `group_id`, `trajectory_id`, `episode_id`, `trial_id`, `turn_idx`, `scene_id` | GRPO group = `(split, episode_id, variant_hash)`; trial_id is sample index inside group |
| `variant_hash`, `prompt_template_hash`, `rollout_sampling_config_hash` | invariants for batch correctness checks |
| `instruction`, `prompt_text_rendered`, `response_text`, `parsed_actions`, `parse_ok` | trainer collator + parse-fail audit |
| `selected_frame_ids`, `image_manifest` | which buffer indices fed this turn; manifest carries sha256 + byte count per image |
| `reward_success`, `success`, `spl`, `oracle_success`, `ne`, `env_steps`, `duration_s`, `done_reason`, `aborted` | terminal SR is the reward; rest are diagnostics |

Deferred to 7.3d (trainer integration): `vllm_token_ids`, `vllm_logprobs`,
`tokenizer_hash`, `processor_hash`, `num_image_tokens`, `image_preprocess_config_hash`.

## Frame buffer & prompt

- Buffer stores JPEG b64 strings as returned by env_server `/reset` and `/step` —
  not decoded PIL images. This guarantees model-input parity with a direct
  habitat pipeline (verified in env_server §parity_test).
- `UniformSampleWithEnds.select_indices(buffer_len, turn_idx)` returns history
  indices into `buffer[:-1]`. The agent appends `buffer[-1]` as the "current
  observation" image after the bridge text, matching NaVIDA's prompt layout.
- Step-0 fallback (buffer has only the current frame): selector returns `[0]`
  so the history slot is filled with the current frame.

## Launch (env1)

Prereq: vLLM serve on port 8001 (LoRA-merged ckpt), env_server on port 8002
(see `vln/rl/env_server/README.md`).

```bash
cd /var/data0/sandbox/janec/WorldModel
source /var/data0/sandbox/janec/miniconda3/bin/activate vln
PYTHONPATH=/var/data0/sandbox/janec/WorldModel:/var/data0/sandbox/janec/WorldModel/vln \
  python -m vln.rl.rollout_client.run_rollout \
    --env-server http://localhost:8002 \
    --vllm-base http://localhost:8001/v1 \
    --split val_unseen_sample33 \
    --policy-version sft-hfov79-merged \
    --num-episodes 2 --pass-k 4 \
    --out /tmp/rollout_v1.parquet
```

## Adding a new ablation variant

1. Add an entry to `frame_selection.strategy` accepted values in `selectors.py`
   (`build_selector` dispatch + concrete `FrameSelector` subclass).
2. If the variant needs more knobs (e.g. window size), add fields to
   `FrameSelectionConfig` and surface them in CLI args.
3. `rollout_sampling_config_hash` will change automatically — trainer-side
   group filtering keeps variants from cross-contaminating GRPO groups.

## Known limitations

- Episodes run serially. asyncio plumbing exists but pool=1 means no parallel
  speedup until env_server `--pool-size > 1` (7.3c).
- vLLM `logprobs` / `token_ids` not yet captured. Add when wiring 7.3d trainer
  collator.
- Action parser is shared with 7.2 / NaVIDA upstream (`vln.rl.messages.parse_actions`).
  If the SFT model emits a different action format we must override here.
