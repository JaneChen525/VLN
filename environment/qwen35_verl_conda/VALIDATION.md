# Validation record

Initial handoff scope: Qwen3.5-4B, native Conda, FSDP2 actor/reference, vLLM
rollout, and the existing independent Habitat environment.

## Passed gates

| Gate | Environment | Result |
|---|---|---|
| Stack/build | Original A100 prefix | CUDA kernels, Qwen3.5 config/processor, package checks passed |
| RL smoke | Original A100 prefix | `training/global_step:1`, optimizer path and vLLM weight sync completed |
| Clean recreation | New Conda prefix | Environment reconstructed from the supplied recipe and manifests |
| RL smoke | Recreated A100 prefix | `training/global_step:1`, optimizer path and vLLM weight sync completed |

Both RL smokes were run by Slurm job `32281360`. Evidence is under:

- `validation/qwen35/original/`
- `validation/qwen35/recreated/`

The diagnostic rollout produced zero reward for every trajectory, so GRPO
advantages and `actor/grad_norm` were zero. This is expected for this tiny
execution smoke and proves environment/runtime reachability, not policy
learning quality.

## Deferred gate

Qwen3-VL config and processor loading were checked, but the Qwen3-VL full
one-step RL smoke is intentionally deferred to a follow-up release. This
initial handoff does not claim end-to-end Qwen3-VL validation.
