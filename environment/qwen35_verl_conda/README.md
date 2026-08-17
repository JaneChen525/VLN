# Qwen3.5 Verl Conda handoff

This directory is the colleague-facing, script/list-only handoff for the native
Conda replacement of the project's Qwen3.5-compatible Verl container. It does
**not** contain or copy a Conda prefix. Habitat remains in its existing,
independent environment and is reached through the env-server API.

## What is included

- `scripts/`: host audit, build, reconstruction, Qwen3.5 FA2 hotfix, stack
  checks, one-step RL smoke, and full acceptance scripts.
- `manifests/`: exact Python/Conda package snapshots, source commits, model
  hashes, CUDA cubin evidence, and the two documented `pip check` exceptions.
- `validation/`: immutable evidence from the original-prefix and clean-prefix
  Qwen3.5 one-step RL smoke tests.

The lightweight directory intentionally omits model weights, datasets, the
Conda prefix, and the two large CUDA extension wheels. Run `scripts/build_env.sh`
to build those wheels and emit a complete artifact. The separately sealed
validated artifact contains the wheels for a no-compile reconstruction path.

## Tested stack

- Python 3.12.13
- PyTorch 2.10.0 + cu129
- vLLM 0.18.0
- Transformers 5.3.0
- FlashAttention 2.8.3
- causal-conv1d 1.6.2.post1
- fla-core / flash-linear-attention 0.4.2
- FSDP2 actor/reference + 8-way vLLM tensor-parallel rollout

The A100 build contains both `sm80` and `sm90` device code. For an H200 host
with Driver 570, first follow `scripts/H200_HOST_SETUP.md`; in particular use
`cuda-compat-12-9`, a matching healthy Fabric Manager on HGX/NVSwitch systems,
and the supplied host preflight. Do not replace the tested cu129 stack with
cu130 merely because the host advertises CUDA 13 support.

## Minimal handoff sequence

1. Run the host audit:

   ```bash
   EXPECTED_GPUS=8 REQUIRE_H200=1 REQUIRE_FABRIC_MANAGER=1 \
     REQUIRE_HABITAT_EGL=1 TARGET_DIR=/path/to/local/scratch \
     bash scripts/h200_host_preflight.sh
   ```

2. Clone the exact source revisions:

   ```bash
   git clone "$(cat manifests/worldmodel-origin.txt)" /path/to/WorldModel
   git -C /path/to/WorldModel checkout "$(cat manifests/worldmodel-commit.txt)"
   git clone "$(cat manifests/verl-origin.txt)" \
     /path/to/WorldModel/vln/reinforcement_learning
   git -C /path/to/WorldModel/vln/reinforcement_learning checkout \
     "$(cat manifests/verl-commit.txt)"
   ```

   The recorded WorldModel working-tree change only rewrote old site-specific
   Habitat paths and is not required by the parameterized smoke configuration;
   the recorded Verl tree is clean.

3. Build from scripts inside a Slurm GPU allocation. Override every site
   default explicitly:

   ```bash
   HANDOFF=/path/to/rl_qwen35_verl_conda
   WORK=/path/to/project-storage
   WORLDMODEL=/path/to/WorldModel

   sbatch --account=<account> --partition=<partition> \
     --output=/path/to/logs/%x-%j.log --error=/path/to/logs/%x-%j.log \
     --export=ALL,WORK=$WORK,WORLDMODEL=$WORLDMODEL,QWEN35_SCRIPTS_DIR=$HANDOFF/scripts,CONDA_ROOT=$WORK/miniconda3,ENV_PREFIX=$WORK/conda-envs/qwen35-verl-vllm018,VERL_SRC=$WORLDMODEL/vln/reinforcement_learning,MODEL_DIR=$WORK/checkpoints/Qwen3.5-4B-base,QWEN3VL_MODEL_DIR=/path/to/Qwen3VL-4B-checkpoint \
     "$HANDOFF/scripts/01_build_env.sbatch"
   ```

   This source-build path needs network access and 220 GiB build-job memory;
   the tested FlashAttention build peaked near 69 GiB with `MAX_JOBS=4`.

4. Run the real one-step acceptance using the independent Habitat Python and
   the colleague's scene/R2R/parquet paths. The exact command and every override
   are in `scripts/README.md` under “Reproduce on another machine”. Run it once
   for Qwen3.5 and once for Qwen3-VL.

## Acceptance meaning

The environment has completed a real Habitat rollout and the full Verl path:
FSDP2 forward/backward, optimizer call, vLLM weight synchronization, and
`training/global_step:1`. The tiny diagnostic rollout happened to return all
zero rewards, so GRPO advantages, policy loss, and `actor/grad_norm` were zero;
this validates the execution path but not learning quality or long-run
stability.

## Current validation status

- **Qwen3.5 original environment:** passed one real RL step.
- **Qwen3.5 clean reconstruction:** passed package/kernel checks and one real
  RL step from a newly created Conda prefix.
- **Qwen3-VL:** config/processor compatibility is covered by the build checks;
  its end-to-end RL smoke will be added in a follow-up and is not claimed by
  this initial handoff.

The concise evidence index is in `VALIDATION.md`.

See `scripts/README.md` for the full reconstruction and validation runbook and
`scripts/DOCKER_PARITY.md` for why Apex, Transformer Engine, Megatron, DeepEP,
and Nsight Systems are omitted from this FSDP2 + dense-vLLM project path.
