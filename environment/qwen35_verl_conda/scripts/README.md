# env4 Qwen3.5-4B Verl Conda environment

Purpose: replace the Qwen3.5-compatible Verl container with a native Conda
training environment while retaining Qwen3-VL compatibility. Habitat remains
in the existing independent `vln` Conda environment and is reached through the
existing env-server API.

This handoff never copies a Conda prefix. There are two supported delivery
forms:

- **Script/list only:** copy this directory and run `build_env.sh` inside a
  Slurm allocation. It installs a Conda-local CUDA compiler, builds the two
  extension wheels, and emits the exact manifests.
- **Validated artifact:** additionally copy the generated manifest directory,
  including its two portable extension wheels, and run `recreate_env.sh`.
  This avoids rebuilding CUDA extensions but is still not an environment
  archive.

## Frozen core stack

- Python 3.12
- PyTorch 2.10.0 + CUDA 12.9 wheels
- torchvision 0.25.0 / torchaudio 2.10.0
- vLLM 0.18.0
- Transformers 5.3.0
- Verl source at the commit recorded in `verl-commit.txt`
- causal-conv1d 1.6.2.post1
- flash-attn 2.8.3
- fla-core 0.4.2
- flash-linear-attention 0.4.2
- Transformers 5.3.0 Qwen3.5/FlashAttention 3-D M-RoPE hotfix
- `use_remove_padding=False` for the Qwen3.5 Verl path; model full-attention
  layers still use FlashAttention 2, matching `Dockerfile.stable.vllm`

The CUDA runtime comes from Python wheels. A host CUDA toolkit is not required.
The tested env4 host is A100 80GB / driver 535.129.03. A colleague's H200 /
driver 570 can use the same cu129 package recipe; do not copy the Conda prefix
byte-for-byte and do not replace it with cu130. Follow `H200_HOST_SETUP.md`
before running the staged recreation script.
See `DOCKER_PARITY.md` for a component-by-component comparison with the source
container and evidence for the deliberately omitted general-purpose backends.
The build Conda environment also carries the minimal CUDA 12.9 compiler used to
produce glibc-2.31-compatible `causal-conv1d` and FlashAttention wheels
containing sm80 and sm90. The build records `cuobjdump` output and fails unless
both cubins are present.

## Build and test on env4

```bash
WORK=/lustre/fsw/portfolios/general/users/jiaychen
cd "$WORK/WorldModel"

sbatch scripts/qwen35_conda/01_build_env.sbatch
# After BUILD_SUCCESS:
sbatch scripts/qwen35_conda/02_rl_smoke.sbatch

# Qwen3-VL compatibility smoke in the same Conda environment:
sbatch \
  --export=ALL,MODEL_FAMILY=qwen3vl,MODEL_PATH_OVERRIDE=$WORK/checkpoints/Qwen3VL_4B_Final_swift,EXPERIMENT_PREFIX=qwen3vl-4b-conda-smoke \
  scripts/qwen35_conda/02_rl_smoke.sbatch
```

At another site, override `--account`, `--partition`, `--output`, and `--error`,
and pass `WORK`, `WORLDMODEL`, and `QWEN35_SCRIPTS_DIR` through
`sbatch --export=ALL,...`. With those site values supplied, the preflight and
build scripts do not depend on the env4 paths embedded in their Slurm defaults.

The build creates:

- Conda env: `$WORK/conda-envs/qwen35-verl-vllm018`
- Official base model: `$WORK/checkpoints/Qwen3.5-4B-base`
- Exact manifests: `$WORLDMODEL/artifacts/qwen35-conda-env`
- Smoke results: `$WORLDMODEL/results/qwen-conda-smoke/{qwen35,qwen3vl}/<job-id>`
- Final sealed release: `$WORLDMODEL/artifacts/qwen35-conda-release-<job-id>.tar.gz`

## Reproduce on another machine

Set the artifact root, then clone the exact recorded sources. For the base
manifest directory use `/path/to/artifacts/qwen35-conda-env`. If using the
sealed release, verify its adjacent `.sha256`, extract it, and point directly
at the extracted `qwen35-conda-release-<job>` directory; the release root is
the artifact root.

```bash
ARTIFACT=/path/to/artifacts/qwen35-conda-env
git clone "$(cat "$ARTIFACT/worldmodel-origin.txt")" /path/to/WorldModel
git -C /path/to/WorldModel checkout "$(cat "$ARTIFACT/worldmodel-commit.txt")"
git clone "$(cat "$ARTIFACT/verl-origin.txt")" /path/to/WorldModel/vln/reinforcement_learning
git -C /path/to/WorldModel/vln/reinforcement_learning checkout "$(cat "$ARTIFACT/verl-commit.txt")"
```

The artifact also records tracked working/index patches and untracked-file
lists. Inspect those provenance files before applying site-specific changes;
the supplied smoke scripts themselves are already self-contained under
`artifact/scripts`.
First run the read-only H200 host audit:

```bash
EXPECTED_GPUS=8 REQUIRE_H200=1 REQUIRE_FABRIC_MANAGER=1 \
  REQUIRE_HABITAT_EGL=1 TARGET_DIR=/path/to/local/scratch \
  bash "$ARTIFACT/h200_host_preflight.sh"
```

Then reconstruct the environment into a new, empty prefix:

```bash
bash "$ARTIFACT/recreate_env.sh" \
  /path/to/miniconda3 \
  /path/to/conda-envs/qwen35-verl-vllm018 \
  "$ARTIFACT" \
  /path/to/WorldModel \
  /path/to/WorldModel/vln/reinforcement_learning
```

Finally submit the real one-step test. Override the site-specific Slurm account,
partition, log location, Habitat environment and dataset paths:

```bash
cd "$ARTIFACT/scripts"
sbatch --account=<account> --partition=<partition> \
  --output=/path/to/logs/%x-%j.log --error=/path/to/logs/%x-%j.log \
  --export=ALL,WORK=/path/to/work,WORLDMODEL=/path/to/WorldModel,MANIFEST_DIR_OVERRIDE=$ARTIFACT,TRAIN_ENV_OVERRIDE=/path/to/conda-envs/qwen35-verl-vllm018,CONDA_EXE_OVERRIDE=/path/to/miniconda3/bin/conda,QWEN35_SCRIPTS_DIR=$PWD,HABITAT_PY_OVERRIDE=/path/to/habitat-env/bin/python,HABITAT_CONFIG_OVERRIDE=$PWD/vln_r2r_smoke.yaml,HABITAT_SCENES_DIR_OVERRIDE=/path/to/scene_datasets,HABITAT_R2R_DATA_PATH_OVERRIDE=/path/to/vln_eval_datasets/r2r/{split}/{split}.json.gz,MODEL_PATH_OVERRIDE=/path/to/Qwen3.5-4B-base,TRAIN_FILE_OVERRIDE=/path/to/train.parquet,VAL_FILE_OVERRIDE=/path/to/val.parquet \
  02_rl_smoke.sbatch
```

Repeat the same command for Qwen3-VL, retaining every site/artifact override
and appending/replacing these fields in the comma-separated `--export` value:

```bash
...,MODEL_FAMILY=qwen3vl,MODEL_PATH_OVERRIDE=/path/to/Qwen3VL_4B_Final_swift,EXPERIMENT_PREFIX=qwen3vl-4b-conda-smoke
```

`requirements-lock.txt` is the exact pip snapshot from the successful smoke
build environment. Smoke jobs never rewrite that baseline; they write package,
input-hash, kernel, and optimizer-step evidence under the results tree.
`pip-list.txt` is the concise colleague-facing package list;
`conda-explicit.txt` and `conda-environment.yml` provide additional provenance.
`requirements-portable.txt` is used by the staged rebuild to preserve the
intentional vLLM-0.18/Transformers-5.3 metadata override from the reference
Dockerfile. The same reference stack also needs Verl's NumPy 1.26 with OpenCV
4.11 rather than vLLM's newer OpenCV metadata floor. `pip-check.txt` may contain
only these two documented vLLM metadata overrides and no unexpected conflicts.
The bundled `wheels/causal_conv1d-*.whl` and `wheels/flash_attn-*.whl` are built
on env4's older glibc and contain both A100 and H200 device code, so the
colleague does not need nvcc.
`recreate_env.sh` applies an idempotent guard for the known Transformers 5.3.0
Qwen3.5/FlashAttention packed-sequence bug, then runs real forward/backward
causal-convolution, FlashAttention, and GDN CUDA kernels. Installation is not
considered successful unless this check passes.

`PREPARE_EXISTING_ENV=1` on `05_full_acceptance.sbatch` is only for repairing
an older maintainer environment created before that guard was added. New users
should always build or recreate from the supplied recipe and leave it unset.
The supplied `03_recreate_verify.sbatch` performs a clean-prefix reconstruction
and dual-model load check. The release is accepted only after that recreated
prefix also completes the one-step RL smoke.

`qwen35-model-revision.txt` pins the exact Hugging Face revision. To fetch the
same model on another host:

```bash
REV=$(cat "$ARTIFACT/qwen35-model-revision.txt")
/path/to/conda-envs/qwen35-verl-vllm018/bin/hf download \
  Qwen/Qwen3.5-4B --revision "$REV" \
  --local-dir /path/to/checkpoints/Qwen3.5-4B-base
(cd /path/to/checkpoints/Qwen3.5-4B-base && \
  sha256sum -c "$ARTIFACT/qwen35-model-files.sha256")
```

`qwen35-model-files.sha256` and `qwen3vl-model-files.sha256` record every
top-level model file, including weight shards and processor/tokenizer metadata.
The private Qwen3-VL checkpoint is not redistributed; provide the same snapshot
and verify it with `qwen3vl-model-files.sha256`. Each successful smoke stores
the train/validation parquet, Habitat config, and R2R JSON hashes in
`results/qwen-conda-smoke/evidence/<model>/<run-id>/input-provenance.txt`.
Matterport scene assets
are licensed external data and are referenced by path rather than bundled.

## Deliberate scope

This is the FSDP2 + vLLM Qwen3.5/Qwen3-VL VLN path. It omits container-only components
not exercised by this path (Apex, Transformer Engine, Megatron, DeepEP and
Nsight Systems). The lightweight Python `nvtx` marker package remains installed
and does not require the Nsight profiler. Correctness/SR is not evaluated: the smoke config limits Habitat to
two environment steps and only verifies one RL optimizer step.
The build performs a local Qwen3-VL config/processor load check; this is not a
weight/forward test. Full Qwen3-VL
RL compatibility is claimed only after running the optional Qwen3-VL one-step
command above. `02_rl_smoke.sbatch` is an env4 template: `WORK`, `WORLDMODEL`,
`TRAIN_ENV_OVERRIDE`, `HABITAT_PY_OVERRIDE`, `HABITAT_CONFIG_OVERRIDE`,
`HABITAT_SCENES_DIR_OVERRIDE`, `HABITAT_R2R_DATA_PATH_OVERRIDE`,
`TRAIN_FILE_OVERRIDE`, `VAL_FILE_OVERRIDE`, and `CONDA_EXE_OVERRIDE` can be supplied for another
machine. Set `QWEN35_SCRIPTS_DIR=/path/to/artifact/scripts` when the scripts are
not checked into the WorldModel clone. Its `#SBATCH --output` path should also
be overridden together with `#SBATCH --error` at submission.

## Release acceptance order

The release is sealed only after all four gates pass: (1) Qwen3.5 one optimizer
step in the build environment, (2) clean-prefix reconstruction and CUDA/config
checks, (3) Qwen3.5 one optimizer step in that recreated prefix, and (4)
Qwen3-VL one optimizer step in the same recreated prefix. Then run
`04_finalize_release.sbatch` with the four recorded job IDs. It verifies the
markers, copies immutable validation evidence, creates `RELEASE_SUCCESS`, and
emits a tarball plus an external SHA256 file.

On a 4-hour cluster, prefer `05_full_acceptance.sbatch`: it reserves one
8-GPU node once and runs all four post-build gates plus finalization serially
inside that allocation. This avoids five separate scheduler waits. While it is
running, an operator can inspect the same node with
`srun --jobid=<job-id> --overlap --pty bash`.
