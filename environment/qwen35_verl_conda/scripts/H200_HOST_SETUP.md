# Ubuntu 24.04 + H200 host setup

This is the host-side half of the Qwen3.5/Qwen3-VL Verl Conda delivery. It is
separate from both the training Conda environment and the existing Habitat
Conda environment.

## Required host baseline

- x86_64 Ubuntu 24.04 LTS (Ubuntu 24.04.2 is the R570.211.01-qualified HGX
  H200 combination; glibc 2.39, minimum accepted glibc is 2.31).
- 8 NVIDIA H200 GPUs visible as compute capability 9.0; MIG disabled.
- NVIDIA data-center driver R570 or newer. Use the latest patch release within
  the installed branch. R570 is validated for HGX H200 on Ubuntu 24.04 and
  exposes the CUDA 12.x API.
- On an HGX/NVSwitch node, a matching Fabric Manager service must be installed,
  enabled and healthy. PCIe-only H200 nodes may not need it.
- At least 100 GiB free local/NVMe space and preferably at least 64 GiB
  `/dev/shm`.

The Conda delivery uses PyTorch CUDA 12.9 wheels. Do **not** switch it to CUDA
13.0 merely because the host advertises CUDA 13 support. A system CUDA toolkit
and `nvcc` are not required. The validated artifact bundles the extension
wheels; the script-only route installs its build compiler inside Conda.

## Driver 570 and CUDA 12.9

CUDA 12.9 Update 1's full driver baseline is 575.57.08. Driver 570 can execute CUDA 12.x
through compatibility, but vLLM/Triton also JIT PTX. For a fixed R570 host,
install NVIDIA's `cuda-compat-12-9` user-space driver package and expose its
libraries to every Slurm/job shell:

```bash
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt-get update
sudo apt-get install -y cuda-compat-12-9

export LD_LIBRARY_PATH=/usr/local/cuda-12.9/compat${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
```

The compatibility package contains user-space driver/JIT libraries; it does
not replace the installed kernel driver. If the administrator can upgrade the
driver, R575.57.08+ (or a current R580 data-center branch) avoids this special
case. Do not expose an old `/usr/local/cuda-12.9/compat` path after upgrading
to R580+: CUDA's compatibility matrix marks that combination unsupported. If
H200 smoke reports `unsupported PTX version` or
`cudaErrorCallRequiresNewerDriver`, first verify the compatibility-library
path; the conservative fallback is a separately qualified cu128 stack, not
cu130.

References:

- https://docs.nvidia.com/deploy/cuda-compatibility/forward-compatibility.html
- https://docs.nvidia.com/cuda/archive/12.9.1/cuda-toolkit-release-notes/index.html
- https://docs.nvidia.com/datacenter/tesla/tesla-release-notes-570-211-01/index.html

## Ubuntu packages

Install the small host layer below. CUDA, cuDNN, NCCL and most Python libraries
still come from the frozen Conda/pip stack.

```bash
sudo apt-get update
sudo apt-get install -y \
  linux-headers-$(uname -r) \
  build-essential ca-certificates curl wget git git-lfs \
  bzip2 unzip xz-utils pkg-config cmake ninja-build \
  numactl libnuma1 libnuma-dev \
  rdma-core libibverbs1 libibverbs-dev librdmacm1 librdmacm-dev \
  libgl1 libegl1 libopengl0 libglvnd0 libglib2.0-0t64 ffmpeg \
  pciutils procps lsof
git lfs install
```

If this is a true bare-host bootstrap rather than an already managed cluster
node, install the NVIDIA data-center driver through the site administrator,
reboot, and only then install/enable the exactly matching Fabric Manager. A
newly installed kernel driver is not active until reboot; verify the loaded
version with `nvidia-smi` rather than only querying installed packages.

For an HGX H200 with NVSwitch, install the Fabric Manager package matching the
driver branch and enable it. Do not install this on a PCIe-only node merely to
silence the preflight warning.

```bash
sudo apt-get install -y cuda-drivers-fabricmanager-570
sudo systemctl enable --now nvidia-fabricmanager
nvidia-smi -q -d FABRIC
nvidia-smi topo -m
nv-fabricmanager -v
```

The driver and Fabric Manager branches must match. NVIDIA's platform guide:
https://docs.nvidia.com/hgx-platforms/fabric-manager-user-guide/index.html

## Limits and storage

Recommended job/session limits:

```bash
ulimit -n 1048576
ulimit -l unlimited
```

If users cannot raise them, the administrator should configure PAM/Slurm limits.
Use node-local NVMe for `TMPDIR`, Ray, Triton and vLLM caches when possible.
Avoid placing high-churn caches on a shared home directory. The staged rebuild
requires access to conda-forge, the PyTorch cu129 index and PyPI; the artifact
does not bundle the full transitive package cache. For an offline node, mirror
those repositories or pre-populate/export a complete Conda and pip cache in
addition to copying the artifact directory and model snapshots.

## Acceptance sequence

Run the read-only host audit before creating Conda:

```bash
EXPECTED_GPUS=8 REQUIRE_H200=1 REQUIRE_FABRIC_MANAGER=1 \
  REQUIRE_HABITAT_EGL=1 TARGET_DIR=/path/to/local/scratch \
  bash h200_host_preflight.sh
```

Use `REQUIRE_FABRIC_MANAGER=0` only when the hardware is confirmed to be a
PCIe/non-NVSwitch topology. The default strict mode requires every visible GPU
to report both `State: Completed` and `Status: Success`.

Then run `recreate_env.sh`. Its final kernel check validates PyTorch CUDA,
`causal-conv1d`, and Qwen3.5's GDN operator on sm90. Finally run the supplied
one-step Verl/vLLM smoke template with the colleague's Habitat/data paths. A
successful import or CUDA matrix multiplication alone is not sufficient to
qualify Driver 570 because it does not exercise Triton/PTX JIT.
