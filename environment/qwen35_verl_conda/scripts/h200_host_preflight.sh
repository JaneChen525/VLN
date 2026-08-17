#!/bin/bash
# Read-only host audit for the colleague's Ubuntu 24.04 + H200 node.
set -uo pipefail

EXPECTED_GPUS=${EXPECTED_GPUS:-8}
REQUIRE_H200=${REQUIRE_H200:-1}
REQUIRE_FABRIC_MANAGER=${REQUIRE_FABRIC_MANAGER:-1}
REQUIRE_HABITAT_EGL=${REQUIRE_HABITAT_EGL:-1}
MIN_FREE_GB=${MIN_FREE_GB:-100}
TARGET_DIR=${TARGET_DIR:-$PWD}
FAILURES=0
WARNINGS=0

pass() { printf '[PASS] %s\n' "$*"; }
warn() { printf '[WARN] %s\n' "$*"; WARNINGS=$((WARNINGS + 1)); }
fail() { printf '[FAIL] %s\n' "$*"; FAILURES=$((FAILURES + 1)); }

echo '=== OS / ABI ==='
if [[ -r /etc/os-release ]]; then
  . /etc/os-release
  echo "os=${PRETTY_NAME:-unknown}"
  [[ ${ID:-} == ubuntu && ${VERSION_ID:-0} == 24.04* ]] \
    && pass 'Ubuntu 24.04 detected' \
    || warn 'delivery was qualified for Ubuntu 24.04'
else
  fail '/etc/os-release is unavailable'
fi

ARCH=$(uname -m)
[[ $ARCH == x86_64 ]] && pass 'x86_64 host' || fail "expected x86_64, got $ARCH"
GLIBC=$(getconf GNU_LIBC_VERSION 2>/dev/null | awk '{print $2}')
if [[ -n ${GLIBC:-} ]] && dpkg --compare-versions "$GLIBC" ge 2.31; then
  pass "glibc=$GLIBC (>=2.31)"
else
  fail "glibc=${GLIBC:-unknown}; bundled extension needs >=2.31"
fi

echo '=== NVIDIA driver / GPUs ==='
if ! command -v nvidia-smi >/dev/null; then
  fail 'nvidia-smi is missing'
else
  DRIVER_LINES=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null || true)
  GPU_NAMES=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || true)
  GPU_CAPS=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null || true)
  GPU_MIG=$(nvidia-smi --query-gpu=mig.mode.current --format=csv,noheader 2>/dev/null || true)
  DRIVER=$(awk 'NF {gsub(/[[:space:]]/, ""); print; exit}' <<< "$DRIVER_LINES")
  GPU_COUNT=$(awk 'NF {n++} END {print n+0}' <<< "$GPU_NAMES")
  echo "driver=$DRIVER gpu_count=$GPU_COUNT"
  nvidia-smi --query-gpu=index,name,driver_version,memory.total,compute_cap,mig.mode.current \
    --format=csv,noheader || true

  dpkg --compare-versions "$DRIVER" ge 570.0 \
    && pass "driver branch is R570 or newer" \
    || fail "driver $DRIVER is older than R570"
  [[ $GPU_COUNT -ge $EXPECTED_GPUS ]] \
    && pass "at least $EXPECTED_GPUS GPUs visible" \
    || fail "only $GPU_COUNT GPUs visible; expected $EXPECTED_GPUS"

  if (( REQUIRE_H200 )); then
    NON_H200=$(awk 'NF && $0 !~ /H200/ {n++} END {print n+0}' <<< "$GPU_NAMES")
    if (( NON_H200 > 0 )); then
      fail 'one or more visible GPUs are not H200'
    else
      pass 'all visible GPUs are H200'
    fi
  fi
  NON_SM90=$(awk 'NF && $0 !~ /^9\.0$/ {n++} END {print n+0}' <<< "$GPU_CAPS")
  if (( NON_SM90 > 0 )); then
    fail 'one or more GPUs do not report compute capability 9.0 (sm90)'
  else
    pass 'all GPUs report sm90'
  fi
  MIG_ENABLED=$(awk 'NF && tolower($0) !~ /^disabled$/ {n++} END {print n+0}' <<< "$GPU_MIG")
  if (( MIG_ENABLED > 0 )); then
    fail 'MIG must be disabled for the 8-GPU TP/FSDP smoke'
  else
    pass 'MIG disabled'
  fi

  # CUDA 12.9 Update 1's full driver baseline is 575.57.08. With R570, install the
  # forward-compatibility user-space driver libraries because vLLM/Triton JITs PTX.
  if dpkg --compare-versions "$DRIVER" lt 575.57.08; then
    COMPAT=/usr/local/cuda-12.9/compat
    if [[ -e $COMPAT/libcuda.so.1 ]]; then
      pass 'cuda-compat-12-9 libraries found'
      [[ :${LD_LIBRARY_PATH:-}: == *":$COMPAT:"* ]] \
        && pass 'cuda-compat-12-9 is active in LD_LIBRARY_PATH' \
        || fail "add $COMPAT to LD_LIBRARY_PATH before launching jobs"
    else
      fail 'R570 requires cuda-compat-12-9 for the cu129 vLLM/Triton path'
    fi
  else
    pass 'driver meets CUDA 12.9 Update 1 full-compatibility baseline'
    if dpkg --compare-versions "$DRIVER" ge 580.0 && \
       [[ :${LD_LIBRARY_PATH:-}: == *':/usr/local/cuda-12.9/compat:'* ]]; then
      fail 'R580+ must not load the stale cuda-compat-12-9 library path'
    fi
  fi
fi

echo '=== NVSwitch / fabric ==='
FM_UNITS=''
if command -v systemctl >/dev/null; then
  FM_UNITS=$(systemctl list-unit-files --type=service 2>/dev/null || true)
fi
if grep -q '^nvidia-fabricmanager' <<< "$FM_UNITS"; then
  if systemctl is-active --quiet nvidia-fabricmanager; then
    pass 'nvidia-fabricmanager is active'
  else
    fail 'nvidia-fabricmanager is installed but inactive'
  fi
  command -v nv-fabricmanager >/dev/null && nv-fabricmanager -v || true
else
  if (( REQUIRE_FABRIC_MANAGER )); then
    fail 'Fabric Manager service is required for the target HGX/NVSwitch node'
  else
    warn 'Fabric Manager service not found; explicitly allowed for a PCIe/non-NVSwitch node'
  fi
fi
if command -v nvidia-smi >/dev/null; then
  nvidia-smi topo -m || true
  FABRIC=$(nvidia-smi -q -d FABRIC 2>/dev/null || true)
  COMPLETED_COUNT=$(awk -F: '/^[[:space:]]*State[[:space:]]*:/ {gsub(/^[[:space:]]+|[[:space:]]+$/, "", $2); if ($2 == "Completed") n++} END {print n+0}' <<< "$FABRIC")
  SUCCESS_COUNT=$(awk -F: '/^[[:space:]]*Status[[:space:]]*:/ {gsub(/^[[:space:]]+|[[:space:]]+$/, "", $2); if ($2 == "Success") n++} END {print n+0}' <<< "$FABRIC")
  if (( REQUIRE_FABRIC_MANAGER )); then
    if (( COMPLETED_COUNT == GPU_COUNT && SUCCESS_COUNT == GPU_COUNT && GPU_COUNT > 0 )); then
      pass "all $GPU_COUNT GPUs report fabric State=Completed and Status=Success"
    else
      fail "fabric healthy GPUs=$COMPLETED_COUNT/$SUCCESS_COUNT of $GPU_COUNT (Completed/Success)"
    fi
  elif [[ -n $FABRIC ]]; then
    warn "fabric check optional: Completed=$COMPLETED_COUNT Success=$SUCCESS_COUNT GPUs=$GPU_COUNT"
  fi
fi

echo '=== Host runtime libraries ==='
LDCONFIG_OUTPUT=$(ldconfig -p 2>/dev/null || true)
for lib in libnuma.so.1 libibverbs.so.1 librdmacm.so.1 libGL.so.1 libglib-2.0.so.0; do
  if grep -Fq "$lib" <<< "$LDCONFIG_OUTPUT"; then
    pass "$lib"
  else
    fail "$lib is missing"
  fi
done
if (( REQUIRE_HABITAT_EGL )); then
  for lib in libEGL.so.1 libOpenGL.so.0 libGLdispatch.so.0 libEGL_nvidia.so.0; do
    if grep -Fq "$lib" <<< "$LDCONFIG_OUTPUT"; then
      pass "$lib"
    else
      fail "$lib is missing; the Habitat GPU-EGL process needs it"
    fi
  done
fi
for cmd in git git-lfs curl gcc g++ cmake ninja numactl; do
  command -v "$cmd" >/dev/null && pass "$cmd available" || warn "$cmd is missing"
done

echo '=== Resource limits / storage ==='
NOFILE=$(ulimit -n)
if [[ $NOFILE == unlimited ]] || (( NOFILE >= 65535 )); then
  pass "open-files limit=$NOFILE"
else
  warn "open-files limit=$NOFILE; set at least 65535 in the job"
fi
MEMLOCK=$(ulimit -l)
[[ $MEMLOCK == unlimited ]] \
  && pass 'memlock=unlimited' \
  || warn "memlock=$MEMLOCK; unlimited is recommended for NCCL/RDMA"
SHM_GB=$(df -Pk /dev/shm 2>/dev/null | awk 'NR==2 {print int($2/1024/1024)}')
if [[ -n ${SHM_GB:-} && $SHM_GB -ge 64 ]]; then
  pass "/dev/shm=${SHM_GB}GiB"
else
  warn "/dev/shm=${SHM_GB:-unknown}GiB; >=64GiB is recommended"
fi
FREE_GB=$(df -Pk "$TARGET_DIR" 2>/dev/null | awk 'NR==2 {print int($4/1024/1024)}')
if [[ -n ${FREE_GB:-} && $FREE_GB -ge $MIN_FREE_GB ]]; then
  pass "free space at $TARGET_DIR=${FREE_GB}GiB"
else
  fail "free space at $TARGET_DIR=${FREE_GB:-unknown}GiB; need >=${MIN_FREE_GB}GiB"
fi

echo "HOST_PREFLIGHT_SUMMARY failures=$FAILURES warnings=$WARNINGS"
(( FAILURES == 0 ))
