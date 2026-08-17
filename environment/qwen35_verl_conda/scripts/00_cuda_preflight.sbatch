#!/bin/bash
#SBATCH --job-name=q35-conda-preflight
#SBATCH --account=general_perflab
#SBATCH --partition=polar,polar3,polar4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:15:00
#SBATCH --output=/lustre/fsw/portfolios/general/users/jiaychen/logs/%x-%j.log
#SBATCH --error=/lustre/fsw/portfolios/general/users/jiaychen/logs/%x-%j.log

set -euo pipefail

WORK=${WORK:-/lustre/fsw/portfolios/general/users/jiaychen}
mkdir -p "$WORK/logs"

echo "timestamp=$(date -Is)"
echo "hostname=$(hostname)"
echo "kernel=$(uname -r)"
grep '^PRETTY_NAME=' /etc/os-release || true
echo "PATH=$PATH"

echo '--- nvidia-smi ---'
nvidia-smi
echo '--- driver query ---'
nvidia-smi --query-gpu=name,driver_version,memory.total,compute_cap --format=csv,noheader
echo '--- CUDA toolkit/compatibility ---'
command -v nvcc || true
nvcc --version || true
find /usr/local -maxdepth 2 -type d \( -name 'cuda*' -o -name 'compat' \) -print 2>/dev/null | sort
ldconfig -p 2>/dev/null | grep -E 'libcuda|libnvidia-ptxjitcompiler' || true

echo '--- host compiler ---'
gcc --version | head -n 1 || true
g++ --version | head -n 1 || true
cmake --version | head -n 1 || true

echo 'PREFLIGHT_OK'
