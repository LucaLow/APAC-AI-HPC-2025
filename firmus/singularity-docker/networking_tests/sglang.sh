#!/bin/bash

#SBATCH -p hpcai
#SBATCH -N 2
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=nvidia_h200:8
#SBATCH --exclusive
#SBATCH -t 00:45:00
#SBATCH -o slurm-%j.out
#SBATCH -e slurm-%j.err
#SBATCH --hint=multithread

set -euo pipefail

# =============================================================================
# CRITICAL FIX: Direct IP resolution from compute fabric interface
# =============================================================================

# Use compute fabric instead of management network
export NCCL_SOCKET_IFNAME=ens40f0np0
export GLOO_SOCKET_IFNAME=ens40f0np0

# Enhanced NCCL configuration for multi-node performance
export NCCL_DEBUG=INFO
export NCCL_NET_GDR_LEVEL=PHB
export NCCL_CROSS_NIC=1
export NCCL_MIN_NCHANNELS=4
export NCCL_TIMEOUT=1800

# CRITICAL: Get IP directly from the compute fabric interface on master node
MASTER_NODE=$(scontrol show hostname "$SLURM_JOB_NODELIST" | head -n 1)

# Try multiple methods to get the actual compute fabric IP
echo "Attempting to resolve compute fabric IP for master node: $MASTER_NODE"

# Method 1: Direct interface query on master node
MASTER_FABRIC_IP=""
if [[ "$SLURM_NODEID" == "0" ]] || [[ "$(hostname)" == "$MASTER_NODE" ]]; then
    # We are on the master node, get IP directly from interface
    MASTER_FABRIC_IP=$(ip addr show ens40f0np0 | grep 'inet ' | awk '{print $2}' | cut -d'/' -f1 | head -n1)
    echo "Direct interface query on master: $MASTER_FABRIC_IP"
fi

# Method 2: If not master node or no IP found, try SSH or alternative methods
if [[ -z "$MASTER_FABRIC_IP" ]]; then
    # Try to get IP from management network and map to fabric
    MGMT_IP=$(scontrol show node $MASTER_NODE | grep -oP 'NodeAddr=\K[^\s]+' | head -n1)
    
    # If management IP is in 198.18.151.x range, map to fabric 198.18.57.x
    if [[ "$MGMT_IP" =~ ^198\.18\.151\.([0-9]+)$ ]]; then
        LAST_OCTET="${BASH_REMATCH[1]}"
        MASTER_FABRIC_IP="198.18.57.${LAST_OCTET}"
        echo "Mapped management IP $MGMT_IP to fabric IP $MASTER_FABRIC_IP"
    else
        echo "Could not map management IP: $MGMT_IP"
    fi
fi

# Method 3: Hardcode mapping based on hostname pattern (fallback)
if [[ -z "$MASTER_FABRIC_IP" ]]; then
    case "$MASTER_NODE" in
        g1) MASTER_FABRIC_IP="198.18.57.1" ;;
        g2) MASTER_FABRIC_IP="198.18.57.2" ;;
        g3) MASTER_FABRIC_IP="198.18.57.3" ;;
        g4) MASTER_FABRIC_IP="198.18.57.4" ;;
        g5) MASTER_FABRIC_IP="198.18.57.5" ;;
        *) 
            echo "ERROR: Cannot determine fabric IP for $MASTER_NODE"
            echo "Please manually set the IP mapping in the script"
            exit 1
            ;;
    esac
    echo "Using hardcoded mapping: $MASTER_NODE -> $MASTER_FABRIC_IP"
fi

# Validate IP format
if [[ ! "$MASTER_FABRIC_IP" =~ ^[0-9]+\.[0-9]+\.[0-9]+\.[0-9]+$ ]]; then
    echo "ERROR: Invalid IP format: $MASTER_FABRIC_IP"
    exit 1
fi

# Set the distribution address with explicit IP
export DIST_INIT_ADDR="${MASTER_FABRIC_IP}:29500"

echo "==============================================="
echo "FIXED MULTI-NODE CONFIGURATION"
echo "==============================================="
echo "Master node: ${MASTER_NODE}"
echo "Compute fabric IP: ${MASTER_FABRIC_IP}"
echo "Distribution address: ${DIST_INIT_ADDR}"
echo "Network interface: ${NCCL_SOCKET_IFNAME}"
echo "NCCL timeout: ${NCCL_TIMEOUT}s"
echo "==============================================="

# Verify connectivity (optional debug step)
echo "Testing connectivity to compute fabric IP..."
if ping -c 1 -W 5 "$MASTER_FABRIC_IP" &>/dev/null; then
    echo "✓ Master fabric IP $MASTER_FABRIC_IP is reachable"
else
    echo "⚠ WARNING: Master fabric IP $MASTER_FABRIC_IP is not reachable"
fi

# Container image
IMAGE="$HOME/sglang_v0.5.3rc1-cu126.sif"

# Hopper architecture
export TORCH_CUDA_ARCH_LIST="9.0"

# Pass environment variables to container
export SINGULARITYENV_NCCL_DEBUG="$NCCL_DEBUG"
export SINGULARITYENV_NCCL_SOCKET_IFNAME="$NCCL_SOCKET_IFNAME"
export SINGULARITYENV_GLOO_SOCKET_IFNAME="$GLOO_SOCKET_IFNAME"
export SINGULARITYENV_NCCL_NET_GDR_LEVEL="$NCCL_NET_GDR_LEVEL"
export SINGULARITYENV_NCCL_CROSS_NIC="$NCCL_CROSS_NIC"
export SINGULARITYENV_NCCL_MIN_NCHANNELS="$NCCL_MIN_NCHANNELS"
export SINGULARITYENV_NCCL_TIMEOUT="$NCCL_TIMEOUT"
export SINGULARITYENV_DIST_INIT_ADDR="$DIST_INIT_ADDR"
export SINGULARITYENV_TORCH_CUDA_ARCH_LIST="$TORCH_CUDA_ARCH_LIST"

# Additional Gloo configuration
export GLOO_TIMEOUT=600
export SINGULARITYENV_GLOO_TIMEOUT="$GLOO_TIMEOUT"

# Force TCP for Gloo (avoid auto-detection issues)
export GLOO_DEVICE_TRANSPORT=TCP
export SINGULARITYENV_GLOO_DEVICE_TRANSPORT="$GLOO_DEVICE_TRANSPORT"

time srun --label --cpu-bind=none \
--nodes="$SLURM_JOB_NUM_NODES" --ntasks-per-node=1 \
bash -lc '

set -euo pipefail

# System limits
ulimit -l unlimited || true

# Clean any stale FlashInfer cache
rm -rf "$HOME/.cache/flashinfer/"* || true

# Per-rank Triton cache on local node tmp
CACHE_ROOT="${SLURM_TMPDIR:-${TMPDIR:-/tmp}}/triton/${SLURM_JOB_ID}/node${SLURM_NODEID}"
export TRITON_CACHE_DIR="${CACHE_ROOT}/rank${SLURM_PROCID:-0}"
mkdir -p "$TRITON_CACHE_DIR"
echo "TRITON_CACHE_DIR=$TRITON_CACHE_DIR"

# Torch profiler dir
export SGLANG_TORCH_PROFILER_DIR="$HOME/sgprofile/node${SLURM_NODEID}"
mkdir -p "$SGLANG_TORCH_PROFILER_DIR"
echo "profile dir: $SGLANG_TORCH_PROFILER_DIR"

# Flashinfer per rank storage
export FLASHINFER_JIT_DIR="$CACHE_ROOT/flashinfer"
export XDG_CACHE_HOME="$CACHE_ROOT/xdg"
mkdir -p "$FLASHINFER_JIT_DIR" "$XDG_CACHE_HOME"

export SINGULARITYENV_FLASHINFER_JIT_DIR="$FLASHINFER_JIT_DIR"
export SINGULARITYENV_XDG_CACHE_HOME="$XDG_CACHE_HOME"

mkdir -p "$CACHE_ROOT/torch_ext"

BIND_CACHE="--bind $CACHE_ROOT/flashinfer:$HOME/.cache/flashinfer \
--bind $CACHE_ROOT/torch_ext:$HOME/.cache/torch/extensions"

# Pass SLURM variables into container
export SINGULARITYENV_TRITON_CACHE_DIR="$TRITON_CACHE_DIR"
export SINGULARITYENV_SGLANG_TORCH_PROFILER_DIR="$SGLANG_TORCH_PROFILER_DIR"
export SINGULARITYENV_SLURM_NODEID="$SLURM_NODEID"
export SINGULARITYENV_SLURM_PROCID="$SLURM_PROCID"
export SINGULARITYENV_SLURM_JOB_ID="$SLURM_JOB_ID"
export SINGULARITYENV_SLURM_JOB_NUM_NODES="$SLURM_JOB_NUM_NODES"

echo "Node ${SLURM_NODEID}: Starting with verified fabric IP connectivity"
echo "Using network interface: $NCCL_SOCKET_IFNAME"
echo "Distribution address: $DIST_INIT_ADDR"

# Test network connectivity before starting SGLang
echo "Final connectivity test from container..."
if ping -c 1 -W 5 $(echo $DIST_INIT_ADDR | cut -d: -f1) &>/dev/null; then
    echo "✓ Connectivity confirmed from node $SLURM_NODEID"
else
    echo "✗ WARNING: No connectivity from node $SLURM_NODEID"
fi

# Run inside the container
singularity exec --nv $BIND_CACHE $HOME/sglang_v0.5.3rc1-cu126.sif bash -lc "
set -euo pipefail

echo \"Container environment loaded on node \$SLURM_NODEID\"
echo \"Network interface: \$NCCL_SOCKET_IFNAME\"
echo \"Distribution address: \$DIST_INIT_ADDR\"
echo \"Gloo transport: \$GLOO_DEVICE_TRANSPORT\"

python3 -m sglang.bench_offline_throughput \
--model-path deepseek-ai/DeepSeek-R1 \
--dataset-path \$HOME/ShareGPT_V3_unfiltered_cleaned_split.json \
--num-prompts 2000 --load-format dummy --seed 2025 --dtype bfloat16 \
--tp 16 --nnodes \$SLURM_JOB_NUM_NODES --trust-remote-code \
--dist-init-addr \$DIST_INIT_ADDR --node-rank \$SLURM_NODEID
"

'

echo "==============================================="
echo "Job completed - check for clean Gloo connections"
echo "Expected: 'Rank X is connected to 15 peer ranks'"
echo "Not: 'Rank 0 is connected to 0 peer ranks'"
echo "==============================================="
