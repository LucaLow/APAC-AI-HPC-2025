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

export NCCL_DEBUG=WARN
export DIST_INIT_ADDR="$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)"
echo "DIST_INIT_ADDR=${DIST_INIT_ADDR}"

# keep your existing SIF or swap to a cu128 build later
IMAGE="$HOME/sglang_v0.5.3rc1-cu126.sif"

# Hopper
export TORCH_CUDA_ARCH_LIST="9.0"

# pass-through to container
export SINGULARITYENV_NCCL_DEBUG="$NCCL_DEBUG"
export SINGULARITYENV_DIST_INIT_ADDR="$DIST_INIT_ADDR"
export SINGULARITYENV_TORCH_CUDA_ARCH_LIST="$TORCH_CUDA_ARCH_LIST"


readarray -t NODES < <(scontrol show hostnames "$SLURM_NODELIST")
NODE0="${NODES[0]}"
NODE1="${NODES[1]}"
echo "NODE0=$NODE0 NODE1=$NODE1"

time srun -N 2 -n 2 -w "$NODE0","$NODE1" --label --cpu-bind=none \
  --ntasks-per-node=1 \
  bash -lc '
    set -euo pipefail
    ulimit -l unlimited || true

    # Clean any stale FlashInfer cache (fixes corrupted .so)
    rm -rf "$HOME/.cache/flashinfer/"* || true

    # Per-rank Triton cache on local node tmp
    CACHE_ROOT="${SLURM_TMPDIR:-${TMPDIR:-/tmp}}/triton/${SLURM_JOB_ID}/node${SLURM_NODEID}"
    export TRITON_CACHE_DIR="${CACHE_ROOT}/rank${SLURM_PROCID:-0}"
    mkdir -p "$TRITON_CACHE_DIR"
    echo "TRITON_CACHE_DIR=$TRITON_CACHE_DIR"

    # Torch profiler dir (optional)
    export SGLANG_TORCH_PROFILER_DIR="$HOME/sgprofile/node${SLURM_NODEID}"
    mkdir -p "$SGLANG_TORCH_PROFILER_DIR"
    echo "profile dir: $SGLANG_TORCH_PROFILER_DIR"

    # Flashinfer per rank storing
    export FLASHINFER_JIT_DIR="$CACHE_ROOT/flashinfer"
    export XDG_CACHE_HOME="$CACHE_ROOT/xdg"      # in case FlashInfer falls back to XDG
    mkdir -p "$FLASHINFER_JIT_DIR" "$XDG_CACHE_HOME"
    export SINGULARITYENV_FLASHINFER_JIT_DIR="$FLASHINFER_JIT_DIR"
    export SINGULARITYENV_XDG_CACHE_HOME="$XDG_CACHE_HOME"

    mkdir -p "$CACHE_ROOT/torch_ext"


    BIND_CACHE="--bind $CACHE_ROOT/flashinfer:$HOME/.cache/flashinfer \
                --bind $CACHE_ROOT/torch_ext:$HOME/.cache/torch/extensions"

    # Pass into container
    export SINGULARITYENV_TRITON_CACHE_DIR="$TRITON_CACHE_DIR"
    export SINGULARITYENV_SGLANG_TORCH_PROFILER_DIR="$SGLANG_TORCH_PROFILER_DIR"
    export SINGULARITYENV_SLURM_NODEID="$SLURM_NODEID"
    export SINGULARITYENV_SLURM_PROCID="$SLURM_PROCID"
    export SINGULARITYENV_SLURM_JOB_ID="$SLURM_JOB_ID"
    export SINGULARITYENV_SLURM_JOB_NUM_NODES="$SLURM_JOB_NUM_NODES"

    # Run inside the container
    singularity exec --nv $BIND_CACHE $HOME/sglang_v0.5.3rc1-cu126.sif bash -lc "
      set -euo pipefail
      python3 -m sglang.launch_server \
        --model-path deepseek-ai/DeepSeek-R1 \
        --load-format dummy --dtype bfloat16 \
        --tp 8 --trust-remote-code --port 30000 \
    "
  '
