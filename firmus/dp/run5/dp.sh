#!/bin/bash
#SBATCH -p hpcai
#SBATCH -N 2
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=nvidia_h200:8
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --cpus-per-task=128
#SBATCH -t 00:10:00
#SBATCH -o slurm-%j.out
#SBATCH -e slurm-%j.err
#SBATCH --hint=multithread

export VENV="$HOME/py312"
export CUDA_HOME="$HOME/cuda-12.9"
export PATH="$VENV/bin:$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"
export TORCH_CUDA_ARCH_LIST="9.0"

which nvcc || true
nvcc --version || true
nvidia-smi -L || true
#export NCCL_NVLS_ENABLE=1

export NCCL_NET=IB


# then try this (if your NCCL+CUDA support it; otherwise it’s ignored)
export NCCL_CUMEM_ENABLE=1


# NCCL (try both algos in separate runs)
#export NCCL_DEBUG=WARN
#export NCCL_IB_PCI_RELAXED_ORDERING=1
#export NCCL_NET_GDR_LEVEL=2
#export NCCL_NTHREADS=128
#export NCCL_MIN_NCHANNELS=8
#export NCCL_SOCKET_NTHREADS=2
# Optional, if dual-rail IB:
# export NCCL_CROSS_NIC=1
# A/B these:
#export NCCL_ALGO=Tree    # often better for latency-bound decode
#export NCCL_ALGO=Ring    # often better for bandwidth-bound prefill

# CUDA/CPU
#export CUDA_DEVICE_MAX_CONNECTIONS=1
#export OMP_NUM_THREADS=2
#export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128,expandable_segments:True


export DIST_INIT_ADDR="$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)"
echo "DIST_INIT_ADDR=${DIST_INIT_ADDR}"

time srun --label --cpu-bind=none \
  --nodes=$SLURM_JOB_NUM_NODES --ntasks-per-node=1 \
  --mpi=none --kill-on-bad-exit=0 --no-kill --wait=60 \
  bash -lc '
    ulimit -l unlimited || true
    set -euo pipefail

    
    CACHE_ROOT="${SLURM_TMPDIR:-${TMPDIR:-/tmp}}/triton/${SLURM_JOB_ID}/node${SLURM_NODEID}"
    export TRITON_CACHE_DIR="${CACHE_ROOT}/rank${SLURM_PROCID:-0}"
    mkdir -p "$TRITON_CACHE_DIR"

    echo "TRITON_CACHE_DIR=$TRITON_CACHE_DIR"

    
    export SGLANG_TORCH_PROFILER_DIR="$HOME/sgprofile/node${SLURM_NODEID}"
    mkdir -p "$SGLANG_TORCH_PROFILER_DIR"
    
    echo "profile dir: $SGLANG_TORCH_PROFILER_DIR"


    time '"${VENV}"'/bin/python3 -m sglang.bench_offline_throughput \
      --model-path deepseek-ai/DeepSeek-R1 \
      --dataset-path $HOME/ShareGPT_V3_unfiltered_cleaned_split.json \
      --num-prompts 2000 --load-format dummy --seed 2025 --dtype bfloat16 \
      --tp 16 --nnodes '"$SLURM_JOB_NUM_NODES"' --trust-remote-code \
      --dp 2 --enable-dp-lm-head --enable-dp-attention --cuda-graph-max-bs 512 \
      --dist-init-addr '"${DIST_INIT_ADDR}"':5000 --node-rank $SLURM_NODEID
    sleep 3
  '

