#!/bin/bash
# Missing data point: venv (no container), 1 node, TP8.
# Same venv/CUDA setup as AITaskFinalSubmission/default/default.sh (the 2N TP16 baseline),
# same benchmark args as firmus/singularity-docker/onenode.sh (the container 1N TP8 run),
# so the only change vs each is container-vs-venv or node-count/TP respectively.
# Submit with: sbatch venv_1node.sh
#
# v2 after job 10621 OOMed mid-decode (flash_attn wanted 3.92 GiB, 135.9/139.8 GiB in
# use, 3.5 GiB fragmented). venv SGLang 0.5.2 defaults leave less decode headroom at
# 1N TP8 than the 0.5.3rc1 container. Two changes, note both in NUMBERS.md:
#   1. PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True  (reclaims fragmented memory)
#   2. --mem-fraction-static 0.87                        (decode headroom)
# If it still OOMs, drop to 0.85. If you want a purer defaults run, try 1. alone first.
#SBATCH -p hpcai
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=nvidia_h200:8
#SBATCH --exclusive
#SBATCH -t 00:45:00
#SBATCH -o slurm-%j-venv-1node.out
#SBATCH -e slurm-%j-venv-1node.err
#SBATCH --hint=multithread

# Make sure you set CUDA_HOME and VENV environment variables :)
export VENV="$HOME/py312"

export CUDA_HOME="/scratch/public/nvidia/cuda/cuda-12.9"
export PATH="$VENV/bin:$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"
export TORCH_CUDA_ARCH_LIST="9.0"

which nvcc || true
nvcc --version || true
nvidia-smi -L || true

export NCCL_DEBUG=WARN
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export DIST_INIT_ADDR="$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)"
echo "DIST_INIT_ADDR=${DIST_INIT_ADDR}"

time srun --label --cpu-bind=none \
  --nodes=$SLURM_JOB_NUM_NODES --ntasks-per-node=1 \
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

    export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

    time '"${VENV}"'/bin/python3 -m sglang.bench_offline_throughput \
      --model-path deepseek-ai/DeepSeek-R1 \
      --dataset-path $HOME/ShareGPT_V3_unfiltered_cleaned_split.json \
      --num-prompts 2000 --load-format dummy --seed 2025 --dtype bfloat16 \
      --tp 8 --nnodes 1 --trust-remote-code \
      --mem-fraction-static 0.87 \
      --dist-init-addr '"${DIST_INIT_ADDR}"':5000 --node-rank $SLURM_NODEID
  '
