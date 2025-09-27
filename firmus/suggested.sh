#!/bin/bash
#SBATCH -p hpcai
#SBATCH -N 2
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=nvidia_h200:8
#SBATCH --exclusive
#SBATCH -t 00:10:00
#SBATCH -o slurm-%j.out
#SBATCH -e slurm-%j.err
#SBATCH --hint=multithread

# module load cuda

export VENV="$HOME/py312"
export CUDA_HOME="$HOME/cuda-12.9"
export PATH="$VENV/bin:$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"
export TORCH_CUDA_ARCH_LIST="9.0"

echo "Allocated nodes: $SLURM_NODELIST"
echo "Job ID: $SLURM_JOB_ID"

which nvcc || true
nvcc --version || true
nvidia-smi -L || true
which ninja || true
ninja --version || true

export NCCL_DEBUG=INFO
export DIST_INIT_ADDR="$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)"
echo "DIST_INIT_ADDR=${DIST_INIT_ADDR}"

time srun --label --cpu-bind=none \
  --nodes=$SLURM_JOB_NUM_NODES --ntasks-per-node=1 \
  bash -lc '
    ulimit -l unlimited || true
    echo "memlock (kB): $(ulimit -l)"

    set -euo pipefail
    # Per-node, per-rank Triton JIT cache on local storage (avoids NFS races)
    CACHE_ROOT="${SLURM_TMPDIR:-${TMPDIR:-/tmp}}/triton/${SLURM_JOB_ID}/node${SLURM_NODEID}"
    export TRITON_CACHE_DIR="${CACHE_ROOT}/rank${SLURM_PROCID:-0}"
    mkdir -p "$TRITON_CACHE_DIR"
    echo "TRITON_CACHE_DIR=$TRITON_CACHE_DIR"
    # Optional cleanup:
    # trap "rm -rf \"$CACHE_ROOT\"" EXIT

    time '"${VENV}"'/bin/python3 -m sglang.bench_offline_throughput \
      --model-path deepseek-ai/DeepSeek-R1 \
      --dataset-path $HOME/ShareGPT_V3_unfiltered_cleaned_split.json \
      --num-prompts 2000 --load-format dummy --seed 2025 --dtype bfloat16 \
      --tp 16 --nnodes '"$SLURM_JOB_NUM_NODES"' --trust-remote-code \
      --mem-fraction-static 0.4 \
      --dist-init-addr '"${DIST_INIT_ADDR}"':5000 --node-rank $SLURM_NODEID
  ' 2>&1 | tee ${HOME}/run/stdout.sglang.${SLURM_JOB_ID}

