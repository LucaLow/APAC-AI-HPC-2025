#!/bin/bash
#SBATCH -p hpcai
#SBATCH -N 2
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=nvidia_h200:8
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --cpus-per-task=128
#SBATCH -t 00:45:00
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
#export NCCL_CUMEM_ENABLE=1

# NCCL (try both algos in separate runs)
export NCCL_DEBUG=WARN
#export NCCL_IB_PCI_RELAXED_ORDERING=1
#export NCCL_NET_GDR_LEVEL=2
#export NCCL_NTHREADS=128
#export NCCL_MIN_NCHANNELS=8
#export NCCL_SOCKET_NTHREADS=2
# Optional, if dual-rail IB:
# export NCCL_CROSS_NIC=1
# A/B these:
# export NCCL_ALGO=Tree    # often better for latency-bound decode
# export NCCL_ALGO=Ring    # often better for bandwidth-bound prefill

# CUDA/CPU
#export CUDA_DEVICE_MAX_CONNECTIONS=1
#export OMP_NUM_THREADS=2
#export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128,expandable_segments:True

export DIST_INIT_ADDR="$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)"
echo "DIST_INIT_ADDR=${DIST_INIT_ADDR}"



LOGDIR="${SLURM_SUBMIT_DIR:-$PWD}/memfrac_logs/${SLURM_JOB_ID}"
mkdir -p "$LOGDIR"

best=""
for MF in $(seq 0.90 0.01 0.99); do
  echo "=== Trying --mem-fraction-static=${MF} ==="
  LOGFILE="${LOGDIR}/memfrac_${MF}.log"

  set +e
  srun --label --cpu-bind=none \
    --nodes=$SLURM_JOB_NUM_NODES --ntasks-per-node=1 \
    --mpi=pmix \
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

      '"${VENV}"'/bin/python3 -m sglang.bench_offline_throughput \
        --model-path deepseek-ai/DeepSeek-R1 \
        --dataset-path $HOME/ShareGPT_V3_unfiltered_cleaned_split.json \
        --num-prompts 2000 --load-format dummy --seed 2025 --dtype bfloat16 \
        --tp 16 --nnodes '"$SLURM_JOB_NUM_NODES"' --trust-remote-code \
        --cuda-graph-max-bs 512 \
        --mem-fraction-static '"$MF"' \
        --dist-init-addr '"${DIST_INIT_ADDR}"':5000 --node-rank $SLURM_NODEID
    ' 2>&1 | tee "$LOGFILE"
  rc=${PIPESTATUS[0]}
  set -e

  # treat common distributed-after-OOM symptoms as OOM too
  if grep -qiE \
    "out of memory|CUDA error: out of memory|torch\.cuda\.OutOfMemoryError|CUBLAS_STATUS_ALLOC_FAILED|OOM|NCCL.*unhandled cuda error|Rendezvous.*failed|ProcessGroupNCCL.*aborted" \
    "$LOGFILE"; then
      echo "OOM (or propagated abort) at --mem-fraction-static=${MF}; stopping."
      break
  fi
  if [ "$rc" -ne 0 ] && [ "$rc" -ne 137 ]; then
      echo "Non-OOM failure (exit $rc) at --mem-fraction-static=${MF}; aborting."
      exit "$rc"
  else
      best="$MF"
  fi
done

if [ -n "$best" ]; then
  echo "Highest successful --mem-fraction-static was ${best}"
else
  echo "No successful run at or above 0.90"
fi

