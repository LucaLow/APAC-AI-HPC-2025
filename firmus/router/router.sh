#!/bin/bash
#SBATCH -p hpcai
#SBATCH -N 2
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=nvidia_h200:8
#SBATCH --exclusive
#SBATCH -t 02:00:00
#SBATCH -o slurm-%j.out
#SBATCH -e slurm-%j.err
#SBATCH --hint=multithread

set -euo pipefail

# -------- base env --------
export NCCL_DEBUG=WARN
export DIST_INIT_ADDR="$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)"
export TORCH_CUDA_ARCH_LIST="9.0" # Hopper

export SINGULARITYENV_NCCL_DEBUG="$NCCL_DEBUG"
export SINGULARITYENV_DIST_INIT_ADDR="$DIST_INIT_ADDR"
export SINGULARITYENV_TORCH_CUDA_ARCH_LIST="$TORCH_CUDA_ARCH_LIST"

# -------- knobs --------
IMAGE="${IMAGE:-$HOME/sglang_v0.5.3rc1-cu126.sif}"

MODEL="${MODEL:-deepseek-ai/DeepSeek-R1}"
DTYPE="${DTYPE:-bfloat16}"
TP="${TP:-8}"

SERVER_PORT="${SERVER_PORT:-30000}"
ROUTER_PORT="${ROUTER_PORT:-31000}"
ROUTER_POLICY="${ROUTER_POLICY:-cache_aware}"

DATASET_PATH="${DATASET_PATH:-$HOME/ShareGPT_V3_unfiltered_cleaned_split.json}"
NUM_PROMPTS="${NUM_PROMPTS:-2000}"
SEED="${SEED:-2025}"

OUT_DIR="${OUT_DIR:-$PWD}"
OUT_JSON="${OUT_DIR}/bench_${SLURM_JOB_ID}.jsonl"

# -------- nodes --------
readarray -t NODES < <(scontrol show hostnames "$SLURM_NODELIST")
NODE0="${NODES[0]}"
NODE1="${NODES[1]}"
echo "NODE0=$NODE0 NODE1=$NODE1"

# -------- helpers --------
wait_ready() {
	local url="$1" tries="${2:-180}"
	until curl -fsS "$url" >/dev/null 2>&1; do
		((tries--)) || {
			echo "Timeout waiting for $url"
			return 1
		}
		sleep 2
	done
}

# -------- launch one server per node --------
srun -N2 -n2 -w "$NODE0","$NODE1" --label --overlap --cpu-bind=none \
	--ntasks-per-node=1 \
	bash -lc '
    set -euo pipefail
    ulimit -l unlimited || true

    # Clean any stale FlashInfer cache
    rm -rf "$HOME/.cache/flashinfer/"* || true

    CACHE_ROOT="${SLURM_TMPDIR:-${TMPDIR:-/tmp}}/triton/${SLURM_JOB_ID}/node${SLURM_NODEID}"
    export TRITON_CACHE_DIR="${CACHE_ROOT}/rank${SLURM_PROCID:-0}"
    export SGLANG_TORCH_PROFILER_DIR="$HOME/sgprofile/node${SLURM_NODEID}"
    export FLASHINFER_JIT_DIR="$CACHE_ROOT/flashinfer"
    export XDG_CACHE_HOME="$CACHE_ROOT/xdg"
    mkdir -p "$TRITON_CACHE_DIR" "$SGLANG_TORCH_PROFILER_DIR" "$FLASHINFER_JIT_DIR" "$XDG_CACHE_HOME" "$CACHE_ROOT/torch_ext"

    BIND_CACHE="--bind $CACHE_ROOT/flashinfer:$HOME/.cache/flashinfer \
                --bind $CACHE_ROOT/torch_ext:$HOME/.cache/torch/extensions"

    export SINGULARITYENV_TRITON_CACHE_DIR="$TRITON_CACHE_DIR"
    export SINGULARITYENV_SGLANG_TORCH_PROFILER_DIR="$SGLANG_TORCH_PROFILER_DIR"
    export SINGULARITYENV_FLASHINFER_JIT_DIR="$FLASHINFER_JIT_DIR"
    export SINGULARITYENV_XDG_CACHE_HOME="$XDG_CACHE_HOME"
    export SINGULARITYENV_SLURM_NODEID="$SLURM_NODEID"
    export SINGULARITYENV_SLURM_PROCID="$SLURM_PROCID"
    export SINGULARITYENV_SLURM_JOB_ID="$SLURM_JOB_ID"
    export SINGULARITYENV_SLURM_JOB_NUM_NODES="$SLURM_JOB_NUM_NODES"

    singularity exec --nv $BIND_CACHE "'"$IMAGE"'" bash -lc "
      set -euo pipefail
      python3 -m sglang.launch_server \
        --model-path "'"$MODEL"'" \
        --dtype "'"$DTYPE"'" \
        --tp "'"$TP"'" \
        --host 0.0.0.0 \
        --port "'"$SERVER_PORT"'" \
        --trust-remote-code \
        --load-format dummy \
   	--max-running-requests 4096
    "
  ' &

# Wait for both servers
echo "Waiting for servers..."
wait_ready "http://$NODE0:$SERVER_PORT/health"
wait_ready "http://$NODE1:$SERVER_PORT/health"
echo "Servers healthy."

# -------- launch router on node0 (background) --------
srun -N1 -n1 -w "$NODE0" --overlap --cpu-bind=none \
	bash -lc "
    singularity exec --nv $IMAGE python3 -m sglang_router.launch_router \
      --worker-urls http://$NODE0:$SERVER_PORT http://$NODE1:$SERVER_PORT \
      --host 0.0.0.0 \
      --port $ROUTER_PORT \
      --max-concurrent-requests 8192 \
      --policy $ROUTER_POLICY
  " &

# Wait for router
echo "Waiting for router..."
wait_ready "http://$NODE0:$ROUTER_PORT/health"
echo "Router healthy at $NODE0:$ROUTER_PORT"

# -------- run the benchmark from node0 --------
# Uses sglang native /generate backend; ShareGPT dataset file you specified.
# --- BENCHMARK (OpenAI chat endpoint via router) ---
srun -N1 -n1 -w "$NODE0" --cpu-bind=none bash -lc "
  test -f '$DATASET_PATH' || { echo 'Dataset not found: $DATASET_PATH' >&2; exit 2; }
  mkdir -p '$OUT_DIR'

  # Native endpoint variant (no OpenAI path)
  singularity exec --nv $IMAGE python3 -m sglang.bench_serving \
  --backend sglang \
  --host $NODE0 --port $ROUTER_PORT \
  --model $MODEL \
  --dataset-name sharegpt \
  --dataset-path "$DATASET_PATH" \
  --num-prompts $NUM_PROMPTS \
  --seed $SEED \
  --request-rate inf --max-concurrency 256 --warmup-requests 32 \
  --output-file "$OUT_JSON" --output-details
"

echo "Benchmark complete. JSONL: $OUT_JSON"
