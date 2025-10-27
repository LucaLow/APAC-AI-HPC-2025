#!/bin/bash
#SBATCH -p hpcai
#SBATCH -N 2
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=nvidia_h200:8
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --cpus-per-task=128
#SBATCH -t 04:00:00
#SBATCH -o slurm-%j.out
#SBATCH -e slurm-%j.err
#SBATCH --hint=multithread

### --- User env (adjust if needed) ---
export VENV="$HOME/py312"
export CUDA_HOME="$HOME/cuda-12.9"
export PATH="$VENV/bin:$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"
export TORCH_CUDA_ARCH_LIST="9.0"

### --- Fixed SGLang params (do NOT tune here) ---
MODEL="deepseek-ai/DeepSeek-R1"
DATA="$HOME/ShareGPT_V3_unfiltered_cleaned_split.json"
DTYPE="bfloat16"
TP=16

### --- Rendezvous addr across nodes ---
export DIST_INIT_ADDR="$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)"
echo "DIST_INIT_ADDR=$DIST_INIT_ADDR"

### --- Global comm defaults (constant) ---
export NCCL_NET=IB
export NCCL_DEBUG=WARN
export NCCL_ASYNC_ERROR_HANDLING=1
export TORCH_DISTRIBUTED_DEFAULT_TIMEOUT=180

### --- Search grids (NETWORKING ONLY) ---
NCCL_ALGOS=(Ring)
NCCL_PROTOS=(LL LL128 Simple) # -> NCCL_PROTO
NCCL_NTHREADS_LIST=(64 160 256)
NCCL_MIN_NCHANNELS_LIST=(4 8 16)
CUDA_DEV_MAX_CONN_LIST=(2)
NCCL_IB_RELAX_LIST=(1) # -> NCCL_IB_PCI_RELAXED_ORDERING
NCCL_SOCKET_NTHREADS_LIST=(1)
NCCL_NET_GDR_LEVEL_LIST=(2) # Found to be better than 0

### Limit trials if desired (unset/empty = try all)
: "${MAX_TRIALS:=}" # e.g. sbatch --export=MAX_TRIALS=48 net_sweep.sh

### --- Logging/outputs ---
JOBDIR="$HOME/net_sweep_${SLURM_JOB_ID:-manual}"
LOGDIR="$JOBDIR/logs"
mkdir -p "$LOGDIR"
CSV="$JOBDIR/results.csv"
echo "throughput_token_s,algo,proto,nthreads,min_nchannels,cuda_dev_max_conn,ib_relaxed,sock_threads,gdr_level,logfile" >"$CSV"

parse_thr() { # prints last "gen throughput (token/s)" found
	awk '
    /gen throughput \(token\/s\):/ {
      gsub(/,/, "", $0);
      for(i=1;i<=NF;i++) if($i ~ /^[0-9.]+$/) val=$i
    }
    END { if (val!="") print val; }
  ' "$1"
}

trial_count=0
best_thr=0
best_row=""

set -uo pipefail

for algo in "${NCCL_ALGOS[@]}"; do
	for proto in "${NCCL_PROTOS[@]}"; do
		for nthr in "${NCCL_NTHREADS_LIST[@]}"; do
			for nch in "${NCCL_MIN_NCHANNELS_LIST[@]}"; do
				for cdmx in "${CUDA_DEV_MAX_CONN_LIST[@]}"; do
					for ibrel in "${NCCL_IB_RELAX_LIST[@]}"; do
						for sockt in "${NCCL_SOCKET_NTHREADS_LIST[@]}"; do
							for gdr in "${NCCL_NET_GDR_LEVEL_LIST[@]}"; do

								((trial_count++))
								if [[ -n "$MAX_TRIALS" && $trial_count -gt $MAX_TRIALS ]]; then
									echo "Reached MAX_TRIALS=$MAX_TRIALS; stopping sweep."
									break 8
								fi

								tag="t${trial_count}_A${algo}_P${proto}_T${nthr}_C${nch}_M${cdmx}_R${ibrel}_S${sockt}_G${gdr}"
								of="$LOGDIR/${tag}.out"
								echo "==== Trial #$trial_count :: $tag ===="

								# Per-trial networking env
								export NCCL_ALGO="$algo"
								export NCCL_PROTO="$proto"
								export NCCL_NTHREADS="$nthr"
								export NCCL_MIN_NCHANNELS="$nch"
								export CUDA_DEVICE_MAX_CONNECTIONS="$cdmx"
								export NCCL_IB_PCI_RELAXED_ORDERING="$ibrel"
								export NCCL_SOCKET_NTHREADS="$sockt"
								export NCCL_NET_GDR_LEVEL="$gdr"

								# Cache dirs (avoid recomp races)
								export TRITON_CACHE_DIR="${SLURM_TMPDIR:-${TMPDIR:-/tmp}}/triton/${SLURM_JOB_ID:-$$}/${tag}"
								mkdir -p "$TRITON_CACHE_DIR"

								# Run one benchmark trial (both nodes)
								srun --label --cpu-bind=none \
									--nodes=${SLURM_JOB_NUM_NODES:-2} --ntasks-per-node=1 \
									--mpi=none --kill-on-bad-exit=0 --no-kill --wait=60 \
									bash -lc "
                    ulimit -l unlimited || true
                    time python3 -m sglang.bench_offline_throughput \
                      --model-path $MODEL \
                      --dataset-path $DATA \
                      --num-prompts 2000 --load-format dummy --seed 2025 \
                      --dtype $DTYPE --tp $TP --nnodes ${SLURM_JOB_NUM_NODES:-2} --trust-remote-code \
                      --dist-init-addr ${DIST_INIT_ADDR}:5000 --node-rank \$SLURM_NODEID || true
                    sleep 2
                  " &>"$of"

								thr="$(parse_thr "$of")"
								thr="${thr:-0}"
								echo "$thr,$algo,$proto,$nthr,$nch,$cdmx,$ibrel,$sockt,$gdr,$of" | tee -a "$CSV" >/dev/null

								# track best
								awk -v t="$thr" 'BEGIN{ if (t+0 > 0) exit 0; else exit 1 }' || true
								if [[ "$thr" != "" ]]; then
									awk "BEGIN{exit !($thr > $best_thr)}" || true
									if [[ $? -eq 0 ]]; then
										best_thr="$thr"
										best_row="$thr,$algo,$proto,$nthr,$nch,$cdmx,$ibrel,$sockt,$gdr,$of"
									fi
								fi

							done
						done
					done
				done
			done
		done
	done
done

echo
echo "==== Sweep complete: $trial_count trials ===="
echo "CSV saved to: $CSV"
echo
echo "Top 5 by throughput:"
# header + top 5
{
	head -n1 "$CSV"
	tail -n +2 "$CSV" | sort -t, -k1,1nr | head -n5
} | column -s, -t
echo
echo "Best:"
echo "$best_row" | column -s, -t
