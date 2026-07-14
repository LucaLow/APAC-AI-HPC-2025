#!/usr/bin/env bash
# Firmus H200 sample: five independent offline repetitions for one topology.
#
# Topology: 1 node, 8 worker GPUs, TP=4, DP=2, PP=2.
# Metrics: offline throughput, GPU energy, power, HBM, utilization, temperature.
#
# Submit all five repetitions:
#   sbatch sample_tp4_dp2_pp2_offline_metrics.sbatch
#
# Submit only one repetition while validating:
#   sbatch --array=1 sample_tp4_dp2_pp2_offline_metrics.sbatch

#SBATCH -p hpcai
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=nvidia_h200:8
#SBATCH --exclusive
#SBATCH --hint=multithread
#SBATCH --array=1-5%1
#SBATCH -t 01:00:00
#SBATCH -J sglang-offline-metrics
#SBATCH -o sglang-offline-metrics-%A_%a.out
#SBATCH -e sglang-offline-metrics-%A_%a.err

set -euo pipefail

IMAGE="${IMAGE:-$HOME/sglang_v0.5.3rc1-cu126.sif}"
DATASET="${DATASET:-$HOME/ShareGPT_V3_unfiltered_cleaned_split.json}"
MODEL="${MODEL:-deepseek-ai/DeepSeek-R1}"
NUM_PROMPTS="${NUM_PROMPTS:-2000}"
SEED="${SEED:-2025}"
TELEMETRY_INTERVAL_S="${TELEMETRY_INTERVAL_S:-1.0}"
IDLE_BASELINE_S="${IDLE_BASELINE_S:-10}"

TP=4
DP=2
PP=2
NODES=1
GPUS_USED=8

REPEAT="${SLURM_ARRAY_TASK_ID:-${REPEAT:-1}}"
ARRAY_ID="${SLURM_ARRAY_JOB_ID:-$SLURM_JOB_ID}"
SUBMIT_DIR="${SLURM_SUBMIT_DIR:-$PWD}"
CAMPAIGN_ROOT="${OUTPUT_ROOT:-$SUBMIT_DIR/outputs/firmus_h200_singularity_matrix/sample_tp4_dp2_pp2_${ARRAY_ID}}"
RUN_DIR="$CAMPAIGN_ROOT/repeat_$(printf '%02d' "$REPEAT")"
# Compilation caches must be node-local. A shared NFS/scratch cache can race
# when the eight SGLang ranks compile Triton kernels concurrently.
CACHE_ROOT="${SLURM_TMPDIR:-${TMPDIR:-/tmp}}/sglang-offline-metrics/${SLURM_JOB_ID}/node${SLURM_NODEID:-0}"
TELEM_DIR="$RUN_DIR/telemetry"

mkdir -p "$RUN_DIR" "$TELEM_DIR" \
  "$CACHE_ROOT/triton" "$CACHE_ROOT/flashinfer" \
  "$CACHE_ROOT/torch_extensions" \
  "$CACHE_ROOT/xdg"

[[ -f "$IMAGE" ]] || { echo "Missing image: $IMAGE" >&2; exit 2; }
[[ -f "$DATASET" ]] || { echo "Missing dataset: $DATASET" >&2; exit 2; }
command -v singularity >/dev/null || { echo "singularity is unavailable" >&2; exit 2; }
command -v srun >/dev/null || { echo "srun is unavailable" >&2; exit 2; }
command -v timeout >/dev/null || { echo "timeout is unavailable" >&2; exit 2; }

MASTER_HOST="$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)"
DIST_PORT="$((20000 + SLURM_JOB_ID % 20000))"

export IMAGE DATASET MODEL NUM_PROMPTS SEED
export TP DP PP NODES GPUS_USED RUN_DIR CACHE_ROOT CAMPAIGN_ROOT MASTER_HOST DIST_PORT
export NCCL_DEBUG=WARN
export NCCL_NET_GDR_LEVEL=5
export TORCH_CUDA_ARCH_LIST=9.0
export TRITON_CACHE_DIR="$CACHE_ROOT/triton"
export FLASHINFER_JIT_DIR="$CACHE_ROOT/flashinfer"
export TORCH_EXTENSIONS_DIR="$CACHE_ROOT/torch_extensions"
export XDG_CACHE_HOME="$CACHE_ROOT/xdg"
export SINGULARITYENV_NCCL_DEBUG="$NCCL_DEBUG"
export SINGULARITYENV_NCCL_NET_GDR_LEVEL="$NCCL_NET_GDR_LEVEL"
export SINGULARITYENV_TORCH_CUDA_ARCH_LIST="$TORCH_CUDA_ARCH_LIST"
export SINGULARITYENV_TRITON_CACHE_DIR="$TRITON_CACHE_DIR"
export SINGULARITYENV_FLASHINFER_JIT_DIR="$FLASHINFER_JIT_DIR"
export SINGULARITYENV_TORCH_EXTENSIONS_DIR="$TORCH_EXTENSIONS_DIR"
export SINGULARITYENV_XDG_CACHE_HOME="$XDG_CACHE_HOME"

printf '%s\n' \
  "array_id=$ARRAY_ID" \
  "job_id=$SLURM_JOB_ID" \
  "repeat=$REPEAT" \
  "nodes=$NODES" \
  "gpus_used=$GPUS_USED" \
  "tp=$TP" \
  "dp=$DP" \
  "pp=$PP" \
  "master_host=$MASTER_HOST" \
  "dist_port=$DIST_PORT" \
  "cache_root=$CACHE_ROOT" \
  "image=$IMAGE" \
  "dataset=$DATASET" \
  >"$RUN_DIR/run.env"

sha256sum "$IMAGE" >"$RUN_DIR/image.sha256"
singularity --version >"$RUN_DIR/singularity.version.txt" 2>&1
nvidia-smi -L >"$RUN_DIR/gpus.txt" 2>&1
nvidia-smi --query-gpu=index,uuid,name,memory.total,power.limit \
  --format=csv,noheader,nounits >"$RUN_DIR/gpu_inventory.csv" 2>&1

[[ "$(wc -l <"$RUN_DIR/gpus.txt")" -eq 8 ]] || {
  echo "Expected eight visible GPUs" >&2
  exit 3
}
if grep -viq 'H200' "$RUN_DIR/gpus.txt"; then
  echo "At least one visible GPU is not an H200" >&2
  exit 3
fi

singularity exec --nv --bind "$CAMPAIGN_ROOT:$CAMPAIGN_ROOT" "$IMAGE" \
  python3 -c 'import json, platform, sglang, torch; print(json.dumps({"python": platform.python_version(), "sglang": getattr(sglang, "__version__", "unknown"), "torch": torch.__version__, "cuda": torch.version.cuda, "nccl": torch.cuda.nccl.version()}))' \
  >"$RUN_DIR/runtime_versions.json" 2>"$RUN_DIR/runtime_versions.stderr.log"

offline_command() {
  local result_file=$1
  local timeout_s=$2
  timeout --signal=TERM --kill-after=30s "$timeout_s" \
    srun --kill-on-bad-exit=1 --label --cpu-bind=socket \
    --nodes=1 --ntasks=1 \
    singularity exec --nv \
      --bind "$CAMPAIGN_ROOT:$CAMPAIGN_ROOT" \
      --bind "$CACHE_ROOT:$CACHE_ROOT" \
      "$IMAGE" \
      python3 -m sglang.bench_offline_throughput \
        --model-path "$MODEL" \
        --dataset-path "$DATASET" \
        --num-prompts "$NUM_PROMPTS" \
        --load-format dummy \
        --seed "$SEED" \
        --dtype bfloat16 \
        --tp "$TP" --dp "$DP" --pp "$PP" \
        --enable-dp-attention \
        --enable-torch-compile \
        --nnodes 1 \
        --node-rank 0 \
        --dist-init-addr "$MASTER_HOST:$DIST_PORT" \
        --trust-remote-code \
        --result-filename "$result_file"
}

# Dependency-free NVML sidecar. Cumulative energy is reported in mJ; allocated
# HBM is bytes; utilization.memory is bandwidth activity rather than capacity.
NVML_HELPER="$RUN_DIR/nvml_monitor.py"
cat >"$NVML_HELPER" <<'PY'
import argparse
import csv
import ctypes
import os
import signal
import socket
import time


class Memory(ctypes.Structure):
    _fields_ = [("total", ctypes.c_ulonglong), ("free", ctypes.c_ulonglong), ("used", ctypes.c_ulonglong)]


class Utilization(ctypes.Structure):
    _fields_ = [("gpu", ctypes.c_uint), ("memory", ctypes.c_uint)]


HEADER = [
    "wall_epoch_ns", "monotonic_ns", "phase", "hostname", "gpu_index",
    "gpu_uuid", "gpu_name", "energy_mj", "power_mw", "power_limit_mw",
    "memory_used_bytes", "memory_total_bytes", "gpu_util_pct",
    "memory_bw_util_pct", "temperature_c",
]


def configure(lib):
    handle = ctypes.c_void_p
    lib.nvmlDeviceGetUUID.argtypes = [handle, ctypes.c_char_p, ctypes.c_uint]
    lib.nvmlDeviceGetName.argtypes = [handle, ctypes.c_char_p, ctypes.c_uint]
    lib.nvmlDeviceGetPowerUsage.argtypes = [handle, ctypes.POINTER(ctypes.c_uint)]
    lib.nvmlDeviceGetEnforcedPowerLimit.argtypes = [handle, ctypes.POINTER(ctypes.c_uint)]
    lib.nvmlDeviceGetMemoryInfo.argtypes = [handle, ctypes.POINTER(Memory)]
    lib.nvmlDeviceGetUtilizationRates.argtypes = [handle, ctypes.POINTER(Utilization)]
    lib.nvmlDeviceGetTemperature.argtypes = [handle, ctypes.c_uint, ctypes.POINTER(ctypes.c_uint)]
    if hasattr(lib, "nvmlDeviceGetTotalEnergyConsumption"):
        lib.nvmlDeviceGetTotalEnergyConsumption.argtypes = [handle, ctypes.POINTER(ctypes.c_ulonglong)]


def scalar(func, value_type, *args):
    value = value_type()
    try:
        status = func(*args, ctypes.byref(value))
    except Exception:
        return ""
    return value.value if status == 0 else ""


def text(func, device):
    value = ctypes.create_string_buffer(128)
    return value.value.decode("utf-8", "replace") if func(device, value, len(value)) == 0 else ""


def rows(lib, devices, phase):
    wall = time.time_ns()
    mono = time.monotonic_ns()
    host = socket.gethostname().split(".")[0]
    output = []
    for index, device in enumerate(devices):
        memory = Memory()
        utilization = Utilization()
        memory_ok = lib.nvmlDeviceGetMemoryInfo(device, ctypes.byref(memory)) == 0
        utilization_ok = lib.nvmlDeviceGetUtilizationRates(device, ctypes.byref(utilization)) == 0
        energy = ""
        if hasattr(lib, "nvmlDeviceGetTotalEnergyConsumption"):
            energy = scalar(lib.nvmlDeviceGetTotalEnergyConsumption, ctypes.c_ulonglong, device)
        output.append([
            wall, mono, phase, host, index,
            text(lib.nvmlDeviceGetUUID, device),
            text(lib.nvmlDeviceGetName, device),
            energy,
            scalar(lib.nvmlDeviceGetPowerUsage, ctypes.c_uint, device),
            scalar(lib.nvmlDeviceGetEnforcedPowerLimit, ctypes.c_uint, device),
            memory.used if memory_ok else "",
            memory.total if memory_ok else "",
            utilization.gpu if utilization_ok else "",
            utilization.memory if utilization_ok else "",
            scalar(lib.nvmlDeviceGetTemperature, ctypes.c_uint, device, 0),
        ])
    return output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    parser.add_argument("--phase", default="unknown")
    parser.add_argument("--phase-file")
    parser.add_argument("--stop")
    parser.add_argument("--ready")
    parser.add_argument("--interval", type=float, default=1.0)
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()

    lib = ctypes.CDLL("libnvidia-ml.so.1")
    init = getattr(lib, "nvmlInit_v2", None) or lib.nvmlInit
    if init() != 0:
        raise RuntimeError("nvmlInit failed")
    configure(lib)
    try:
        count = ctypes.c_uint()
        get_count = getattr(lib, "nvmlDeviceGetCount_v2", None) or lib.nvmlDeviceGetCount
        if get_count(ctypes.byref(count)) != 0:
            raise RuntimeError("nvmlDeviceGetCount failed")
        get_handle = getattr(lib, "nvmlDeviceGetHandleByIndex_v2", None) or lib.nvmlDeviceGetHandleByIndex
        devices = []
        for index in range(count.value):
            device = ctypes.c_void_p()
            if get_handle(index, ctypes.byref(device)) != 0:
                raise RuntimeError(f"Unable to open GPU {index}")
            devices.append(device)

        exists = os.path.exists(args.out) and os.path.getsize(args.out) > 0
        with open(args.out, "a", newline="", encoding="utf-8") as file:
            writer = csv.writer(file)
            if not exists:
                writer.writerow(HEADER)

            if args.once:
                writer.writerows(rows(lib, devices, args.phase))
                file.flush()
                return

            stopped = False

            def stop(_signum, _frame):
                nonlocal stopped
                stopped = True

            signal.signal(signal.SIGINT, stop)
            signal.signal(signal.SIGTERM, stop)
            if args.ready:
                open(args.ready, "w", encoding="utf-8").close()

            while not stopped and not (args.stop and os.path.exists(args.stop)):
                phase = args.phase
                if args.phase_file:
                    try:
                        phase = open(args.phase_file, encoding="utf-8").read().strip() or phase
                    except OSError:
                        pass
                writer.writerows(rows(lib, devices, phase))
                file.flush()
                deadline = time.monotonic() + args.interval
                while not stopped and time.monotonic() < deadline:
                    if args.stop and os.path.exists(args.stop):
                        stopped = True
                        break
                    time.sleep(min(0.1, max(0.0, deadline - time.monotonic())))
    finally:
        lib.nvmlShutdown()


if __name__ == "__main__":
    main()
PY

export NVML_HELPER
TELEM_STOP="$TELEM_DIR/.stop"
TELEM_PHASE="$TELEM_DIR/current_phase"
TELEM_READY="$TELEM_DIR/.ready"
TELEM_SAMPLES="$TELEM_DIR/gpu_samples_$(hostname -s).csv"
TELEM_BOUNDARIES="$TELEM_DIR/boundaries_$(hostname -s).csv"
export TELEM_STOP TELEM_PHASE TELEM_READY TELEM_SAMPLES TELEM_BOUNDARIES
rm -f "$TELEM_STOP" "$TELEM_READY"
printf 'allocation_idle\n' >"$TELEM_PHASE"

singularity exec --nv --bind "$CAMPAIGN_ROOT:$CAMPAIGN_ROOT" "$IMAGE" \
  python3 "$NVML_HELPER" \
    --out "$TELEM_SAMPLES" \
    --phase-file "$TELEM_PHASE" \
    --stop "$TELEM_STOP" \
    --ready "$TELEM_READY" \
    --interval "$TELEMETRY_INTERVAL_S" \
  >"$TELEM_DIR/monitor.stdout.log" 2>"$TELEM_DIR/monitor.stderr.log" &
TELEM_PID=$!

cleanup() {
  local status=$?
  touch "$TELEM_STOP"
  wait "${TELEM_PID:-}" 2>/dev/null || true
  exit "$status"
}
trap cleanup EXIT INT TERM

for _ in $(seq 1 120); do
  [[ -f "$TELEM_READY" ]] && break
  sleep 0.5
done
[[ -f "$TELEM_READY" ]] || { echo "NVML monitor failed to start" >&2; exit 70; }

mark_phase() {
  local phase=$1
  local temporary="$TELEM_DIR/.phase.$$"
  printf '%s\n' "$phase" >"$temporary"
  mv "$temporary" "$TELEM_PHASE"
  singularity exec --nv --bind "$CAMPAIGN_ROOT:$CAMPAIGN_ROOT" "$IMAGE" \
    python3 "$NVML_HELPER" --once --phase "$phase" --out "$TELEM_BOUNDARIES"
}

sleep "$IDLE_BASELINE_S"
mark_phase offline_begin
set +e
offline_command "$RUN_DIR/offline_result.jsonl" 420s \
  >"$RUN_DIR/offline.stdout.log" \
  2>"$RUN_DIR/offline.stderr.log"
OFFLINE_RC=$?
set -e
mark_phase offline_end
printf '%s\n' "$OFFLINE_RC" >"$RUN_DIR/offline.exit_code"

touch "$TELEM_STOP"
wait "$TELEM_PID" 2>/dev/null || true
TELEM_PID=""
trap - EXIT INT TERM

singularity exec --bind "$CAMPAIGN_ROOT:$CAMPAIGN_ROOT" "$IMAGE" \
  python3 - "$RUN_DIR" "$OFFLINE_RC" "$NUM_PROMPTS" <<'PY'
import csv
import json
import pathlib
import sys
from collections import defaultdict


run_dir = pathlib.Path(sys.argv[1])
exit_code = int(sys.argv[2])
expected_requests = int(sys.argv[3])


def numeric(value):
    if value in (None, "", "N/A", "[Not Supported]"):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


result_lines = []
result_path = run_dir / "offline_result.jsonl"
if result_path.exists():
    result_lines = [line for line in result_path.read_text(encoding="utf-8").splitlines() if line.strip()]
offline = json.loads(result_lines[-1]) if result_lines else {}

with open(next((run_dir / "telemetry").glob("boundaries_*.csv")), newline="", encoding="utf-8") as file:
    boundaries = list(csv.DictReader(file))
with open(next((run_dir / "telemetry").glob("gpu_samples_*.csv")), newline="", encoding="utf-8") as file:
    samples = list(csv.DictReader(file))

starts = {row["gpu_uuid"]: row for row in boundaries if row["phase"] == "offline_begin"}
ends = {row["gpu_uuid"]: row for row in boundaries if row["phase"] == "offline_end"}
uuids = sorted(set(starts) & set(ends))
if len(uuids) != 8:
    raise RuntimeError(f"Expected eight GPU telemetry boundaries, found {len(uuids)}")

start_ns = min(int(starts[uuid]["monotonic_ns"]) for uuid in uuids)
end_ns = max(int(ends[uuid]["monotonic_ns"]) for uuid in uuids)
end_to_end_s = (end_ns - start_ns) / 1e9

counter_deltas = []
for uuid in uuids:
    first = numeric(starts[uuid]["energy_mj"])
    last = numeric(ends[uuid]["energy_mj"])
    if first is None or last is None or last < first:
        counter_deltas = []
        break
    counter_deltas.append((last - first) / 1000.0)
counter_energy_j = sum(counter_deltas) if len(counter_deltas) == len(uuids) else None

phase_rows = [row for row in samples if row["phase"] == "offline_begin"]
power_by_gpu = defaultdict(list)
memory_by_time = defaultdict(float)
gpu_util = []
memory_bw_util = []
temperatures = []
per_gpu_peak_hbm = defaultdict(float)
for row in phase_rows:
    timestamp = int(row["monotonic_ns"])
    uuid = row["gpu_uuid"]
    power = numeric(row["power_mw"])
    memory = numeric(row["memory_used_bytes"])
    if power is not None:
        power_by_gpu[uuid].append((timestamp, power / 1000.0))
    if memory is not None:
        memory_by_time[timestamp] += memory
        per_gpu_peak_hbm[uuid] = max(per_gpu_peak_hbm[uuid], memory)
    value = numeric(row["gpu_util_pct"])
    if value is not None:
        gpu_util.append(value)
    value = numeric(row["memory_bw_util_pct"])
    if value is not None:
        memory_bw_util.append(value)
    value = numeric(row["temperature_c"])
    if value is not None:
        temperatures.append(value)

sampled_energy_j = 0.0
integrated_gpus = 0
for points in power_by_gpu.values():
    points.sort()
    if len(points) < 2:
        continue
    integrated_gpus += 1
    sampled_energy_j += sum(
        ((power0 + power1) / 2.0) * ((time1 - time0) / 1e9)
        for (time0, power0), (time1, power1) in zip(points, points[1:])
    )
if integrated_gpus != 8:
    sampled_energy_j = None

gpu_energy_j = counter_energy_j if counter_energy_j is not None else sampled_energy_j
energy_source = "nvml_cumulative_counter" if counter_energy_j is not None else "sampled_power_trapezoid_estimate"
total_tokens = offline.get("total_input_tokens", 0) + offline.get("total_output_tokens", 0)

accepted = (
    exit_code == 0
    and offline.get("successful_requests") == expected_requests
    and offline.get("total_throughput") is not None
    and gpu_energy_j is not None
)

summary = {
    "status": "accepted" if accepted else "invalid",
    "scenario": "1_node_8_gpus",
    "repeat": int(run_dir.name.split("_")[-1]),
    "nodes": 1,
    "gpus_allocated": 8,
    "gpus_used": 8,
    "tp": 4,
    "dp": 2,
    "pp": 2,
    "dp_attention": True,
    "exit_code": exit_code,
    "offline": offline,
    "telemetry_scope": "complete offline process: engine startup, internal warmup, and measured generate call",
    "end_to_end_s": end_to_end_s,
    "gpu_energy_j": gpu_energy_j,
    "energy_source": energy_source,
    "counter_gpu_energy_j": counter_energy_j,
    "sampled_gpu_energy_j": sampled_energy_j,
    "energy_counter_vs_sampled_difference_pct": (
        abs(counter_energy_j - sampled_energy_j) / counter_energy_j * 100
        if counter_energy_j not in (None, 0) and sampled_energy_j is not None else None
    ),
    "average_aggregate_gpu_power_w": gpu_energy_j / end_to_end_s if gpu_energy_j is not None else None,
    "peak_aggregate_hbm_gib": max(memory_by_time.values()) / (1024**3) if memory_by_time else None,
    "max_per_gpu_hbm_gib": max(per_gpu_peak_hbm.values()) / (1024**3) if per_gpu_peak_hbm else None,
    "mean_gpu_util_pct": sum(gpu_util) / len(gpu_util) if gpu_util else None,
    "mean_memory_bw_util_pct": sum(memory_bw_util) / len(memory_bw_util) if memory_bw_util else None,
    "max_temperature_c": max(temperatures) if temperatures else None,
    "joules_per_request": gpu_energy_j / expected_requests if gpu_energy_j is not None else None,
    "tokens_per_joule": total_tokens / gpu_energy_j if gpu_energy_j not in (None, 0) else None,
    "joules_per_1000_tokens": gpu_energy_j / total_tokens * 1000 if gpu_energy_j is not None and total_tokens else None,
    "raw_logs": {
        "stdout": str(run_dir / "offline.stdout.log"),
        "stderr": str(run_dir / "offline.stderr.log"),
        "telemetry": str(next((run_dir / "telemetry").glob("gpu_samples_*.csv"))),
        "boundaries": str(next((run_dir / "telemetry").glob("boundaries_*.csv"))),
    },
}
(run_dir / "run_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

flat = {
    "status": summary["status"],
    "repeat": summary["repeat"],
    "throughput_tok_s": offline.get("total_throughput"),
    "benchmark_duration_s": offline.get("total_latency"),
    "end_to_end_s": end_to_end_s,
    "gpu_energy_j": gpu_energy_j,
    "energy_source": energy_source,
    "average_aggregate_gpu_power_w": summary["average_aggregate_gpu_power_w"],
    "peak_aggregate_hbm_gib": summary["peak_aggregate_hbm_gib"],
    "max_per_gpu_hbm_gib": summary["max_per_gpu_hbm_gib"],
    "mean_gpu_util_pct": summary["mean_gpu_util_pct"],
    "max_temperature_c": summary["max_temperature_c"],
    "joules_per_request": summary["joules_per_request"],
    "tokens_per_joule": summary["tokens_per_joule"],
    "joules_per_1000_tokens": summary["joules_per_1000_tokens"],
}
with open(run_dir / "run_summary.csv", "w", newline="", encoding="utf-8") as file:
    writer = csv.DictWriter(file, fieldnames=list(flat))
    writer.writeheader()
    writer.writerow(flat)
PY

# The fifth accepted array task creates the aggregate files. Percentages and
# efficiency values remain available for every individual run as well.
singularity exec --bind "$CAMPAIGN_ROOT:$CAMPAIGN_ROOT" "$IMAGE" \
  python3 - "$CAMPAIGN_ROOT" <<'PY'
import csv
import json
import math
import pathlib
import statistics
import sys


root = pathlib.Path(sys.argv[1])
records = [
    json.loads(path.read_text(encoding="utf-8"))
    for path in sorted(root.glob("repeat_*/run_summary.json"))
]
accepted = [record for record in records if record.get("status") == "accepted"]

individual_fields = [
    "repeat", "status", "throughput_tok_s", "benchmark_duration_s",
    "end_to_end_s", "gpu_energy_j", "energy_source",
    "average_aggregate_gpu_power_w", "peak_aggregate_hbm_gib",
    "max_per_gpu_hbm_gib", "mean_gpu_util_pct", "max_temperature_c",
    "joules_per_request", "tokens_per_joule", "joules_per_1000_tokens",
    "stdout_path", "stderr_path", "telemetry_path",
]
individual_rows = []
for record in records:
    offline = record.get("offline", {})
    logs = record.get("raw_logs", {})
    individual_rows.append({
        "repeat": record.get("repeat"),
        "status": record.get("status"),
        "throughput_tok_s": offline.get("total_throughput"),
        "benchmark_duration_s": offline.get("total_latency"),
        "end_to_end_s": record.get("end_to_end_s"),
        "gpu_energy_j": record.get("gpu_energy_j"),
        "energy_source": record.get("energy_source"),
        "average_aggregate_gpu_power_w": record.get("average_aggregate_gpu_power_w"),
        "peak_aggregate_hbm_gib": record.get("peak_aggregate_hbm_gib"),
        "max_per_gpu_hbm_gib": record.get("max_per_gpu_hbm_gib"),
        "mean_gpu_util_pct": record.get("mean_gpu_util_pct"),
        "max_temperature_c": record.get("max_temperature_c"),
        "joules_per_request": record.get("joules_per_request"),
        "tokens_per_joule": record.get("tokens_per_joule"),
        "joules_per_1000_tokens": record.get("joules_per_1000_tokens"),
        "stdout_path": logs.get("stdout"),
        "stderr_path": logs.get("stderr"),
        "telemetry_path": logs.get("telemetry"),
    })
with open(root / "individual_runs.csv", "w", newline="", encoding="utf-8") as file:
    writer = csv.DictWriter(file, fieldnames=individual_fields)
    writer.writeheader()
    writer.writerows(individual_rows)

if len(accepted) != 5:
    print(f"Aggregate deferred: {len(accepted)}/5 repetitions are accepted")
    raise SystemExit(0)


def summary_stats(values):
    values = [float(value) for value in values]
    mean = statistics.fmean(values)
    sample_sd = statistics.stdev(values)
    half_width = 2.776 * sample_sd / math.sqrt(5)
    return {
        "n": 5,
        "mean": mean,
        "sample_sd": sample_sd,
        "min": min(values),
        "max": max(values),
        "ci95_low": mean - half_width,
        "ci95_high": mean + half_width,
    }


metrics = {
    "throughput_tok_s": [r["offline"]["total_throughput"] for r in accepted],
    "benchmark_duration_s": [r["offline"]["total_latency"] for r in accepted],
    "end_to_end_s": [r["end_to_end_s"] for r in accepted],
    "gpu_energy_j": [r["gpu_energy_j"] for r in accepted],
    "average_aggregate_gpu_power_w": [r["average_aggregate_gpu_power_w"] for r in accepted],
    "peak_aggregate_hbm_gib": [r["peak_aggregate_hbm_gib"] for r in accepted],
    "max_per_gpu_hbm_gib": [r["max_per_gpu_hbm_gib"] for r in accepted],
    "mean_gpu_util_pct": [r["mean_gpu_util_pct"] for r in accepted],
    "max_temperature_c": [r["max_temperature_c"] for r in accepted],
    "joules_per_request": [r["joules_per_request"] for r in accepted],
    "tokens_per_joule": [r["tokens_per_joule"] for r in accepted],
    "joules_per_1000_tokens": [r["joules_per_1000_tokens"] for r in accepted],
}
aggregate = {
    "accepted_repetitions": 5,
    "telemetry_scope": accepted[0]["telemetry_scope"],
    "metrics": {name: summary_stats(values) for name, values in metrics.items()},
}
(root / "aggregate_5_runs.json").write_text(json.dumps(aggregate, indent=2) + "\n", encoding="utf-8")

rows = [{"metric": name, **values} for name, values in aggregate["metrics"].items()]
with open(root / "aggregate_5_runs.csv", "w", newline="", encoding="utf-8") as file:
    writer = csv.DictWriter(file, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
PY

if [[ "$OFFLINE_RC" -ne 0 ]]; then
  echo "Offline run failed with exit code $OFFLINE_RC; logs retained in $RUN_DIR" >&2
  exit "$OFFLINE_RC"
fi

echo "Completed repeat $REPEAT. Outputs: $RUN_DIR"
