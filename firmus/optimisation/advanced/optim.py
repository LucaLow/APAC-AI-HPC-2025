# hpo_sglang.py
import os, re, subprocess, optuna, shlex

# Parse "gen throughput (token/s): 72.60" from your logs
THR_RE = re.compile(r"gen throughput \(token/s\):\s*([\d.]+)")

def run_bench(params):
    nnodes = os.environ.get("SLURM_JOB_NUM_NODES", "2")
    dist   = os.environ.get("DIST_INIT_ADDR", "g3") + ":5000"

    # (Tune anything you like; these are examples)
    dtype        = params["dtype"]
    tp           = params["tp"]
    num_prompts  = params["num_prompts"]
    nccl_algo    = params["nccl_algo"]
    nthreads     = params["nccl_nthreads"]
    cdmx         = params["cuda_dev_max_conn"]

    # One trial = one srun, with graceful teardown flags to avoid your peer-close noise
    bash = f'''
      set -uo pipefail
      export NCCL_DEBUG=WARN
      export NCCL_NET=IB
      export NCCL_ASYNC_ERROR_HANDLING=1
      export TORCH_DISTRIBUTED_DEFAULT_TIMEOUT=180
      export NCCL_ALGO={nccl_algo}
      export NCCL_NTHREADS={nthreads}
      export CUDA_DEVICE_MAX_CONNECTIONS={cdmx}

      time python3 -m sglang.bench_offline_throughput \
        --model-path deepseek-ai/DeepSeek-R1 \
        --dataset-path "$HOME/ShareGPT_V3_unfiltered_cleaned_split.json" \
        --num-prompts {num_prompts} --load-format dummy --seed 2025 \
        --dtype {dtype} --tp {tp} --nnodes {nnodes} --trust-remote-code \
        --dist-init-addr {dist} --node-rank $SLURM_NODEID || true
      sleep 2
    '''

    cmd = [
      "srun","--label","--cpu-bind=none",
      f"--nodes={nnodes}","--ntasks-per-node=1",
      "--mpi=none","--kill-on-bad-exit=0","--no-kill","--wait=60",
      "bash","-lc", bash
    ]
    out = subprocess.run(cmd, text=True, capture_output=True).stdout
    m = THR_RE.search(out)
    if not m:
        raise RuntimeError("Could not parse throughput from output.")
    return float(m.group(1)), out

def objective(trial: optuna.Trial):
    params = {
      "dtype": trial.suggest_categorical("dtype", ["bfloat16","float16"]),
      "tp": trial.suggest_categorical("tp", [8,16]),
      "num_prompts": trial.suggest_int("num_prompts", 250, 4000, log=True),
      "nccl_algo": trial.suggest_categorical("nccl_algo", ["Tree","Ring"]),
      "nccl_nthreads": trial.suggest_int("nccl_nthreads", 64, 256, step=32),
      "cuda_dev_max_conn": trial.suggest_categorical("cuda_dev_max_conn", [1,2,8]),
    }
    thr, _ = run_bench(params)
    return thr  # maximize throughput

if __name__ == "__main__":
    storage = "sqlite:///" + os.path.expanduser("~/optuna_sglang.db")  # resumable
    study = optuna.create_study(
        study_name="sglang-hpo",
        direction="maximize",
        storage=storage, load_if_exists=True,
        sampler=optuna.samplers.TPESampler(seed=2025),
    )
    study.optimize(objective, n_trials=20, n_jobs=1)
    print("BEST:", study.best_value, study.best_trial.params)