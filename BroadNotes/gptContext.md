# Concise Context for AI Task in 2025 APAC HPC-AI Competition

## Task Overview
- **Focus**: Optimize SGLang-based DeepSeek inference performance for offline throughput benchmarking. Model: deepseek-ai/DeepSeek-R1 (671B total parameters, 37B active per token). Data type: BF16 for weights and activations. Dataset: ShareGPT_V3_unfiltered_cleaned_split.json (conversational benchmark).
- **Requirements**: Achieve high throughput (tokens/second) in offline inference. Use 2 GPU nodes (8 NVIDIA H100 GPUs per node, total 16 H100 GPUs). Benchmark 2000 prompts; load-format=dummy; random seed=2025. Total execution time (including model loading, warm-up, and benchmarking) must not exceed 420 seconds—results beyond this are invalid.
- **Environment**: NSCC Singapore ASPIRE-2A+ supercomputer (or other GPU clusters for preliminary work, presented as extra info). CUDA 12.9, Python 3.12 venv, OpenMPI 4.1.7, NCCL. Use any tagged SGLang version or master branch (https://github.com/sgl-project/sglang). Source code modifications allowed if justified with validation that output quality is preserved.
  - Assumed Directories/Paths:
    - CUDA at $HOME/cuda-12.9
    - Python venv at $HOME/scratch/py312
    - HF cache at $HOME/scratch/hf-cache
    - Dataset: $HOME/scratch/ShareGPT_V3_unfiltered_cleaned_split.json
- **Metrics**: Throughput (tokens/second); include comparisons with baseline, scaling analysis across GPU configs, and optional profiling data.

## Detailed Requirements, Constraints, and Limitations
From the official GitHub repo (https://github.com/hpcac/2025-APAC-HPC-AI), task descriptions, rules.md, and benchmarks guidelines as of September 2025:

### Requirements
- **Model and Setup**:
  - Use DeepSeek-R1 with MLA and DeepSeekMoE architecture.
  - Benchmark offline throughput with SGLang (sglang.bench_offline_throughput module).
  - Dataset: ShareGPT_V3_unfiltered_cleaned_split.json; process exactly 2000 prompts; fixed seed=2025 for reproducibility.
  - Data type: BF16 (fixed; no alternatives).
  - Distributed: Tensor Parallelism (TP=16) across exactly 2 nodes; nnodes=2; load-format=dummy.
  - Trust-remote-code: Enabled for model loading.
  - Dist-init-addr: Derived from PBS_NODEFILE (e.g., first node IP:5000); node-rank from environment (e.g., OMPI_COMM_WORLD_RANK).

- **Benchmarking**:
  - Measure and report: Tokens/second throughput; include before/after comparisons with baseline.
  - Provide scaling analysis showing performance across different GPU configurations.
  - Execution must complete within 420 seconds total.

- **Optimizations**:
  - Allowed (including but not limited to): Parallel strategies (tensor parallelism, pipeline parallelism, data parallelism configs); communication optimization (NCCL tuning, inter-node comms reduction); framework optimization (custom inference engines, runtime opts); memory management (KV-cache opt, memory pooling, batch processing); inference optimization (kernel opt, operator opt, custom CUDA kernels); software environment (CUDA/Python versions, dependencies).
  - If modifying SGLang source code, provide justification and validation (e.g., output quality verification reports) demonstrating preserved generation quality.
  - NCCL: Tune for multi-node (e.g., NCCL_DEBUG=INFO, NCCL_IB_DISABLE=0, NCCL_IB_HCA=mlx5_0:1, NCCL_SOCKET_IFNAME=ib0, NCCL_LL_THRESHOLD=0).

### Constraints
- **Hardware/Cluster**:
  - Fixed: 2 GPU servers with 8 H100 GPUs each (16 total); results executed on NSCC ASPIRE-2A+.
  - No external hardware or cloud resources beyond provided cluster (other clusters for extra info only).
  - Walltime limit: 420 seconds max (includes all phases); exceeds this = invalid.
  - Networking: Must use cluster interconnect (e.g., InfiniBand); no custom ports conflicting with 5000.

- **Software**:
  - Fixed versions: CUDA 12.9, OpenMPI 4.1.7, Python 3.12 in venv.
  - Launcher: MPI (mpirun) or torchrun; no other frameworks.
  - Model: No core alterations (e.g., no retraining); optimizations only via allowed methods.

- **Performance and Scaling**:
  - Must scale to exactly 2 nodes for primary results; include analysis for other configs.
  - TP limited to 16 (8 GPUs/node x 2); no unapproved parallelism.
  - Memory: Stay under allocation (1880GB per select); handle via allowed memory opts.

### Limitations and Prohibited Modifications
- **All Prohibited Items** (explicitly listed; violations invalidate submissions):
  - Model architecture changes: Altering fundamental structure or parameters.
  - Reducing computation: Quantization, LoRA, pruning, knowledge distillation.
  - Output quality degradation: Any mods significantly impacting generation quality.
  - Input modification: Changing benchmark dataset content or evaluation criteria.
  - Experimental features
  - Additional restrictions due to task design/resources: Double sparsity attention, multimodal capabilities, custom logit processors.

## Possible Optimizations for SGLang
- **Core Command**: python -m sglang.bench_offline_throughput --model-path deepseek-ai/DeepSeek-R1 --dataset-path $HOME/scratch/ShareGPT_V3_unfiltered_cleaned_split.json --num-prompts 2000 --load-format dummy --seed 2025 --dtype bfloat16 --tp 16 --nnodes 2 --trust-remote-code --dist-init-addr $DIST_INIT_ADDR:5000 --node-rank $RANK.
- **Distributed**: TP=16; NCCL tuning as above.
- **Enhancements** (with objective notes on usage; adhere to allowed list):
  - Parallel strategies: e.g., tensor/pipeline/data parallelism configs; may affect scalability and memory.
  - Communication: NCCL tuning; can reduce inter-node overhead.
  - Memory: KV-cache opt, pooling, batch processing; addresses memory constraints.
  - Inference: Kernel/operator opt, custom CUDA kernels; may impact throughput.
  - Framework: Custom engines/runtime opts; requires code mod validation.
- **Launcher**: mpirun (-map-by socket:PE=56 -bind-to core) or torchrun (--nnodes=2 --nproc_per_node=8 --master_addr $MASTER_ADDR --master_port 29500 --rdzv_backend c10d).
- **Monitoring**: Use profiling tools for bottlenecks; include in optional analysis.

## Common Issues and Resolutions
- **NCCL Timeouts/Comms Failures**: Due to network latency; resolution: Tune NCCL vars; verify InfiniBand.
- **OOM Errors**: From large model; resolution: Use allowed memory opts like KV-cache tuning.
- **Init Failures**: Port/address conflicts; resolution: Ensure port 5000 free; stable master IP.
- **Low Throughput**: Bottlenecks; resolution: Profile and apply allowed opts.
- **Time Exceeded**: Over 420s; resolution: Optimize loading/warm-up phases.
- **Reproducibility Issues**: Varying results; resolution: Enforce seed=2025; validate mods.

## Baseline Script
```bash
#!/bin/bash
#PBS -P 50000097
#PBS -l walltime=660
#PBS -l select=2:ncpus=112:ngpus=8:mpiprocs=8:mem=1880gb
#PBS -j oe
#PBS -M oculus.quest11@gmail.com
#PBS -m abe
##PBS -l other=hyperthread

# module load cuda

export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"
export VENV="$HOME/scratch/py312"
export CUDA_HOME="$HOME/cuda-12.9"

export LD_LIBRARY_PATH="$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"
export PATH="$VENV/bin:$PATH"
export PATH="$CUDA_HOME/bin:$PATH"

which nvcc || true
nvcc --version || true
nvidia-smi -L || true
which ninja || true
ninja --version || true

time /usr/mpi/gcc/openmpi-4.1.7a1/bin/mpirun \
-hostfile ${PBS_NODEFILE} \
-map-by ppr:1:node:PE=112 -oversubscribe -use-hwthread-cpus \
-bind-to none --report-bindings -display-map \
-tag-output -output-filename ${HOME}/run/sglang.${PBS_JOBID} \
-x PATH=$HOME/scratch/py312/bin:$PATH \
-x NCCL_DEBUG=INFO \
-x DIST_INIT_ADDR=$(head -n 1 $PBS_NODEFILE) \
bash -c 'time ${HOME}/scratch/py312/bin/python3 \
-m sglang.bench_offline_throughput \
--model-path deepseek-ai/DeepSeek-R1 \
--dataset-path ${HOME}/scratch/ShareGPT_V3_unfiltered_cleaned_split.json \
--num-prompts 2000 --load-format dummy --seed 2025 --dtype bfloat16 \
--tp 16 --nnodes 2 --trust-remote-code \
--dist-init-addr ${DIST_INIT_ADDR}:5000 --node-rank ${OMPI_COMM_WORLD_RANK}' \
2>&1 | tee ${HOME}/run/stdout.sglang.${PBS_JOBID}
```

## Resources
- Repo: https://github.com/hpcac/2025-APAC-HPC-AI
- SGLang: https://github.com/sgl-project/sglang
- Model: https://github.com/deepseek-ai/DeepSeek-V3 (R1 details)
- Slack: https://join.slack.com/t/2025apachpcai-lcl5303/shared_invite/zt-34376r42p-itO0z_DwW~PJHpTbdjldxg