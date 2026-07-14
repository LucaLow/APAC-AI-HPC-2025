# Ground truth — every number verified against a log file in the repo

## Benchmark (fixed by competition)
- `sglang.bench_offline_throughput`, DeepSeek-R1 (671B MoE, 37B active), --dtype bfloat16
- ShareGPT_V3_unfiltered_cleaned_split.json, 2000 prompts, --load-format dummy, --seed 2025
- 420 s wall-clock limit incl. model load + warm-up
- Metric: total token throughput (tok/s)

## Systems
- **System A — NSCC ASPIRE-2A+**: 2 nodes x 8x NVIDIA H100 80GB HBM3 (PBS Pro, 112 CPU cores + 1,880 GB RAM/node, OpenMPI 4.1.7a1). SGLang 0.5.2, PyTorch 2.8.0+cu128, Python 3.12, user-space CUDA 12.9. (DeepseekTests/, PBS logs)
- **System B — Firmus (H200)**: Slurm `hpcai` partition, nodes = 8x NVIDIA H200 (141 GB), multi-rail InfiniBand (8x mlx5 HCAs/node; NCCL "Using network IB"). Singularity image `sglang_v0.5.3rc1-cu126.sif`. (firmus/, AITaskFinalSubmission/)

## Final submission numbers (AITaskFinalSubmission/, System B)
| Run | Config | tok/s | Log |
|---|---|---|---|
| Official baseline | venv, 2N, TP=16 | 9,610.57 | default/default.out |
| Best single-node | Singularity, 1N, TP=4 DP=4 PP=2, dp-attention, torch.compile, NCCL_NET_GDR_LEVEL=5 | **17,417.40** | one_node/one_node.out (58.30 s, 626,729 in + 388,685 out tokens, 34.31 req/s) |
| Best two-node | same hybrid config, 2N (16 GPUs) | 17,032.39 | two_node/two-node.out |

Headline: **+81.2%** vs official baseline; single node beats two nodes in absolute terms with half the GPUs → **2.05x per-GPU efficiency** (2,177.2 vs 1,064.5 tok/s/GPU).

## Optimization ladder (System B, all log-verified)
| Step | Config | tok/s |
|---|---|---|
| Baseline venv 2N TP16 (repeats) | 9,610.57 / 9,890.30 / 10,053.70 | default/, firmus/default/ |
| venv 1N TP8 | 11,558.91 | run 2026-07-11 via firmus/venv_1node.sh — log NOT yet in repo. NOT pure defaults: needed --mem-fraction-static 0.87 + PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True (SGLang 0.5.2 auto-sizing OOMs at 1N TP8 with ~4 GB headroom vs 0.5.3rc1 container's ~12.6 GB; first attempt job 10621 OOMed mid-decode). Footnote if used in paper. |
| Container 2N TP16 | 11,253.39 | singularity-docker/2node.out |
| Container 1N TP8 | 12,449.13 | singularity-docker/1node.out |
| Container 2N TP8 PP2 | 14,632.44 | singularity-docker/pp2.out (NOT in tech report) |
| 1N TP4 DP4 PP2 + dp-attention (repeats) | 17,308.39 / 17,218.03 | xlsx table |
| + NCCL_NET_GDR_LEVEL sweep | 2→17,255.30, 3→17,351.82, 5→17,417.40 | xlsx + one_node.out |
| Router (2x independent 1N servers + sglang-router, online) | 8,438.05 | firmus/router/slurm-758.out |

## mem-fraction-static x max batch (System B, 2N TP16 container — NOT single node)
batch 256: 0.90=9,695.82  0.91=11,057.12  0.92=10,957.14  0.93=11,130.89
batch 512: 0.90=11,460.25 0.91=12,366.44  0.92=12,465.98  0.93=12,311.90
batch 724: 0.90=11,822.57 0.91=12,989.38  0.92=13,059.21  0.93=OOM
0.94 = OOM at all batch sizes. Halving prompts 2000→1000 dropped ~17k→~10k (report claim; no log found).

## NCCL manual sweep (System B, 2N TP16, 40 configs; net_sweep_496/501/506)
- All manual configs 2,019.30–9,561.03 < default 10,053.70 (the "null result")
- Protocol max: LL 6,474.61 | Simple 8,251.19 | LL128 9,561.03
- GDR level (LL/64/4): 0 → ~2,035–2,066; 2 → ~4,779–4,866 (≈2.4x)
- GDR detail (LL128/256/16): never=9,452.12, same-pci=9,460.21, pci-connected=9,561.23, same-root-complex=9,612.12, same-numa=9,712.24, always=9,761.03

## System A (H100) results
- CUDA 12.9 2N TP16: 7,863.04 (sglang5/cuda129.o); second run 7,962.78 (cuda129_2.md)
- CUDA 12.6 comparison value 5,839 tok/s: **from tech report only — no log in repo** (flag to Luca)
- TP16 + dp-attention dp2: 7,169.09 | EP experiments: 5,571.94 / 5,734.18 | later-default: 6,275.82

## Corrections vs tech report
1. Report says "two-node H100" for the main campaign — main campaign was on **H200** (System B); H100 was System A only.
2. Report's single-node TP8 figures (12,520.75 §2.2 / 12,414.55 table) are internally inconsistent; log = **12,449.13**.
3. Report's final "17,255.40 avg" — logs show best = 17,417.40 (GDR=5); 17,255.30 was the GDR=2 run.
4. Report omits 2N TP8 PP2 = 14,632.44, the strongest evidence that PP is the right cross-node axis.
5. Report's router "~10,000" — log shows 8,438.05.
6. Report's baseline for "+75%": we use official submitted pair → +81.2% (9,610.57 → 17,417.40); conservative vs best default run (10,053.70) → +73.2%.
7. Final config includes --enable-torch-compile (in one_node_best.sh), despite report §8.2 saying torch.compile gave no benefit.
