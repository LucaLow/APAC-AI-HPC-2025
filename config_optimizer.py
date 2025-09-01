#!/usr/bin/env python3
"""
Configuration optimizer for 2025-APAC-HPC-AI competition
Generates optimal SGLang configurations based on hardware and workload requirements
"""

import json
import argparse
from typing import Dict, Any

def calculate_memory_settings(num_gpus: int, gpu_memory_gb: int = 80) -> Dict[str, Any]:
    """Calculate optimal memory settings for given hardware configuration"""
    
    # DeepSeek-R1 model requirements (671B parameters, 37B active)
    model_memory_gb = 74  # Approximate BF16 memory for active parameters
    
    # Total available memory across all GPUs
    total_memory_gb = num_gpus * gpu_memory_gb
    
    # Reserve memory for model weights, overhead, and KV cache
    model_overhead = model_memory_gb * 1.2  # 20% overhead for model loading
    available_for_kv = total_memory_gb - model_overhead
    
    # Calculate optimal settings
    settings = {
        "mem_fraction_static": min(0.85, max(0.7, 1.0 - (model_overhead / total_memory_gb))),
        "chunked_prefill_size": 4096 if available_for_kv > 800 else 2048,
        "max_prefill_tokens": 8192 if available_for_kv > 800 else 4096,
        "max_total_tokens": min(1048576, int(available_for_kv * 1024 * 13)),  # Rough tokens per GB
        "page_size": 16
    }
    
    return settings

def calculate_performance_settings(num_prompts: int, target_time_s: int = 400) -> Dict[str, Any]:
    """Calculate performance-oriented settings based on workload"""
    
    # Estimate optimal batch and concurrency settings
    target_rps = num_prompts / target_time_s  # requests per second
    
    settings = {
        "max_running_requests": min(256, max(64, num_prompts // 16)),
        "cuda_graph_max_bs": 64,
        "cuda_graph_bs": [8, 16, 32, 64],
        "num_continuous_decode_steps": 2 if num_prompts > 1000 else 1,
        "schedule_conservativeness": 0.8 if num_prompts > 1000 else 1.0
    }
    
    return settings

def generate_sglang_config(
    num_gpus: int = 16,
    num_nodes: int = 2,
    num_prompts: int = 2000,
    model_path: str = "$HOME/scratch/models/DeepSeek-R1",
    dataset_path: str = "$HOME/scratch/ShareGPT_V3_unfiltered_cleaned_split.json",
    competition_mode: bool = True
) -> Dict[str, Any]:
    """Generate optimized SGLang configuration for competition"""
    
    # Base configuration
    config = {
        # Model and data
        "model_path": model_path,
        "dataset_path": dataset_path,
        "num_prompts": num_prompts,
        "load_format": "dummy",
        "seed": 2025,
        "dtype": "bfloat16",
        "trust_remote_code": True,
        
        # Parallelism
        "tp": num_gpus,
        "nnodes": num_nodes,
        
        # Data types and precision
        "kv_cache_dtype": "fp8_e5m2",  # Memory efficient KV cache
        
        # Backend optimizations
        "attention_backend": "flashinfer",
        "enable_mixed_chunk": True,
        
        # Advanced features (if competition allows)
        "disable_radix_cache": False,  # Keep RadixAttention enabled
        "enable_p2p_check": True,      # Ensure proper GPU communication
    }
    
    # Add memory settings
    memory_settings = calculate_memory_settings(num_gpus)
    config.update(memory_settings)
    
    # Add performance settings
    performance_settings = calculate_performance_settings(num_prompts)
    config.update(performance_settings)
    
    # Competition-specific optimizations
    if competition_mode:
        config.update({
            "enable_tokenizer_batch_encode": False,  # Avoid with large batches
            "disable_custom_all_reduce": False,      # Use optimized all-reduce
            "triton_attention_reduce_in_fp32": True, # Numerical stability
        })
    
    return config

def generate_pbs_script(config: Dict[str, Any], output_file: str = None) -> str:
    """Generate PBS job script with optimal configuration"""
    
    # Extract key parameters
    tp = config.get("tp", 16)
    nnodes = config.get("nnodes", 2)
    num_prompts = config.get("num_prompts", 2000)
    
    # Build SGLang command arguments
    sglang_args = []
    
    # Required arguments
    sglang_args.extend([
        f"--model-path {config.get('model_path')}",
        f"--dataset-path {config.get('dataset_path')}",
        f"--num-prompts {num_prompts}",
        f"--load-format {config.get('load_format', 'dummy')}",
        f"--seed {config.get('seed', 2025)}",
        f"--dtype {config.get('dtype', 'bfloat16')}",
        f"--tp {tp}",
        f"--nnodes {nnodes}",
        "--trust-remote-code",
        "--dist-init-addr $DIST_INIT_ADDR:5000",
        "--node-rank $OMPI_COMM_WORLD_RANK"
    ])
    
    # Memory optimization arguments
    if "mem_fraction_static" in config:
        sglang_args.append(f"--mem-fraction-static {config['mem_fraction_static']}")
    if "kv_cache_dtype" in config:
        sglang_args.append(f"--kv-cache-dtype {config['kv_cache_dtype']}")
    if "chunked_prefill_size" in config:
        sglang_args.append(f"--chunked-prefill-size {config['chunked_prefill_size']}")
    if "max_prefill_tokens" in config:
        sglang_args.append(f"--max-prefill-tokens {config['max_prefill_tokens']}")
    if "max_total_tokens" in config:
        sglang_args.append(f"--max-total-tokens {config['max_total_tokens']}")
    if "page_size" in config:
        sglang_args.append(f"--page-size {config['page_size']}")
    
    # Performance optimization arguments
    if "cuda_graph_max_bs" in config:
        sglang_args.append(f"--cuda-graph-max-bs {config['cuda_graph_max_bs']}")
    if "cuda_graph_bs" in config:
        bs_list = ",".join(map(str, config['cuda_graph_bs']))
        sglang_args.append(f"--cuda-graph-bs {bs_list}")
    if "attention_backend" in config:
        sglang_args.append(f"--attention-backend {config['attention_backend']}")
    if config.get("enable_mixed_chunk"):
        sglang_args.append("--enable-mixed-chunk")
    if "num_continuous_decode_steps" in config:
        sglang_args.append(f"--num-continuous-decode-steps {config['num_continuous_decode_steps']}")
    if "max_running_requests" in config:
        sglang_args.append(f"--max-running-requests {config['max_running_requests']}")
    if "schedule_conservativeness" in config:
        sglang_args.append(f"--schedule-conservativeness {config['schedule_conservativeness']}")
    
    # Additional optimization flags
    if config.get("enable_p2p_check"):
        sglang_args.append("--enable-p2p-check")
    if config.get("triton_attention_reduce_in_fp32"):
        sglang_args.append("--triton-attention-reduce-in-fp32")
    
    # Join all arguments
    sglang_command = " \\\n    ".join(sglang_args)
    
    pbs_script = f'''#!/bin/bash
#PBS -P 50000097
#PBS -l walltime=420
#PBS -l select=2:ncpus=112:ngpus=8:mpiprocs=8:mem=1880gb
#PBS -j oe
#PBS -M oculus.quest11@gmail.com
#PBS -m abe

module load cuda

# Enhanced NCCL configuration for optimal communication
export NCCL_DEBUG=WARN
export NCCL_IB_HCA=mlx5_0:1,mlx5_1:1,mlx5_3:1,mlx5_4:1,mlx5_5:1,mlx5_6:1,mlx5_9:1,mlx5_10:1,mlx5_11:1
export NCCL_NET=IB
export NCCL_IB_PCI_RELAXED_ORDERING=1
export NCCL_TREE_THRESHOLD=0
export NCCL_ALGO=Tree
export NCCL_NET_GDR_LEVEL=5
export NCCL_NET_GDR_READ=1

# Environment optimization
export HF_HOME=$HOME/scratch/hf_home
export HUGGINGFACE_HUB_CACHE=$HOME/scratch/hf_home/hub
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1
export HF_HUB_DISABLE_TELEMETRY=1

echo "=== Competition Run Configuration ==="
echo "Job ID: $PBS_JOBID"
echo "Nodes: {nnodes}"
echo "GPUs: {tp}"
echo "Prompts: {num_prompts}"
echo "Started: $(date)"
echo "=================================="

time /usr/mpi/gcc/openmpi-4.1.7a1/bin/mpirun \\
-hostfile $PBS_NODEFILE \\
-map-by ppr:1:node:PE=112 -oversubscribe -use-hwthread-cpus \\
-bind-to none --report-bindings -display-map \\
-tag-output -output-filename $HOME/run/sglang.$PBS_JOBID \\
-x PATH \\
-x NCCL_DEBUG -x NCCL_IB_HCA -x NCCL_NET -x NCCL_IB_PCI_RELAXED_ORDERING \\
-x NCCL_TREE_THRESHOLD -x NCCL_ALGO -x NCCL_NET_GDR_LEVEL -x NCCL_NET_GDR_READ \\
-x HF_HOME -x HUGGINGFACE_HUB_CACHE -x TRANSFORMERS_OFFLINE -x HF_HUB_OFFLINE -x HF_HUB_DISABLE_TELEMETRY \\
-x DIST_INIT_ADDR=$(head -n 1 $PBS_NODEFILE) \\
bash -c 'time $HOME/scratch/py312/bin/python3 -m sglang.bench_offline_throughput \\
    {sglang_command}'

echo "=== Run Completed ==="
echo "Finished: $(date)"
echo "Job ID: $PBS_JOBID"
echo "====================="
'''
    
    if output_file:
        with open(output_file, "w") as f:
            f.write(pbs_script)
        print(f"PBS script written to {output_file}")
    
    return pbs_script

def main():
    parser = argparse.ArgumentParser(description="Generate optimized SGLang configuration for 2025-APAC-HPC-AI competition")
    parser.add_argument("--num-gpus", type=int, default=16, help="Number of GPUs (default: 16)")
    parser.add_argument("--num-nodes", type=int, default=2, help="Number of nodes (default: 2)")
    parser.add_argument("--num-prompts", type=int, default=2000, help="Number of prompts (default: 2000)")
    parser.add_argument("--model-path", default="$HOME/scratch/models/DeepSeek-R1", help="Model path")
    parser.add_argument("--dataset-path", default="$HOME/scratch/ShareGPT_V3_unfiltered_cleaned_split.json", help="Dataset path")
    parser.add_argument("--output-config", help="Output JSON config file")
    parser.add_argument("--output-pbs", help="Output PBS script file")
    parser.add_argument("--competition-mode", action="store_true", default=True, help="Enable competition optimizations")
    
    args = parser.parse_args()
    
    # Generate configuration
    config = generate_sglang_config(
        num_gpus=args.num_gpus,
        num_nodes=args.num_nodes,
        num_prompts=args.num_prompts,
        model_path=args.model_path,
        dataset_path=args.dataset_path,
        competition_mode=args.competition_mode
    )
    
    # Output configuration
    if args.output_config:
        with open(args.output_config, "w") as f:
            json.dump(config, f, indent=2)
        print(f"Configuration written to {args.output_config}")
    else:
        print("Generated Configuration:")
        print(json.dumps(config, indent=2))
    
    # Generate PBS script
    if args.output_pbs:
        generate_pbs_script(config, args.output_pbs)
    
    # Print summary
    memory_settings = calculate_memory_settings(args.num_gpus)
    performance_settings = calculate_performance_settings(args.num_prompts)
    
    print(f"\n=== Configuration Summary ===")
    print(f"Hardware: {args.num_gpus} GPUs across {args.num_nodes} nodes")
    print(f"Workload: {args.num_prompts} prompts")
    print(f"Memory fraction: {memory_settings['mem_fraction_static']:.2f}")
    print(f"KV cache: FP8 E5M2")
    print(f"Chunked prefill: {memory_settings['chunked_prefill_size']} tokens")
    print(f"Max running requests: {performance_settings['max_running_requests']}")
    print(f"CUDA graph batch sizes: {performance_settings['cuda_graph_bs']}")

if __name__ == "__main__":
    main()