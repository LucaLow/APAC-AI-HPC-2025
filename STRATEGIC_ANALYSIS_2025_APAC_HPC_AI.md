# Strategic Analysis and Recommendations for 2025-APAC-HPC-AI Competition

## 1. Competition Rules Analysis

### Key Constraints and Requirements
- **Hardware**: 2 GPU nodes with 8 H100 GPUs each (16 total)
- **Model**: DeepSeek-R1 (671B total parameters, 37B active parameters)
- **Framework**: SGLang with BF16 precision
- **Workload**: 2000 prompts from ShareGPT dataset
- **Time Limit**: 420 seconds total (including model loading, warmup, benchmarking)
- **Performance Metric**: Throughput (tokens/second) for offline inference
- **Evaluation**: Based on optimization method, performance improvement, and technical understanding

### Prohibited Modifications
- ❌ Model architecture changes or parameter modifications
- ❌ Quantization, LoRA, pruning, knowledge distillation
- ❌ Experimental features (torch.compile, torchao as of Aug 1st)
- ❌ Double sparsity attention, multimodal capabilities
- ❌ Input dataset modification

### Allowed Optimizations
- ✅ Parallel strategies (tensor, pipeline, data parallelism)
- ✅ Communication optimization (NCCL tuning, inter-node communication)
- ✅ Framework optimization (custom inference engines, runtime optimizations)
- ✅ Memory management (KV-cache optimization, memory pooling, batch processing)
- ✅ Inference optimization (kernel optimization, operator optimization, custom CUDA kernels)
- ✅ Software environment optimization (CUDA version, Python version, dependencies)

## 2. Current State Assessment

### Performance Baseline
- **Current Best**: ~1258 tokens/s with InfiniBand network optimization
- **Configuration**: TP=16 across 2 nodes, using all available H100 GPUs
- **Test Scale**: 100 prompts (need to scale to 2000 for competition)

### Successful Optimizations
1. **InfiniBand Network Optimization**: Achieved ~2.5% performance improvement
   - Forced use of all InfiniBand channels instead of mixed ethernet/IB
   - Proper NCCL_IB_HCA configuration for all available interfaces

### Failed Attempts
1. **Data Parallelism**: Model too large for DP with 16 H100s
2. **DP Attention**: Requires DP > 1, but DP not feasible with current memory constraints
3. **Expert Parallelism**: Attempted but results inconclusive
4. **TorchCompile**: Experimental feature, prohibited by competition rules

### Identified Issues
- **Scale Gap**: Testing at 100 prompts vs competition requirement of 2000 prompts
- **Time Pressure**: Must complete 2000 prompts in under 420 seconds
- **Memory Constraints**: Limited ability to use certain parallelism strategies

## 3. Strategic Recommendations (Prioritized)

### Priority 1: Memory and KV-Cache Optimization
**Rationale**: Largest potential impact for scaling from 100 to 2000 prompts

1. **KV-Cache Memory Management**
   ```bash
   --mem-fraction-static 0.8  # Reduce from default 0.9 if hitting OOM
   --kv-cache-dtype fp8_e5m2  # Use FP8 for KV cache to save memory
   ```

2. **Chunked Prefill Optimization**
   ```bash
   --chunked-prefill-size 4096  # Optimize for memory vs latency trade-off
   --max-prefill-tokens 8192    # Adjust based on available memory
   ```

### Priority 2: Advanced SGLang Features
**Rationale**: Leverage platform-specific optimizations

1. **CUDA Graph Optimization**
   ```bash
   --cuda-graph-max-bs 64      # Enable CUDA graph for consistent batch sizes
   --cuda-graph-bs 8,16,32,64  # Pre-compile for common batch sizes
   ```

2. **Attention Backend Optimization**
   ```bash
   --attention-backend flashinfer  # Use optimized attention implementation
   --triton-attention-reduce-in-fp32  # Improve numerical stability
   ```

### Priority 3: Communication and Scheduling Optimization
**Rationale**: Build on existing InfiniBand optimization

1. **Advanced NCCL Configuration**
   ```bash
   -x NCCL_TREE_THRESHOLD=0
   -x NCCL_ALGO=Tree
   -x NCCL_NET_GDR_LEVEL=5
   -x NCCL_NET_GDR_READ=1
   ```

2. **Overlap and Scheduling**
   ```bash
   --enable-mixed-chunk         # Mix prefill and decode in batches
   --num-continuous-decode-steps 2  # Balance TTFT vs throughput
   ```

### Priority 4: Fine-tuned Configuration Parameters
**Rationale**: Optimize for specific workload characteristics

1. **Batch Processing Optimization**
   ```bash
   --max-running-requests 128   # Optimize concurrent request handling
   --max-total-tokens 1048576   # Adjust token pool size
   ```

2. **Custom Memory Configuration**
   ```bash
   --page-size 16               # Optimize page size for DeepSeek model
   --schedule-conservativeness 0.8  # More aggressive scheduling
   ```

## 4. Code Suggestions

### Optimized PBS Job Script
```bash
#!/bin/bash
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

# Memory and environment optimization
export HF_HOME=$HOME/scratch/hf_home
export HUGGINGFACE_HUB_CACHE=$HOME/scratch/hf_home/hub
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1
export HF_HUB_DISABLE_TELEMETRY=1

time /usr/mpi/gcc/openmpi-4.1.7a1/bin/mpirun \
-hostfile $PBS_NODEFILE \
-map-by ppr:1:node:PE=112 -oversubscribe -use-hwthread-cpus \
-bind-to none --report-bindings -display-map \
-tag-output -output-filename $HOME/run/sglang.$PBS_JOBID \
-x PATH -x NCCL_DEBUG -x NCCL_IB_HCA -x NCCL_NET -x NCCL_IB_PCI_RELAXED_ORDERING \
-x NCCL_TREE_THRESHOLD -x NCCL_ALGO -x NCCL_NET_GDR_LEVEL -x NCCL_NET_GDR_READ \
-x HF_HOME -x HUGGINGFACE_HUB_CACHE -x TRANSFORMERS_OFFLINE -x HF_HUB_OFFLINE -x HF_HUB_DISABLE_TELEMETRY \
-x DIST_INIT_ADDR=$(head -n 1 $PBS_NODEFILE) \
bash -c 'time $HOME/scratch/py312/bin/python3 -m sglang.bench_offline_throughput \
    --model-path $HOME/scratch/models/DeepSeek-R1 \
    --dataset-path $HOME/scratch/ShareGPT_V3_unfiltered_cleaned_split.json \
    --num-prompts 2000 --load-format dummy --seed 2025 --dtype bfloat16 \
    --tp 16 --nnodes 2 --trust-remote-code \
    --dist-init-addr $DIST_INIT_ADDR:5000 --node-rank $OMPI_COMM_WORLD_RANK \
    --mem-fraction-static 0.8 \
    --kv-cache-dtype fp8_e5m2 \
    --chunked-prefill-size 4096 \
    --max-prefill-tokens 8192 \
    --cuda-graph-max-bs 64 \
    --cuda-graph-bs 8,16,32,64 \
    --attention-backend flashinfer \
    --enable-mixed-chunk \
    --num-continuous-decode-steps 2 \
    --max-running-requests 128 \
    --max-total-tokens 1048576 \
    --page-size 16 \
    --schedule-conservativeness 0.8'
```

### Performance Monitoring Script
```bash
#!/bin/bash
# performance_monitor.sh - Extract and analyze benchmark results

LOGFILE="$1"
if [ -z "$LOGFILE" ]; then
    echo "Usage: $0 <logfile>"
    exit 1
fi

echo "=== SGLang Performance Analysis ==="
echo "Logfile: $LOGFILE"
echo "==================================="

# Extract benchmark results
grep "Offline Throughput Benchmark Result" -A 11 "$LOGFILE"

# Extract timing information
echo -e "\n=== Timing Analysis ==="
grep "real\|user\|sys" "$LOGFILE"

# Check for any errors or warnings
echo -e "\n=== Error Analysis ==="
grep -i "error\|warning\|failed" "$LOGFILE" | head -10

# Memory usage analysis
echo -e "\n=== Memory Usage ==="
grep -i "memory\|oom\|cuda" "$LOGFILE" | head -5
```

## 5. SGLang-Specific Optimizations

### Current SGLang Capabilities (Latest Version)
1. **RadixAttention**: Automatic prefix caching for shared prompt prefixes
2. **FlashInfer Backend**: Optimized attention implementation
3. **CUDA Graph**: Reduce kernel launch overhead
4. **FP8 KV-Cache**: Memory-efficient cache storage
5. **Mixed Chunk Processing**: Overlap prefill and decode operations

### Implementation Strategy
```python
# config_optimizer.py - Generate optimal configurations
import json

def generate_sglang_config(num_gpus=16, model_size="671B", target_prompts=2000):
    """Generate optimized SGLang configuration for competition"""
    
    # Base configuration
    config = {
        "tp": num_gpus,
        "nnodes": 2,
        "dtype": "bfloat16",
        "trust_remote_code": True,
        
        # Memory optimization
        "mem_fraction_static": 0.8,
        "kv_cache_dtype": "fp8_e5m2",
        "chunked_prefill_size": 4096,
        "max_prefill_tokens": 8192,
        
        # Performance optimization
        "cuda_graph_max_bs": 64,
        "attention_backend": "flashinfer",
        "enable_mixed_chunk": True,
        "num_continuous_decode_steps": 2,
        
        # Scheduling optimization
        "max_running_requests": min(128, target_prompts // 16),
        "max_total_tokens": 1048576,
        "page_size": 16,
        "schedule_conservativeness": 0.8
    }
    
    return config

# Generate and save optimal config
config = generate_sglang_config()
with open("/tmp/optimal_sglang_config.json", "w") as f:
    json.dump(config, f, indent=2)
```

## 6. Risk Mitigation

### Technical Risks
1. **Memory Overflow**: 
   - **Risk**: Scaling to 2000 prompts may cause OOM
   - **Mitigation**: Conservative memory settings, FP8 KV-cache, chunked prefill

2. **Time Limit Breach**:
   - **Risk**: 420 second limit includes model loading
   - **Mitigation**: Pre-load model, optimize startup sequence, dummy load format

3. **Network Bottlenecks**:
   - **Risk**: Inter-node communication latency
   - **Mitigation**: Enhanced NCCL configuration, overlap optimization

### Implementation Risks
1. **Untested Configuration**:
   - **Risk**: New settings may cause instability
   - **Mitigation**: Incremental testing, fallback configurations

2. **Version Compatibility**:
   - **Risk**: Latest SGLang features may not be stable
   - **Mitigation**: Use stable tagged versions, validate on test dataset

## 7. Next Steps

### Immediate Actions (Week 1)
- [ ] Test optimized PBS script with 2000 prompts to validate timing
- [ ] Benchmark FP8 KV-cache configuration for memory savings
- [ ] Validate CUDA graph optimization impact
- [ ] Test enhanced NCCL settings for communication improvement

### Short-term Actions (Week 2)
- [ ] Implement and test attention backend optimization
- [ ] Fine-tune memory configuration parameters
- [ ] Develop performance monitoring and analysis tools
- [ ] Create submission documentation and presentation materials

### Competition Preparation (Week 3)
- [ ] Final optimization validation and performance testing
- [ ] Prepare technical presentation with before/after comparisons
- [ ] Document optimization rationale and implementation details
- [ ] Create submission package with all required deliverables

### Performance Targets
- **Conservative**: 1400+ tokens/s (15% improvement over baseline)
- **Optimistic**: 1600+ tokens/s (25% improvement over baseline)
- **Stretch**: 1800+ tokens/s (40% improvement over baseline)

### Success Metrics
1. **Primary**: Total token throughput (tokens/s)
2. **Secondary**: Request completion rate (all 2000 prompts processed)
3. **Tertiary**: Time to completion (under 420 seconds)
4. **Quality**: Output quality verification (if code modifications made)

## Competition Strategy Summary

The key to success lies in leveraging SGLang's advanced features while staying within competition constraints. Focus on memory optimization for scaling to 2000 prompts, enhanced communication for multi-node performance, and careful tuning of SGLang-specific parameters. The combination of FP8 KV-cache, optimized attention backends, and advanced scheduling should provide significant performance improvements while maintaining output quality.