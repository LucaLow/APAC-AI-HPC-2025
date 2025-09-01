# 2025-APAC-HPC-AI Competition Implementation Guide

This repository contains optimized configurations and tools for the 2025-APAC-HPC-AI competition, focusing on SGLang-based DeepSeek inference performance optimization.

## Quick Start

### 1. Use the Optimized Competition Script
The fastest way to get started with competition-ready optimization:

```bash
# Submit the optimized job
qsub optimized_competition.pbs

# Monitor the job
qstat -u $USER

# Analyze results when complete
./performance_monitor.sh $HOME/run/stdout.sglang.$JOBID
```

### 2. Generate Custom Configurations
For fine-tuning or different scenarios:

```bash
# Generate configuration for competition (2000 prompts)
python3 config_optimizer.py --output-config competition_config.json --output-pbs competition_run.pbs

# Generate configuration for testing (100 prompts)
python3 config_optimizer.py --num-prompts 100 --output-config test_config.json --output-pbs test_run.pbs

# View generated configuration
cat competition_config.json
```

## Key Optimizations Implemented

### Network Communication
- **InfiniBand Optimization**: Complete IB channel utilization
- **Advanced NCCL Configuration**: Tree algorithm, GDR optimizations
- **P2P GPU Communication**: Enabled for direct GPU-to-GPU transfers

### Memory Management
- **FP8 KV-Cache**: 50% memory reduction for key-value cache
- **Optimized Memory Allocation**: 85% static allocation for stability
- **Chunked Prefill**: 4096 token chunks for memory efficiency

### SGLang Features
- **FlashInfer Backend**: Optimized attention implementation
- **CUDA Graph**: Pre-compiled execution graphs for common batch sizes
- **Mixed Chunk Processing**: Overlap prefill and decode operations
- **RadixAttention**: Automatic prefix caching for shared prompts

### Performance Tuning
- **Batch Optimization**: Dynamic batch sizing with CUDA graphs
- **Scheduling**: Conservative scheduling for throughput optimization
- **Continuous Decode**: Optimized token generation pipeline

## Competition Compliance

### ✅ Allowed Optimizations Used
- Tensor parallelism (TP=16 across 2 nodes)
- Communication optimization (enhanced NCCL configuration)
- Memory management (FP8 KV-cache, chunked prefill)
- Framework optimization (SGLang runtime optimizations)
- Kernel optimization (FlashInfer attention backend)

### ❌ Prohibited Features Avoided
- No model architecture changes
- No quantization of model weights
- No experimental features (torch.compile, torchao)
- No input dataset modifications
- No output quality degradation

## Performance Expectations

### Baseline vs Optimized
- **Baseline**: ~1226-1258 tokens/s (100 prompts)
- **Target**: 1400+ tokens/s (2000 prompts, competition setting)
- **Optimization Areas**: 15-25% improvement expected

### Competition Metrics
- **Primary**: Total token throughput (tokens/s)
- **Constraint**: Complete within 420 seconds
- **Requirement**: Process all 2000 prompts successfully

## File Structure

```
apac/
├── STRATEGIC_ANALYSIS_2025_APAC_HPC_AI.md    # Comprehensive analysis document
├── optimized_competition.pbs                  # Ready-to-submit competition script
├── config_optimizer.py                        # Configuration generator tool
├── performance_monitor.sh                     # Results analysis tool
├── current_best.pbs                          # Previous best configuration
├── DeepseekTests/                             # Test results and experiments
│   ├── (Complete) IB_Force_NetworkOptimisation/
│   ├── DPAttention/
│   ├── DPParallelism/
│   └── Default/
└── 2025-APAC-HPC-AI/                         # Official competition materials
    ├── 2_DeepSeek_AI_Task_Rules.md
    └── 5_1_SGLang_DeepSeek_Application_Notes_ASPIRE2A+.md
```

## Usage Examples

### Running Competition Configuration
```bash
# Submit optimized job
qsub optimized_competition.pbs

# Alternative: Generate custom script
python3 config_optimizer.py --num-prompts 2000 --output-pbs my_competition.pbs
qsub my_competition.pbs
```

### Performance Analysis
```bash
# Analyze results
./performance_monitor.sh $HOME/run/stdout.sglang.123456

# Expected output includes:
# - Token throughput metrics
# - Time compliance check
# - Error analysis
# - Competition readiness score
```

### Testing Different Configurations
```bash
# Test with smaller prompt count
python3 config_optimizer.py --num-prompts 500 --output-pbs test_500.pbs

# Test with different memory settings
python3 config_optimizer.py --num-prompts 2000 --output-config config.json
# Edit config.json as needed
# Use custom config in manual PBS script
```

## Troubleshooting

### Common Issues

1. **Out of Memory (OOM)**
   ```bash
   # Reduce memory allocation
   python3 config_optimizer.py --output-config conservative_config.json
   # Edit mem_fraction_static to 0.75 or lower
   ```

2. **Time Limit Exceeded**
   ```bash
   # Check if model loading is slow
   # Consider pre-loading model or using different load format
   ```

3. **Network Communication Issues**
   ```bash
   # Verify InfiniBand configuration
   ibstat
   # Check NCCL environment variables in job output
   ```

### Performance Debugging
```bash
# Enable detailed logging
export NCCL_DEBUG=INFO

# Check GPU utilization
nvidia-smi

# Monitor memory usage
cat /proc/meminfo
```

## Competition Submission

### Required Deliverables
1. **PBS Scripts**: Use `optimized_competition.pbs` or generated scripts
2. **Output Logs**: Captured automatically in `$HOME/run/`
3. **Performance Analysis**: Use `performance_monitor.sh` for analysis
4. **Configuration Documentation**: Generated configs and this README

### Presentation Preparation
1. **Before/After Comparison**: Use baseline vs optimized results
2. **Technical Understanding**: Reference STRATEGIC_ANALYSIS document
3. **Performance Metrics**: Focus on token throughput improvements
4. **Optimization Rationale**: Explain each optimization choice

## Advanced Configuration

### Custom NCCL Settings
The optimized script includes advanced NCCL configuration. For different network topologies:

```bash
# For different IB configurations, modify:
export NCCL_IB_HCA=<your_ib_devices>
export NCCL_ALGO=Ring  # Alternative to Tree
```

### Memory Optimization
For different model sizes or GPU memory:

```python
# Use config_optimizer.py with custom parameters
python3 config_optimizer.py \
  --num-gpus 16 \
  --num-prompts 2000 \
  --output-config custom.json

# Then edit custom.json for specific tweaks
```

### SGLang Version Management
```bash
# Verify SGLang version
$HOME/scratch/py312/bin/pip show sglang

# Update if needed (ensure competition compliance)
$HOME/scratch/py312/bin/pip install "sglang[all]>=0.5.0rc2"
```

## Next Steps

1. **Test Current Configuration**: Run `optimized_competition.pbs` to establish baseline
2. **Analyze Results**: Use `performance_monitor.sh` for detailed analysis
3. **Fine-tune**: Use `config_optimizer.py` to generate custom configurations
4. **Validate**: Ensure all 2000 prompts complete within 420 seconds
5. **Document**: Prepare presentation materials based on results

## Support

For issues or questions:
1. Check the STRATEGIC_ANALYSIS document for detailed explanations
2. Review SGLang documentation: https://docs.sglang.ai/
3. Consult competition materials in `2025-APAC-HPC-AI/` directory
4. Use performance monitoring tools for debugging

## Competition Timeline

- **Week 1**: Test optimized configuration, validate performance
- **Week 2**: Fine-tune parameters, prepare submission materials
- **Week 3**: Final validation, complete submission package