#!/bin/bash
# validate_optimizations.sh - Quick validation script for competition optimizations

set -e

echo "=============================================="
echo "2025-APAC-HPC-AI Competition Validation"
echo "=============================================="

# Check if we're on the correct system
if [ ! -d "/home/users/industry/ai-hpc" ]; then
    echo "⚠️  Warning: This appears not to be the ASPIRE-2A+ system"
    echo "Some validations may not apply"
fi

echo -e "\n=== File Validation ==="

# Check required files exist
REQUIRED_FILES=(
    "optimized_competition.pbs"
    "config_optimizer.py"
    "performance_monitor.sh"
    "STRATEGIC_ANALYSIS_2025_APAC_HPC_AI.md"
    "README.md"
)

for file in "${REQUIRED_FILES[@]}"; do
    if [ -f "$file" ]; then
        echo "✅ $file exists"
    else
        echo "❌ $file missing"
        exit 1
    fi
done

echo -e "\n=== Script Permissions ==="
if [ -x "config_optimizer.py" ]; then
    echo "✅ config_optimizer.py is executable"
else
    echo "❌ config_optimizer.py not executable"
    chmod +x config_optimizer.py
    echo "🔧 Fixed permissions for config_optimizer.py"
fi

if [ -x "performance_monitor.sh" ]; then
    echo "✅ performance_monitor.sh is executable"
else
    echo "❌ performance_monitor.sh not executable"
    chmod +x performance_monitor.sh
    echo "🔧 Fixed permissions for performance_monitor.sh"
fi

echo -e "\n=== Configuration Generator Test ==="
if python3 config_optimizer.py --help > /dev/null 2>&1; then
    echo "✅ config_optimizer.py runs successfully"
else
    echo "❌ config_optimizer.py has issues"
    exit 1
fi

# Test configuration generation
echo "🔧 Testing configuration generation..."
if python3 config_optimizer.py --num-prompts 100 --output-config /tmp/test_config.json --output-pbs /tmp/test.pbs > /dev/null 2>&1; then
    echo "✅ Configuration generation works"
    rm -f /tmp/test_config.json /tmp/test.pbs
else
    echo "❌ Configuration generation failed"
    exit 1
fi

echo -e "\n=== PBS Script Validation ==="

# Check PBS script syntax
if [ -f "optimized_competition.pbs" ]; then
    # Basic syntax checks
    if grep -q "#PBS" optimized_competition.pbs; then
        echo "✅ PBS directives found"
    else
        echo "❌ No PBS directives found"
        exit 1
    fi
    
    if grep -q "python3 -m sglang.bench_offline_throughput" optimized_competition.pbs; then
        echo "✅ SGLang benchmark command found"
    else
        echo "❌ SGLang benchmark command not found"
        exit 1
    fi
    
    if grep -q "num-prompts 2000" optimized_competition.pbs; then
        echo "✅ Competition prompt count (2000) configured"
    else
        echo "⚠️  Warning: Prompt count may not be set to 2000"
    fi
    
    if grep -q "tp 16" optimized_competition.pbs; then
        echo "✅ Tensor parallelism (TP=16) configured"
    else
        echo "❌ Tensor parallelism not properly configured"
        exit 1
    fi
    
    if grep -q "nnodes 2" optimized_competition.pbs; then
        echo "✅ Multi-node configuration (2 nodes) found"
    else
        echo "❌ Multi-node configuration not found"
        exit 1
    fi
fi

echo -e "\n=== Optimization Validation ==="

# Check for key optimizations in PBS script
OPTIMIZATIONS=(
    "NCCL_IB_HCA"
    "fp8_e5m2"
    "chunked-prefill-size"
    "cuda-graph-max-bs"
    "attention-backend flashinfer"
    "enable-mixed-chunk"
)

for opt in "${OPTIMIZATIONS[@]}"; do
    if grep -q "$opt" optimized_competition.pbs; then
        echo "✅ $opt optimization enabled"
    else
        echo "⚠️  $opt optimization not found"
    fi
done

echo -e "\n=== Performance Monitor Test ==="
# Create a fake log file for testing
cat > /tmp/test_log.txt << 'EOF'
====== Offline Throughput Benchmark Result =======
Backend:                                 engine    
Successful requests:                     2000      
Benchmark duration (s):                  350.25    
Total input tokens:                      626729    
Total generated tokens:                  388685    
Last generation throughput (tok/s):      33.36     
Request throughput (req/s):              5.71      
Input token throughput (tok/s):          1789.23   
Output token throughput (tok/s):         1109.45   
Total token throughput (tok/s):          2898.68   
==================================================
EOF

if ./performance_monitor.sh /tmp/test_log.txt > /dev/null 2>&1; then
    echo "✅ Performance monitor script works"
else
    echo "❌ Performance monitor script has issues"
    exit 1
fi

# Test with actual output
echo "🔧 Testing performance analysis..."
MONITOR_OUTPUT=$(./performance_monitor.sh /tmp/test_log.txt 2>/dev/null)
if echo "$MONITOR_OUTPUT" | grep -q "READY FOR COMPETITION"; then
    echo "✅ Performance monitor correctly identifies good results"
elif echo "$MONITOR_OUTPUT" | grep -q "tokens/s"; then
    echo "✅ Performance monitor extracts metrics correctly"
else
    echo "⚠️  Performance monitor output unclear"
fi

rm -f /tmp/test_log.txt

echo -e "\n=== Environment Check ==="

# Check for Python 3
if command -v python3 > /dev/null 2>&1; then
    PYTHON_VERSION=$(python3 --version 2>&1 | awk '{print $2}')
    echo "✅ Python 3 available: $PYTHON_VERSION"
else
    echo "❌ Python 3 not found"
    exit 1
fi

# Check for required Python modules
PYTHON_MODULES=("json" "argparse")
for module in "${PYTHON_MODULES[@]}"; do
    if python3 -c "import $module" 2>/dev/null; then
        echo "✅ Python module '$module' available"
    else
        echo "❌ Python module '$module' not available"
        exit 1
    fi
done

echo -e "\n=== Competition Compliance Check ==="

# Check that prohibited features are not used
PROHIBITED=("torch.compile" "torchao" "quantization" "lora")
COMPLIANCE_ISSUES=0

for feature in "${PROHIBITED[@]}"; do
    if grep -q "$feature" optimized_competition.pbs; then
        echo "❌ Prohibited feature '$feature' found in PBS script"
        COMPLIANCE_ISSUES=$((COMPLIANCE_ISSUES + 1))
    else
        echo "✅ No prohibited feature '$feature' found"
    fi
done

if [ $COMPLIANCE_ISSUES -eq 0 ]; then
    echo "✅ All competition compliance checks passed"
else
    echo "❌ $COMPLIANCE_ISSUES compliance issues found"
    exit 1
fi

echo -e "\n=== Final Validation Summary ==="
echo "✅ All validation checks passed!"
echo ""
echo "Next steps:"
echo "1. Run: qsub optimized_competition.pbs"
echo "2. Monitor: qstat -u \$USER"
echo "3. Analyze: ./performance_monitor.sh \$HOME/run/stdout.sglang.JOBID"
echo ""
echo "For custom configurations:"
echo "python3 config_optimizer.py --help"
echo ""
echo "Competition ready! 🏆"