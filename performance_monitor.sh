#!/bin/bash
# performance_monitor.sh - Extract and analyze benchmark results for 2025-APAC-HPC-AI competition

LOGFILE="$1"
if [ -z "$LOGFILE" ]; then
    echo "Usage: $0 <logfile>"
    echo "Example: $0 \$HOME/run/stdout.sglang.123456"
    exit 1
fi

if [ ! -f "$LOGFILE" ]; then
    echo "Error: Logfile '$LOGFILE' not found"
    exit 1
fi

echo "============================================="
echo "2025-APAC-HPC-AI SGLang Performance Analysis"
echo "============================================="
echo "Logfile: $LOGFILE"
echo "Analysis Time: $(date)"
echo "============================================="

# Extract benchmark results
echo -e "\n=== BENCHMARK RESULTS ==="
if grep -q "Offline Throughput Benchmark Result" "$LOGFILE"; then
    grep "Offline Throughput Benchmark Result" -A 11 "$LOGFILE"
    
    # Extract key metrics
    THROUGHPUT=$(grep "Total token throughput" "$LOGFILE" | tail -1 | awk '{print $5}')
    DURATION=$(grep "Benchmark duration" "$LOGFILE" | tail -1 | awk '{print $4}')
    REQUESTS=$(grep "Successful requests" "$LOGFILE" | tail -1 | awk '{print $3}')
    
    echo -e "\n=== KEY METRICS SUMMARY ==="
    echo "Total Token Throughput: $THROUGHPUT tokens/s"
    echo "Benchmark Duration: $DURATION seconds"
    echo "Successful Requests: $REQUESTS"
    
    # Competition compliance check
    if [ -n "$DURATION" ]; then
        if (( $(echo "$DURATION > 420" | bc -l) )); then
            echo "⚠️  WARNING: Exceeds 420 second time limit! (INVALID for competition)"
        else
            echo "✅ Within 420 second time limit"
        fi
    fi
    
    if [ "$REQUESTS" == "2000" ]; then
        echo "✅ All 2000 prompts completed"
    else
        echo "⚠️  WARNING: Only $REQUESTS out of 2000 prompts completed"
    fi
else
    echo "❌ No benchmark results found in logfile"
fi

# Extract timing information
echo -e "\n=== EXECUTION TIMING ==="
if grep -q "real\|user\|sys" "$LOGFILE"; then
    echo "Total execution time:"
    grep "real\|user\|sys" "$LOGFILE" | tail -3
else
    echo "No timing information found"
fi

# Check for optimization confirmations
echo -e "\n=== OPTIMIZATION STATUS ==="
echo "InfiniBand Configuration:"
grep -i "NCCL_IB_HCA\|NCCL_NET" "$LOGFILE" | head -2

echo -e "\nMemory Optimizations:"
if grep -q "fp8_e5m2" "$LOGFILE"; then
    echo "✅ FP8 KV-cache enabled"
else
    echo "❌ FP8 KV-cache not detected"
fi

if grep -q "chunked-prefill" "$LOGFILE"; then
    echo "✅ Chunked prefill enabled"
else
    echo "❌ Chunked prefill not detected"
fi

# Check for any errors or warnings
echo -e "\n=== ERROR ANALYSIS ==="
ERROR_COUNT=$(grep -i "error\|failed\|exception" "$LOGFILE" | wc -l)
WARNING_COUNT=$(grep -i "warning" "$LOGFILE" | grep -v "NCCL_DEBUG=WARN" | wc -l)

echo "Errors found: $ERROR_COUNT"
echo "Warnings found: $WARNING_COUNT"

if [ $ERROR_COUNT -gt 0 ]; then
    echo -e "\nFirst 5 errors:"
    grep -i "error\|failed\|exception" "$LOGFILE" | head -5
fi

if [ $WARNING_COUNT -gt 0 ]; then
    echo -e "\nFirst 5 warnings:"
    grep -i "warning" "$LOGFILE" | grep -v "NCCL_DEBUG=WARN" | head -5
fi

# Memory usage analysis
echo -e "\n=== MEMORY ANALYSIS ==="
if grep -q -i "oom\|out of memory" "$LOGFILE"; then
    echo "❌ Out of Memory detected!"
    grep -i "oom\|out of memory" "$LOGFILE" | head -3
else
    echo "✅ No OOM errors detected"
fi

# GPU utilization check
echo -e "\n=== GPU UTILIZATION ==="
if grep -q "cuda" "$LOGFILE"; then
    echo "CUDA activity detected"
    grep -i "cuda\|gpu" "$LOGFILE" | head -3
else
    echo "No specific GPU utilization information found"
fi

# Competition readiness assessment
echo -e "\n=== COMPETITION READINESS ASSESSMENT ==="
SCORE=0

# Check throughput
if [ -n "$THROUGHPUT" ]; then
    if (( $(echo "$THROUGHPUT > 1400" | bc -l) )); then
        echo "✅ Throughput > 1400 tokens/s (competitive)"
        SCORE=$((SCORE + 2))
    elif (( $(echo "$THROUGHPUT > 1200" | bc -l) )); then
        echo "⚠️  Throughput > 1200 tokens/s (baseline)"
        SCORE=$((SCORE + 1))
    else
        echo "❌ Throughput < 1200 tokens/s (needs improvement)"
    fi
fi

# Check timing
if [ -n "$DURATION" ] && (( $(echo "$DURATION <= 420" | bc -l) )); then
    echo "✅ Meets time constraint"
    SCORE=$((SCORE + 1))
fi

# Check completion
if [ "$REQUESTS" == "2000" ]; then
    echo "✅ Processes all required prompts"
    SCORE=$((SCORE + 1))
fi

echo -e "\nReadiness Score: $SCORE/4"
if [ $SCORE -ge 3 ]; then
    echo "🏆 READY FOR COMPETITION"
elif [ $SCORE -ge 2 ]; then
    echo "⚠️  NEEDS MINOR IMPROVEMENTS"
else
    echo "❌ REQUIRES SIGNIFICANT OPTIMIZATION"
fi

echo -e "\n============================================="
echo "Analysis Complete"
echo "============================================="