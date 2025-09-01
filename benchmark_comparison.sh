#!/bin/bash
# benchmark_comparison.sh - Compare baseline vs optimized performance

set -e

BASELINE_LOG="$1"
OPTIMIZED_LOG="$2"

if [ -z "$BASELINE_LOG" ] || [ -z "$OPTIMIZED_LOG" ]; then
    echo "Usage: $0 <baseline_log> <optimized_log>"
    echo "Example: $0 baseline_output.log optimized_output.log"
    exit 1
fi

echo "================================================"
echo "2025-APAC-HPC-AI Performance Comparison Analysis"
echo "================================================"
echo "Baseline: $BASELINE_LOG"
echo "Optimized: $OPTIMIZED_LOG"
echo "Analysis Time: $(date)"
echo "================================================"

# Function to extract metrics
extract_metric() {
    local logfile="$1"
    local metric="$2"
    if [ -f "$logfile" ]; then
        grep "$metric" "$logfile" | tail -1 | awk '{print $5}' | tr -d ':'
    else
        echo "N/A"
    fi
}

# Extract metrics from both logs
echo -e "\n=== BASELINE PERFORMANCE ==="
if [ -f "$BASELINE_LOG" ]; then
    grep "Offline Throughput Benchmark Result" -A 11 "$BASELINE_LOG" | tail -11
    
    BASE_THROUGHPUT=$(extract_metric "$BASELINE_LOG" "Total token throughput")
    BASE_DURATION=$(extract_metric "$BASELINE_LOG" "Benchmark duration")
    BASE_REQUESTS=$(extract_metric "$BASELINE_LOG" "Successful requests")
    BASE_INPUT_THROUGHPUT=$(extract_metric "$BASELINE_LOG" "Input token throughput")
    BASE_OUTPUT_THROUGHPUT=$(extract_metric "$BASELINE_LOG" "Output token throughput")
else
    echo "❌ Baseline log file not found"
    exit 1
fi

echo -e "\n=== OPTIMIZED PERFORMANCE ==="
if [ -f "$OPTIMIZED_LOG" ]; then
    grep "Offline Throughput Benchmark Result" -A 11 "$OPTIMIZED_LOG" | tail -11
    
    OPT_THROUGHPUT=$(extract_metric "$OPTIMIZED_LOG" "Total token throughput")
    OPT_DURATION=$(extract_metric "$OPTIMIZED_LOG" "Benchmark duration")
    OPT_REQUESTS=$(extract_metric "$OPTIMIZED_LOG" "Successful requests")
    OPT_INPUT_THROUGHPUT=$(extract_metric "$OPTIMIZED_LOG" "Input token throughput")
    OPT_OUTPUT_THROUGHPUT=$(extract_metric "$OPTIMIZED_LOG" "Output token throughput")
else
    echo "❌ Optimized log file not found"
    exit 1
fi

# Calculate improvements
echo -e "\n=== PERFORMANCE COMPARISON ==="

if [ "$BASE_THROUGHPUT" != "N/A" ] && [ "$OPT_THROUGHPUT" != "N/A" ]; then
    THROUGHPUT_IMPROVEMENT=$(python3 -c "print(f'{(($OPT_THROUGHPUT - $BASE_THROUGHPUT) / $BASE_THROUGHPUT * 100):.2f}')")
    echo "Total Token Throughput:"
    echo "  Baseline: $BASE_THROUGHPUT tokens/s"
    echo "  Optimized: $OPT_THROUGHPUT tokens/s"
    echo "  Improvement: $THROUGHPUT_IMPROVEMENT%"
    
    if (( $(echo "$THROUGHPUT_IMPROVEMENT > 10" | bc -l) )); then
        echo "  Status: 🚀 Excellent improvement!"
    elif (( $(echo "$THROUGHPUT_IMPROVEMENT > 5" | bc -l) )); then
        echo "  Status: ✅ Good improvement"
    elif (( $(echo "$THROUGHPUT_IMPROVEMENT > 0" | bc -l) )); then
        echo "  Status: ⚠️  Marginal improvement"
    else
        echo "  Status: ❌ Performance regression"
    fi
else
    echo "❌ Cannot compare throughput - missing data"
fi

if [ "$BASE_DURATION" != "N/A" ] && [ "$OPT_DURATION" != "N/A" ]; then
    DURATION_IMPROVEMENT=$(python3 -c "print(f'{(($BASE_DURATION - $OPT_DURATION) / $BASE_DURATION * 100):.2f}')")
    echo -e "\nBenchmark Duration:"
    echo "  Baseline: $BASE_DURATION seconds"
    echo "  Optimized: $OPT_DURATION seconds"
    echo "  Improvement: $DURATION_IMPROVEMENT% faster"
fi

if [ "$BASE_INPUT_THROUGHPUT" != "N/A" ] && [ "$OPT_INPUT_THROUGHPUT" != "N/A" ]; then
    INPUT_IMPROVEMENT=$(python3 -c "print(f'{(($OPT_INPUT_THROUGHPUT - $BASE_INPUT_THROUGHPUT) / $BASE_INPUT_THROUGHPUT * 100):.2f}')")
    echo -e "\nInput Token Throughput:"
    echo "  Baseline: $BASE_INPUT_THROUGHPUT tokens/s"
    echo "  Optimized: $OPT_INPUT_THROUGHPUT tokens/s"
    echo "  Improvement: $INPUT_IMPROVEMENT%"
fi

if [ "$BASE_OUTPUT_THROUGHPUT" != "N/A" ] && [ "$OPT_OUTPUT_THROUGHPUT" != "N/A" ]; then
    OUTPUT_IMPROVEMENT=$(python3 -c "print(f'{(($OPT_OUTPUT_THROUGHPUT - $BASE_OUTPUT_THROUGHPUT) / $BASE_OUTPUT_THROUGHPUT * 100):.2f}')")
    echo -e "\nOutput Token Throughput:"
    echo "  Baseline: $BASE_OUTPUT_THROUGHPUT tokens/s"
    echo "  Optimized: $OPT_OUTPUT_THROUGHPUT tokens/s"
    echo "  Improvement: $OUTPUT_IMPROVEMENT%"
fi

# Competition readiness assessment
echo -e "\n=== COMPETITION ASSESSMENT ==="

# Check competition constraints
COMP_READY=true

if [ "$OPT_REQUESTS" != "2000" ] && [ "$OPT_REQUESTS" != "N/A" ]; then
    echo "⚠️  Warning: Optimized run processed $OPT_REQUESTS requests (expected 2000)"
    COMP_READY=false
fi

if [ "$OPT_DURATION" != "N/A" ]; then
    if (( $(echo "$OPT_DURATION > 420" | bc -l) )); then
        echo "❌ Optimized run exceeds 420 second time limit ($OPT_DURATION seconds)"
        COMP_READY=false
    else
        echo "✅ Optimized run meets time constraint ($OPT_DURATION seconds)"
    fi
fi

if [ "$OPT_THROUGHPUT" != "N/A" ]; then
    if (( $(echo "$OPT_THROUGHPUT > 1500" | bc -l) )); then
        echo "🏆 Excellent competitive throughput ($OPT_THROUGHPUT tokens/s)"
    elif (( $(echo "$OPT_THROUGHPUT > 1300" | bc -l) )); then
        echo "✅ Good competitive throughput ($OPT_THROUGHPUT tokens/s)"
    elif (( $(echo "$OPT_THROUGHPUT > 1000" | bc -l) )); then
        echo "⚠️  Moderate throughput ($OPT_THROUGHPUT tokens/s) - room for improvement"
    else
        echo "❌ Low throughput ($OPT_THROUGHPUT tokens/s) - needs optimization"
        COMP_READY=false
    fi
fi

# Overall assessment
echo -e "\n=== OPTIMIZATION SUMMARY ==="
if [ "$COMP_READY" = true ] && [ "$THROUGHPUT_IMPROVEMENT" != "" ]; then
    if (( $(echo "$THROUGHPUT_IMPROVEMENT > 15" | bc -l) )); then
        echo "🌟 OUTSTANDING: Ready for competition with excellent improvements!"
    elif (( $(echo "$THROUGHPUT_IMPROVEMENT > 5" | bc -l) )); then
        echo "🏆 EXCELLENT: Ready for competition with good improvements!"
    else
        echo "✅ READY: Meets competition requirements"
    fi
else
    echo "⚠️  NEEDS WORK: Address issues before competition submission"
fi

# Recommendations
echo -e "\n=== RECOMMENDATIONS ==="
if [ "$THROUGHPUT_IMPROVEMENT" != "" ]; then
    if (( $(echo "$THROUGHPUT_IMPROVEMENT < 5" | bc -l) )); then
        echo "• Consider additional memory optimizations (FP8 KV-cache, larger chunked prefill)"
        echo "• Experiment with different CUDA graph batch sizes"
        echo "• Try alternative attention backends"
    fi
fi

if [ "$OPT_DURATION" != "N/A" ] && (( $(echo "$OPT_DURATION > 350" | bc -l) )); then
    echo "• Monitor startup time - consider dummy load format optimization"
    echo "• Check for memory allocation delays"
fi

echo "• Document optimization rationale for competition presentation"
echo "• Prepare before/after comparison charts"
echo "• Validate output quality if source code was modified"

echo -e "\n================================================"
echo "Comparison Analysis Complete"
echo "================================================"