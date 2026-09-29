#!/bin/bash
# Run all IFWI improvement experiments
# Usage: bash run_all_experiments.sh [device]

DEVICE=${1:-cuda:0}
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUTPUT_BASE="${SCRIPT_DIR}/../results"

echo "========================================="
echo "Running All IFWI Improvement Experiments"
echo "========================================="
echo "Device: $DEVICE"
echo "Output directory: $OUTPUT_BASE"
echo ""

# List of experiments to run
EXPERIMENTS=(
    "baseline"
    "depth_weighted_loss"
    "attention"
    "adaptive_lr"
    "combined_best"
)

# Track success/failure
declare -a SUCCESS=()
declare -a FAILED=()

for exp in "${EXPERIMENTS[@]}"; do
    echo ""
    echo "========================================="
    echo "Starting Experiment: $exp"
    echo "========================================="

    CONFIG="${SCRIPT_DIR}/configs/${exp}.yaml"

    if [ ! -f "$CONFIG" ]; then
        echo "ERROR: Config file not found: $CONFIG"
        FAILED+=("$exp (config not found)")
        continue
    fi

    # Run experiment
    python "${SCRIPT_DIR}/run_experiment.py" \
        --config "$CONFIG" \
        --output-dir "$OUTPUT_BASE" \
        --device "$DEVICE" \
        --seed 42

    # Check exit status
    if [ $? -eq 0 ]; then
        echo "✓ Experiment '$exp' completed successfully"
        SUCCESS+=("$exp")
    else
        echo "✗ Experiment '$exp' failed"
        FAILED+=("$exp")
    fi
done

# Summary
echo ""
echo "========================================="
echo "EXPERIMENT SUMMARY"
echo "========================================="
echo "Successful: ${#SUCCESS[@]}/${#EXPERIMENTS[@]}"
for exp in "${SUCCESS[@]}"; do
    echo "  ✓ $exp"
done

if [ ${#FAILED[@]} -gt 0 ]; then
    echo ""
    echo "Failed: ${#FAILED[@]}/${#EXPERIMENTS[@]}"
    for exp in "${FAILED[@]}"; do
        echo "  ✗ $exp"
    done
fi

echo ""
echo "Results saved to: $OUTPUT_BASE"
echo ""

# Generate comparison report if all experiments succeeded
if [ ${#FAILED[@]} -eq 0 ]; then
    echo "All experiments completed. Generating comparison report..."
    python "${SCRIPT_DIR}/compare_results.py" --results-dir "$OUTPUT_BASE"
fi
