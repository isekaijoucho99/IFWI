#!/usr/bin/env bash
set -euo pipefail
DEVICE="${1:-cuda:0}"
UPDATES="${2:-4000}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUTPUT_BASE="${SCRIPT_DIR}/../results/batch_$(date +%Y%m%d_%H%M%S)_$$"
for exp in feature_baseline depth_weighted_loss attention adaptive_lr prior_only combined_best; do
    python -u "${SCRIPT_DIR}/run_experiment.py" --config "${SCRIPT_DIR}/configs/${exp}.yaml" --device "$DEVICE" --iterations "$UPDATES" --seed 42 --output-dir "$OUTPUT_BASE"
done
python "${SCRIPT_DIR}/compare_results.py" --results-dir "$OUTPUT_BASE"
