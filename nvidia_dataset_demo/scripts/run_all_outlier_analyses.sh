#!/bin/bash

# Default gt_metric if not provided as first argument
GT_METRIC=${1:-surprise_potential_p90}

# Define paths
BASE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
EMBEDDINGS_DIR="$BASE_DIR/analysis_results/embeddings"
SCRIPT_PATH="$BASE_DIR/src/analysis/analyze_outliers.py"

echo "Looking for strategies in $EMBEDDINGS_DIR..."

# Iterate over each subdirectory in the embeddings folder
for strategy_path in "$EMBEDDINGS_DIR"/*; do
    if [ -d "$strategy_path" ]; then
        strategy_name=$(basename "$strategy_path")
        echo "---------------------------------------------------"
        echo "Running outlier analysis for strategy: $strategy_name"
        echo "---------------------------------------------------"
        
        python3 "$SCRIPT_PATH" --strategy "$strategy_name" --gt_metric "$GT_METRIC"
        
        echo ""
    fi
done

echo "Batch analysis complete. Check analysis_results/outlier_analysis/ for plots."
