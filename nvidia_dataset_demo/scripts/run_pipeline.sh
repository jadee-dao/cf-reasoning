#!/bin/bash
set -e

# Default values
DATASET="nvidia_demo"
STRATEGY="naive"
LIMIT=""
INTERVAL=1.5

# Help function
usage() {
    echo "Usage: $0 --dataset <name> --strategy <name> [--limit <N>]"
    echo "  --dataset:  Dataset name (default: nvidia_demo)"
    echo "  --strategy: Embedding strategy (default: naive)"
    echo "  --limit:    Limit number of samples (optional)"
    exit 1
}

# Parse arguments
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --dataset) DATASET="$2"; shift ;;
        --strategy) STRATEGY="$2"; shift ;;
        --limit) LIMIT="$2"; shift ;;
        -h|--help) usage ;;
        *) echo "Unknown parameter passed: $1"; usage ;;
    esac
    shift
done

echo "========================================================"
echo "Running Pipeline for Dataset: $DATASET, Strategy: $STRATEGY"
if [ -n "$LIMIT" ]; then
    echo "Limit: $LIMIT"
fi
echo "========================================================"

# 1. Process Dataset
# This step verifies samples exist and prepares them.
echo ""
echo "[1/3] Processing/Verifying Dataset..."
CMD_PROCESS="python3 scripts/process_dataset.py --dataset_name $DATASET --interval $INTERVAL"
if [ -n "$LIMIT" ]; then
    CMD_PROCESS="$CMD_PROCESS --limit $LIMIT"
fi
echo "Running: $CMD_PROCESS"
$CMD_PROCESS

# 2. Run Strategy
# This generates embeddings and similarities.
echo ""
echo "[2/3] Running Embedding Strategy..."
CMD_STRATEGY="python3 scripts/run_strategy.py --strategy $STRATEGY --dataset $DATASET"
if [ -n "$LIMIT" ]; then
    CMD_STRATEGY="$CMD_STRATEGY --limit $LIMIT"
fi
echo "Running: $CMD_STRATEGY"
$CMD_STRATEGY

# 3. Compute Projections
# This computes UMAP/TSNE for visualization.
echo ""
echo "[3/3] Computing Projections..."
CMD_PROJ="python3 scripts/compute_projections.py --dataset $DATASET --strategy $STRATEGY"
echo "Running: $CMD_PROJ"
$CMD_PROJ

echo ""
echo "========================================================"
echo "Pipeline Complete!"
echo "========================================================"
