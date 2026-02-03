#!/bin/bash

# Default strategy
STRATEGY=""

# Parse arguments
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --strategy) STRATEGY="$2"; shift ;;
        *) echo "Unknown parameter: $1"; exit 1 ;;
    esac
    shift
done

if [ -z "$STRATEGY" ]; then
    echo "Error: --strategy argument is required (e.g. video_mae)"
    exit 1
fi

echo "========================================================"
echo "Starting Pipeline for Strategy: $STRATEGY"
echo "========================================================"

# Directory of this script
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# 1. Generate Embeddings (for known datasets)
echo ""
echo "[1/3] Generating Embeddings..."

# Nvidia Demo
echo ">> Processing nvidia_demo..."
python3 ../src/analysis/run_strategy.py --strategy "$STRATEGY" --dataset nvidia_demo
if [ $? -ne 0 ]; then
    echo "Error processing nvidia_demo"
    exit 1
fi

# NuScenes Ego
echo ">> Processing nuscenes_ego..."
python3 ../src/analysis/run_strategy.py --strategy "$STRATEGY" --dataset nuscenes_ego
if [ $? -ne 0 ]; then
    # Warn but maybe continue if dataset is optional?
    echo "Warning: Error processing nuscenes_ego, or maybe failure to load? Continuing..."
fi

# 2. Compute Similarities
echo ""
echo "[2/3] Computing Global Similarities..."
python3 ../src/analysis/compute_similarities.py --strategy "$STRATEGY"
if [ $? -ne 0 ]; then
    echo "Error in compute_similarities.py"
    exit 1
fi

# 3. Compute Joint Projections
echo ""
echo "[3/3] Computing Joint Projections..."
python3 ../src/analysis/compute_projections.py --strategy "$STRATEGY"
if [ $? -ne 0 ]; then
    echo "Error in compute_projections.py"
    exit 1
fi

echo ""
echo "========================================================"
echo "Strategy Pipeline Complete!"
echo "========================================================"
