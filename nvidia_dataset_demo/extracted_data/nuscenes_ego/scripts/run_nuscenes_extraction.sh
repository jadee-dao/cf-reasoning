#!/bin/bash

# Default limit
LIMIT=100

# Parse arguments
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --limit) LIMIT="$2"; shift ;;
        *) echo "Unknown parameter: $1"; exit 1 ;;
    esac
    shift
done

echo "========================================================"
echo "Starting NuScenes Extraction Pipeline"
echo "Limit: $LIMIT scenes"
echo "========================================================"

# Directory of this script
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# 1. Extract Images
echo ""
echo "[1/4] Extracting Images..."
python3 extract_ego_images.py --limit "$LIMIT"
if [ $? -ne 0 ]; then
    echo "Error in extract_ego_images.py"
    exit 1
fi

# 2. Create Videos
echo ""
echo "[2/4] Creating Videos..."
python3 create_ego_videos.py --limit "$LIMIT"
if [ $? -ne 0 ]; then
    echo "Error in create_ego_videos.py"
    exit 1
fi

# 3. Cut Ego Samples (9s clips)
echo ""
echo "[3/4] Cutting Ego Samples..."
python3 cut_ego_samples.py --limit "$LIMIT"
if [ $? -ne 0 ]; then
    echo "Error in cut_ego_samples.py"
    exit 1
fi

# 4. Cut Comparison Samples (1.5s clips)
echo ""
echo "[4/4] Cutting Comparison Samples..."
# cut_comparison_samples.py iterates over whatever is in processed input dir, so it doesn't need limit arg
# It just processes the output of step 3.
python3 cut_comparison_samples.py
if [ $? -ne 0 ]; then
    echo "Error in cut_comparison_samples.py"
    exit 1
fi

echo ""
echo "========================================================"
echo "Pipeline Complete!"
echo "========================================================"
