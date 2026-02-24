#!/bin/bash
strategies=("naive" "fastvit_attention" "video_mae" "intern_video" "object_semantics" "video_vit" "object_graph")

for strategy in "${strategies[@]}"; do
    echo "Running analysis for $strategy..."
    python3 ../src/analysis/analyze_outliers.py --strategy "$strategy"
done
