import os
import json
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import List, Dict

# --- Configuration ---
# Assuming this script is run from project root or scripts directory, we adjust paths relative to this file
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "../extracted_data")
RESULTS_DIR = os.path.join(BASE_DIR, "analysis_results")
GT_FILE = os.path.join(DATA_DIR, "accel_outliers_sample_ids.txt")

def load_gt_ids(filepath: str) -> set:
    """Loads ground truth outlier IDs from a text file."""
    if not os.path.exists(filepath):
        print(f"Error: GT file not found at {filepath}")
        return set()
    with open(filepath, 'r') as f:
        # Strip whitespace and ignore empty lines
        ids = {line.strip() for line in f if line.strip()}
    return ids

def analyze_strategy(json_path: str, gt_ids: set) -> Dict:
    """Analyzes a single strategy result file."""
    try:
        with open(json_path, 'r') as f:
            data = json.load(f)
    except Exception as e:
        print(f"Error loading {json_path}: {e}")
        return None

    strategy_name = data.get("strategy", os.path.basename(json_path).replace("results_", "").replace(".json", ""))
    points = data.get("points", [])
    
    if not points:
        return None

    # -- 1. Outlier Detection Analysis --
    # "is_outlier": True/False
    predicted_outliers_ids = {p["id"] for p in points if p.get("is_outlier")}
    total_gt = len(gt_ids)
    
    # GT Outliers that were detected
    detected_gt = gt_ids.intersection(predicted_outliers_ids)
    
    total_samples = len(points)
    total_negatives = total_samples - total_gt
    
    # False Positives: Directed as outlier but NOT in GT
    fp_count = len(predicted_outliers_ids) - len(detected_gt)
    fpr = fp_count / total_negatives if total_negatives > 0 else 0.0
    
    outlier_recall = len(detected_gt) / total_gt if total_gt > 0 else 0.0
    outlier_precision = len(detected_gt) / len(predicted_outliers_ids) if len(predicted_outliers_ids) > 0 else 0.0

    # -- 2. Cluster Analysis --
    # Check concentrations in available clusters
    # We look for keys starting with "cluster_"
    cluster_metrics = {}
    
    # Identfy available cluster keys from the first point
    first_point = points[0]
    cluster_keys = [k for k in first_point.keys() if k.startswith("cluster_")]
    
    best_cluster_recall = 0.0
    best_cluster_fpr = 0.0
    
    for k_key in cluster_keys:
        # Group points by cluster label
        clusters = {}
        for p in points:
            label = p.get(k_key)
            if label is None: continue
            if label not in clusters: clusters[label] = []
            clusters[label].append(p["id"])
            
        # For this K, find the cluster with the highest number of GT outliers
        max_gt_in_cluster = 0
        target_cluster_label = None
        
        for label, cluster_ids in clusters.items():
            cluster_id_set = set(cluster_ids)
            gt_in_this = len(gt_ids.intersection(cluster_id_set))
            if gt_in_this > max_gt_in_cluster:
                max_gt_in_cluster = gt_in_this
                target_cluster_label = label
        
        # Compute metrics for the best cluster
        if target_cluster_label is not None:
            recall = max_gt_in_cluster / total_gt if total_gt > 0 else 0.0
            
            # Precision: How many in this cluster are actually GT?
            total_in_cluster = len(clusters[target_cluster_label])
            precision = max_gt_in_cluster / total_in_cluster if total_in_cluster > 0 else 0.0
            
            # Cluster FPR: (Non-GT in Target Cluster) / Total Negatives
            fp_in_cluster = total_in_cluster - max_gt_in_cluster
            cluster_fpr = fp_in_cluster / total_negatives if total_negatives > 0 else 0.0
            
            cluster_metrics[k_key] = {
                "recall": recall,
                "precision": precision,
                "fpr": cluster_fpr,
                "target_label": target_cluster_label,
                "gt_count": max_gt_in_cluster
            }
            
            if recall > best_cluster_recall:
                best_cluster_recall = recall
                best_cluster_fpr = cluster_fpr

    return {
        "strategy": strategy_name,
        "outlier_recall": outlier_recall,
        "outlier_precision": outlier_precision,
        "fpr": fpr,
        "fp_count": fp_count,
        "best_cluster_recall": best_cluster_recall,
        "best_cluster_fpr": best_cluster_fpr,
        "cluster_details": cluster_metrics
    }

def main():
    print(f"Loading GT IDs from {GT_FILE}...")
    gt_ids = load_gt_ids(GT_FILE)
    if not gt_ids:
        print("No GT IDs found. Exiting.")
        return
    print(f"Found {len(gt_ids)} GT outlier IDs.")

    # Find result files
    result_files = glob.glob(os.path.join(RESULTS_DIR, "results_*.json"))
    if not result_files:
        print(f"No results_*.json files found in {RESULTS_DIR}")
        return

    results = []
    print(f"Analyzing {len(result_files)} result files...")
    
    for rf in result_files:
        res = analyze_strategy(rf, gt_ids)
        if res:
            results.append(res)

    if not results:
        print("No valid results extracted.")
        return

    # Sort by Outlier Recall
    results.sort(key=lambda x: x["outlier_recall"], reverse=True)

    # --- Print Table ---
    print("\n" + "="*130)
    print(f"{'Strategy':<35} | {'Outlier Recall':<15} | {'Cluster Recall':<15} | {'Outlier FPR':<12} | {'Cluster FPR':<12}")
    print("-" * 130)
    for r in results:
        print(f"{r['strategy']:<35} | {r['outlier_recall']:.4f}          | {r['best_cluster_recall']:.4f}          | {r['fpr']:.4f}       | {r['best_cluster_fpr']:.4f}")
    print("="*130 + "\n")

    # --- Visualization ---
    strategies = [r['strategy'] for r in results]
    o_recalls = [r['outlier_recall'] for r in results]
    c_recalls = [r['best_cluster_recall'] for r in results]
    o_fprs = [r['fpr'] for r in results]
    c_fprs = [r['best_cluster_fpr'] for r in results]

    x = np.arange(len(strategies))
    width = 0.2

    fig, ax = plt.subplots(figsize=(15, 7))
    
    # Colors
    c_iso = 'tab:blue'
    c_cluster = 'tab:orange'
    
    # Group 1: IsoForest
    rects1 = ax.bar(x - 1.5*width, o_recalls, width, label='IsoForest Recall', color=c_iso)
    rects2 = ax.bar(x - 0.5*width, o_fprs, width, label='IsoForest FPR', color=c_iso, alpha=0.5, hatch='//')
    
    # Group 2: Cluster
    rects3 = ax.bar(x + 0.5*width, c_recalls, width, label='Cluster Recall', color=c_cluster)
    rects4 = ax.bar(x + 1.5*width, c_fprs, width, label='Cluster FPR', color=c_cluster, alpha=0.5, hatch='//')

    ax.set_ylabel('Score')
    ax.set_title('Acceleration Outlier Detection vs Clustering')
    ax.set_xticks(x)
    ax.set_xticklabels(strategies, rotation=45, ha='right')
    ax.legend()
    ax.grid(axis='y', linestyle='--', alpha=0.7)

    plt.tight_layout()
    plot_path = os.path.join(RESULTS_DIR, "comparison_plot.png")
    plt.savefig(plot_path)
    print(f"Comparison plot saved to {plot_path}")

if __name__ == "__main__":
    main()
