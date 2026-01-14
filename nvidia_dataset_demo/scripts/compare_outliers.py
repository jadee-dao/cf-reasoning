import os
import json
import glob
import argparse
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict, Set, Optional

# --- Configuration ---
# Assuming this script is run from project root or scripts directory, we adjust paths relative to this file
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "../extracted_data")
RESULTS_DIR = os.path.join(BASE_DIR, "analysis_results")
CALIBRATION_DIR = os.path.join(DATA_DIR, "calibration_set")

def load_gt_ids(filepath: str, interval_sec: float = 1.5) -> Set[str]:
    """Loads ground truth outlier IDs from a file (txt or json).
       For JSON, it bins timestamps into interval buckets to match sub-video IDs.
    """
    if not os.path.exists(filepath):
        print(f"Error: GT file not found at {filepath}")
        return set()
    
    ids = set()
    interval_us = int(interval_sec * 1_000_000)

    if filepath.endswith('.txt'):
        try:
            with open(filepath, 'r') as f:
                # Strip whitespace and ignore empty lines
                ids = {line.strip() for line in f if line.strip()}
        except Exception as e:
            print(f"Error reading TXT file {filepath}: {e}")

    elif filepath.endswith('.json'):
        try:
            with open(filepath, 'r') as f:
                data = json.load(f)
                # Handle the format seen in worst-ade-log-10.json
                if "results" in data and isinstance(data["results"], dict):
                     for key, val in data["results"].items():
                         scene_id = val.get("scene_id", key)
                         
                         # If we have specific bad frames, rely on them for temporal matching
                         if "top3_worst" in val and isinstance(val["top3_worst"], list):
                             for frame in val["top3_worst"]:
                                 t_rel = frame.get("t_rel_us")
                                 if t_rel is not None:
                                     # Binning logic
                                     bin_start = (int(t_rel) // interval_us) * interval_us
                                     composite_id = f"{scene_id}_{bin_start}"
                                     ids.add(composite_id)
                         else:
                             # Fallback: Just the Scene ID (will fail strict matching of sub-videos)
                             ids.add(scene_id)
        except Exception as e:
            print(f"Error loading JSON GT file {filepath}: {e}")
            
    return ids

def select_gt_file() -> Optional[str]:
    """Interactively select a GT file from the calibration set."""
    if not os.path.exists(CALIBRATION_DIR):
         print(f"Calibration directory not found: {CALIBRATION_DIR}")
         return None

    # Filter for relevant files
    files = [f for f in os.listdir(CALIBRATION_DIR) if f.endswith('.txt') or f.endswith('.json')]
    files.sort()
    
    if not files:
        print(f"No valid GT files found in {CALIBRATION_DIR}")
        return None
        
    print("\n" + "="*50)
    print("Available Calibration Files:")
    print("="*50)
    for i, f in enumerate(files):
        print(f"[{i+1}] {f}")
    print("="*50)
    
    while True:
        try:
            choice = input("\nSelect a file number (or 'q' to quit): ").strip()
            if choice.lower() == 'q':
                return None
            idx = int(choice) - 1
            if 0 <= idx < len(files):
                return os.path.join(CALIBRATION_DIR, files[idx])
            else:
                print("Invalid selection. Please try again.")
        except ValueError:
            print("Please enter a valid number.")

def analyze_strategy(json_path: str, gt_ids: Set[str]) -> Optional[Dict]:
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
    
    # Check how many GTs are actually present in this log/file
    point_ids = {p["id"] for p in points}
    gt_present_in_dataset = gt_ids.intersection(point_ids)
    
    # GT Outliers that were detected
    detected_gt = gt_ids.intersection(predicted_outliers_ids)
    
    total_samples = len(points)
    # Total negatives are the points in the dataset that are NOT in GT
    total_negatives = total_samples - len(gt_present_in_dataset)

    
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
    parser = argparse.ArgumentParser(description="Compare outlier detection strategies against a ground truth set.")
    parser.add_argument("--gt-file", type=str, help="Path to the Ground Truth file (txt or json).")
    parser.add_argument("--interval", type=float, default=1.5, help="Time interval in seconds for binning (default 1.5).")
    args = parser.parse_args()

    # Select GT File
    gt_file_path = args.gt_file
    if not gt_file_path:
        gt_file_path = select_gt_file()
        
    if not gt_file_path:
        print("No GT file selected. Exiting.")
        return

    print(f"Loading GT IDs from {gt_file_path} (Interval: {args.interval}s)...")
    gt_ids = load_gt_ids(gt_file_path, interval_sec=args.interval)
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
    
    # Group 1: Outlier Detection
    rects1 = ax.bar(x - 1.5*width, o_recalls, width, label='Outlier Recall', color=c_iso)
    rects2 = ax.bar(x - 0.5*width, o_fprs, width, label='Outlier FPR', color=c_iso, alpha=0.5, hatch='//')
    
    # Group 2: Cluster
    rects3 = ax.bar(x + 0.5*width, c_recalls, width, label='Cluster Recall', color=c_cluster)
    rects4 = ax.bar(x + 1.5*width, c_fprs, width, label='Cluster FPR', color=c_cluster, alpha=0.5, hatch='//')

    ax.set_ylabel('Score')
    gt_filename = os.path.basename(gt_file_path)
    ax.set_title(f'Outlier Detection vs Clustering\n(GT: {gt_filename})')
    ax.set_xticks(x)
    ax.set_xticklabels(strategies, rotation=45, ha='right')
    
    # Set x-limits to prevent super wide bars if only 1 or few strategies
    # Each group is 4 * width = 0.8 wide centered at integer locations.
    # We want some padding.
    ax.set_xlim(-1, len(strategies))
    
    ax.legend()
    ax.grid(axis='y', linestyle='--', alpha=0.7)

    plt.tight_layout()
    plot_path = os.path.join(RESULTS_DIR, f"comparison_plot_{gt_filename}.png")
    plt.savefig(plot_path)
    print(f"Comparison plot saved to {plot_path}")

if __name__ == "__main__":
    main()
