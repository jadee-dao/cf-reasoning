import os
import json
import glob
import argparse
import numpy as np
import pandas as pd
from typing import Set, Dict, List
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.covariance import EllipticEnvelope
from sklearn.cluster import DBSCAN
from sklearn.mixture import GaussianMixture

# --- Configuration ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, "analysis_results")

def load_gt_ids(filepath: str, interval_sec: float = 1.5) -> Set[str]:
    """Loads ground truth outlier IDs from a file."""
    if not os.path.exists(filepath):
        print(f"Error: GT file not found at {filepath}")
        return set()
    
    ids = set()
    interval_us = int(interval_sec * 1_000_000)

    try:
        with open(filepath, 'r') as f:
            data = json.load(f)
            if "results" in data and isinstance(data["results"], dict):
                    for key, val in data["results"].items():
                        scene_id = val.get("scene_id", key)
                        if "top3_worst" in val and isinstance(val["top3_worst"], list):
                            for frame in val["top3_worst"]:
                                t_rel = frame.get("t_rel_us")
                                if t_rel is not None:
                                    bin_start = (int(t_rel) // interval_us) * interval_us
                                    composite_id = f"{scene_id}_{bin_start}"
                                    ids.add(composite_id)
                        else:
                            ids.add(scene_id)
    except Exception as e:
        print(f"Error loading GT file: {e}")
            
    return ids

def evaluate_methods(csv_path: str, gt_ids: Set[str]):
    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        print(f"Error reading {csv_path}: {e}")
        return None

    strategy_name = os.path.basename(csv_path).replace("projections_", "").replace(".csv", "")
    
    # Extract 5D PCA features
    feature_cols = [c for c in df.columns if c.startswith("pca_")]
    if not feature_cols:
        return None
        
    X = df[feature_cols].values
    ids = df["id"].values
    
    # Standardize for distance-based methods
    X_std = StandardScaler().fit_transform(X)
    
    results = {}
    total_gt = len(gt_ids)
    
    # Helper to calculate metrics
    def calc_metrics(pred_outlier_indices, method_name):
        predicted_ids = {ids[i] for i in pred_outlier_indices}
        detected_gt = gt_ids.intersection(predicted_ids)
        
        recall = len(detected_gt) / total_gt if total_gt > 0 else 0.0
        precision = len(detected_gt) / len(predicted_ids) if len(predicted_ids) > 0 else 0.0
        
        return {
            "method": method_name,
            "recall": recall,
            "precision": precision,
            "count": len(predicted_ids)
        }

    # 1. Isolation Forest
    iso = IsolationForest(contamination=0.05, random_state=42)
    y_iso = iso.fit_predict(X) # -1 outlier
    results["IsoForest"] = calc_metrics(np.where(y_iso == -1)[0], "IsoForest")

    # 2. Local Outlier Factor
    lof = LocalOutlierFactor(n_neighbors=20, contamination=0.05)
    y_lof = lof.fit_predict(X_std)
    results["LOF"] = calc_metrics(np.where(y_lof == -1)[0], "LOF")
    
    # 3. Elliptic Envelope
    try:
        ee = EllipticEnvelope(contamination=0.05, random_state=42)
        y_ee = ee.fit_predict(X)
        results["EllipticEnv"] = calc_metrics(np.where(y_ee == -1)[0], "EllipticEnv")
    except Exception:
        results["EllipticEnv"] = {"method": "EllipticEnv", "recall": 0, "precision": 0, "count": 0}

    # 4. DBSCAN (Noise + Small Clusters)
    # eps=0.5 is default for standardized data, usually good start
    db = DBSCAN(eps=2.0, min_samples=3) # Relaxed eps to find clusters
    y_db = db.fit_predict(X_std)
    
    # Outliers = Noise (-1) AND Small Clusters (< 1% of data)
    noise_indices = np.where(y_db == -1)[0]
    
    # Check cluster sizes
    labels, counts = np.unique(y_db, return_counts=True)
    small_cluster_labels = []
    threshold = max(3, len(X) * 0.01) # Clusters smaller than 1% or 3 points
    
    for label, count in zip(labels, counts):
        if label != -1 and count < threshold:
            small_cluster_labels.append(label)
            
    small_cluster_indices = np.where(np.isin(y_db, small_cluster_labels))[0]
    all_db_outliers = np.union1d(noise_indices, small_cluster_indices)
    
    results["DBSCAN"] = calc_metrics(all_db_outliers, "DBSCAN (Noise+Small)")

    # 5. GMM (Low Log-Likelihood)
    gmm = GaussianMixture(n_components=5, random_state=42)
    gmm.fit(X)
    scores = gmm.score_samples(X)
    # Threshold at bottom 5%
    limit = np.percentile(scores, 5)
    y_gmm = scores < limit
    results["GMM"] = calc_metrics(np.where(y_gmm)[0], "GMM (Low LL)")
    
    # 6. Random Baseline
    # Select random 5% of samples to match contamination
    n_random = int(len(X) * 0.05)
    random_indices = np.random.choice(len(X), n_random, replace=False)
    results["Random"] = calc_metrics(random_indices, "Random (5%)")
    
    # Flatten for return
    flat_results = []
    for method, metrics in results.items():
        metrics["strategy"] = strategy_name
        flat_results.append(metrics)
        
    return flat_results

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gt-file", type=str, required=True)
    parser.add_argument("--interval", type=float, default=1.5)
    args = parser.parse_args()
    
    print(f"Loading GT from {args.gt_file}...")
    gt_ids = load_gt_ids(args.gt_file, args.interval)
    print(f"Found {len(gt_ids)} GT IDs.")
    
    csv_files = glob.glob(os.path.join(OUTPUT_DIR, "projections_*.csv"))
    if not csv_files:
        print("No projection CSVs found.")
        return

    all_metrics = []
    print(f"Evaluating {len(csv_files)} strategies...")
    
    for csv in csv_files:
        res = evaluate_methods(csv, gt_ids)
        if res:
            all_metrics.extend(res)
            
    # Create Table
    res_df = pd.DataFrame(all_metrics)
    if res_df.empty:
        print("No results.")
        return
        
    # Sort by Recall
    res_df = res_df.sort_values("recall", ascending=False)
    
    print("\n" + "="*100)
    print(f"{'Strategy':<35} | {'Method':<20} | {'Recall':<10} | {'Precision':<10} | {'Count':<8}")
    print("-" * 100)
    
    for _, row in res_df.iterrows():
        print(f"{row['strategy']:<35} | {row['method']:<20} | {row['recall']:.4f}     | {row['precision']:.4f}     | {row['count']:<8}")
    print("="*100 + "\n")

    # Save
    out_path = os.path.join(OUTPUT_DIR, "outlier_method_evaluation.csv")
    res_df.to_csv(out_path, index=False)
    print(f"Saved evaluation to {out_path}")

if __name__ == "__main__":
    main()
