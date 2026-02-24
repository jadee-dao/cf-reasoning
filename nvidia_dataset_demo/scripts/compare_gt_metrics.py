import os
import json
import argparse
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import seaborn as sns
import matplotlib.pyplot as plt

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="nuscenes_ego", help="Dataset name")
    args = parser.parse_args()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    
    # We can use any projection file since they should all have the same gt_outliers for dataset
    proj_path = os.path.join(project_root, 'analysis_results', 'projections', 'naive', f'{args.dataset}.json')
    
    if not os.path.exists(proj_path):
        print(f"Error: Projection file not found at {proj_path}")
        return
        
    print(f"Reading ground truth metrics from {proj_path}...")
    with open(proj_path, 'r') as f:
        data = json.load(f)
        
    records = []
    
    # Determine which metrics to extract based on dataset
    if args.dataset == "nvidia_demo":
        metrics_to_use = ['ade_p90', 'asil_ge_A', 'asil_ge_B']
    else:
        metrics_to_use = ['surprise_p90', 'surprise_p99', 'asil_ge_A', 'asil_ge_B']
        
    for pt in data.get("points", []):
        gt = pt.get("gt_outliers", {})
        record = {"id": pt["id"]}
        
        if args.dataset == "nvidia_demo":
            record["ade_p90"] = bool(gt.get("ade_p90", False))
            record["asil_ge_A"] = bool(gt.get("asil_ge_A", False))
            record["asil_ge_B"] = bool(gt.get("asil_ge_B", False))
        else:
            record["surprise_p90"] = bool(gt.get("surprise_potential_p90", False))
            record["surprise_p99"] = bool(gt.get("surprise_potential_p99", False))
            record["asil_ge_A"] = bool(gt.get("asil_ge_A", False))
            record["asil_ge_B"] = bool(gt.get("asil_ge_B", False))
            
        records.append(record)
        
    df = pd.DataFrame(records)
    print(f"Extracted ground truth for {len(df)} samples.")
    
    # Calculate sums (how many positives for each metric)
    print("\nTotal Positives per Metric:")
    print(df[metrics_to_use].sum())
    
    # Correlation Matrix (Pearson correlation coefficient for binary variables is the phi coefficient)
    corr_matrix = df[metrics_to_use].corr()
    
    print("\nCorrelation Matrix (Phi coefficient):")
    print(corr_matrix.round(4))
    
    print("\n--- Pairwise Overlap Analysis ---")
    
    for i in range(len(metrics_to_use)):
        for j in range(i+1, len(metrics_to_use)):
            m1, m2 = metrics_to_use[i], metrics_to_use[j]
            
            # Intersection: both true
            intersection = len(df[(df[m1] == True) & (df[m2] == True)])
            # Union: either true
            union = len(df[(df[m1] == True) | (df[m2] == True)])
            
            # Counts individually
            count_m1 = df[m1].sum()
            count_m2 = df[m2].sum()
            
            jaccard = intersection / union if union > 0 else 0
            
            print(f"> {m1} & {m2}:")
            print(f"  Intersection (Both True): {intersection} samples")
            print(f"  Union (Either True):      {union} samples")
            print(f"  Overlap Proportion (IoU): {jaccard:.2%}")
            
    # Plotting
    output_dir = os.path.join(project_root, 'analysis_results', 'ground_truth')
    os.makedirs(output_dir, exist_ok=True)
    plot_path = os.path.join(output_dir, 'gt_metrics_correlation.png')
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1, fmt=".3f")
    plt.title(f"Correlation between Ground Truth Metrics ({args.dataset})")
    plt.tight_layout()
    
    plot_path = os.path.join(output_dir, f'gt_metrics_correlation_{args.dataset}.png')
    plt.savefig(plot_path)
    print(f"\nCorrelation heatmap saved to: {plot_path}")

if __name__ == "__main__":
    main()
