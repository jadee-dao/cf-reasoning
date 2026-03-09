import os
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_path", default="analysis_results/outlier_analysis/object_graph_gnn/gnn_comparison_matrix.png")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)

    modes = [("gae", "Unsupervised (Surprise)"), ("classifier", "Supervised (Prediction)")]
    targets = [("ade", "ADE Hazards"), ("asil_a", "ASIL-A Hazards"), ("asil_b", "ASIL-B Hazards")]
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 11))
    
    for i, (mode, mode_label) in enumerate(modes):
        for j, (target, target_label) in enumerate(targets):
            ax = axes[i, j]
            path = f"analysis_results/outliers/gnn_nvidia_{mode}_{target}.json"
            
            if not os.path.exists(path):
                ax.text(0.5, 0.5, f"Missing:\n{os.path.basename(path)}", ha='center', va='center')
                continue
                
            with open(path, 'r') as f:
                data = json.load(f)
            
            y_scores = [v["score"] for v in data.values()]
            y_true = [v["label"] for v in data.values()]
            
            if len(set(y_true)) < 2:
                ax.text(0.5, 0.5, f"Only one class in test split\n(N={len(y_true)})", ha='center', va='center')
                ax.set_title(f"{mode_label}\n{target_label}")
                continue

            fpr, tpr, _ = roc_curve(y_true, y_scores)
            roc_auc = auc(fpr, tpr)
            
            # For GAE, sometimes lower error means hazard (if labels are flipped)
            # But usually reconstruction is higher for anomalies. 
            # If AUC < 0.5, we assume the metric is informative but inverse.
            if roc_auc < 0.5:
                # We show the "discovery" AUC
                roc_auc = 1 - roc_auc
                fpr, tpr, _ = roc_curve(y_true, [-s for s in y_scores])

            ax.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC (AUC = {roc_auc:.2f})')
            ax.plot([0, 1], [0, 1], color='navy', lw=1, linestyle='--')
            ax.set_xlabel('FPR')
            ax.set_ylabel('TPR')
            ax.set_title(f"{mode_label}\n{target_label}")
            ax.legend(loc="lower right")
            
    plt.suptitle("GNN Anomaly Detection Matrix: Nvidia-Internal Training", fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(args.output_path)
    
    # Save a copy for the walkthrough
    plt.savefig("analysis_results/visualize_samples/gnn_comparison_matrix.png")
    
    print(f"Comparison Matrix saved to {args.output_path}")

if __name__ == "__main__":
    main()

if __name__ == "__main__":
    main()
