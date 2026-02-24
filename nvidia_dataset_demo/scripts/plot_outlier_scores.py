
import os
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
import glob
import numpy as np
import re
import argparse
from sklearn.metrics import roc_curve, auc

def parse_filename(filename):
    """
    Parses the filename to extract strategy and outlier method.
    Expected format: <strategy>_<outlier_method>_scores.json
    Known outlier methods: isolation_forest, lof, one_class_svm
    """
    base_name = os.path.basename(filename)
    name_no_ext = os.path.splitext(base_name)[0]
    
    # Remove '_scores' suffix
    if name_no_ext.endswith('_scores'):
        name_no_ext = name_no_ext[:-7]
        
    # Order matters: check more specific keys first
    # e.g., lof_k20 before lof (though here lof_k20 doesn't end with just 'lof' if we check properly)
    
    # We treat 'mlp' and 'lof_k*' as methods
    if 'mlp' in name_no_ext:
         return name_no_ext.replace('_mlp', ''), 'mlp'
         
    if 'lof_k' in name_no_ext:
        # extract strategy. <strat>_lof_k<N>
        # find where lof_k starts
        match = re.search(r'_(lof_k\d+)$', name_no_ext)
        if match:
             method = match.group(1)
             strategy = name_no_ext[:-len(method)-1]
             return strategy, method
             
    methods = ['isolation_forest', 'lof', 'one_class_svm']
    
    for method in methods:
        if name_no_ext.endswith(method):
            strategy = name_no_ext[:-len(method)-1] # -1 for the underscore
            return strategy, method
            
    return None, None

def load_ground_truth(dataset, gt_metric):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    projections_dir = os.path.join(project_root, 'analysis_results', 'projections')
    
    # Find any projection file for this dataset to read GT
    files = glob.glob(os.path.join(projections_dir, '*', f"{dataset}.json"))
    if not files:
        print(f"Error: No projection files found for dataset {dataset}")
        return None
        
    proj_path = files[0]
    print(f"Loading Ground Truth from {proj_path} using metric: {gt_metric}")
    
    with open(proj_path, 'r') as f:
        data = json.load(f)
        
    outliers_set = set()
    for pt in data.get("points", []):
        gt = pt.get("gt_outliers", {})
        if gt.get(gt_metric):
            outliers_set.add(pt["id"])
            
    return set(outliers_set)

def load_scores(data_dir):
    data = []
    files = glob.glob(os.path.join(data_dir, '*_scores.json'))
    
    print(f"Found {len(files)} score files in {data_dir}")
    
    for f in files:
        strategy, method = parse_filename(f)
        if not strategy or not method:
            # print(f"Skipping {f}: Could not parse strategy/method")
            continue
            
        try:
            with open(f, 'r') as json_file:
                scores_dict = json.load(json_file)
                
            for scene_id, info in scores_dict.items():
                # Filter for nuscenes_ego only
                if info.get('dataset') != 'nuscenes_ego':
                    continue
                    
                data.append({
                    'Strategy': strategy,
                    'Method': method,
                    'Score': info['score'],
                    'Scene': scene_id
                })
        except Exception as e:
            print(f"Error loading {f}: {e}")
            
    return pd.DataFrame(data)

def plot_roc_curves(df, gt_outliers, output_path):
    
    methods = sorted(df['Method'].unique())
    strategies = df['Strategy'].unique()
    
    print(f"Plotting ROC for methods: {methods}")
    
    # Calculate number of rows/cols
    n_methods = len(methods)
    cols = 3
    rows = (n_methods + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(6*cols, 6*rows), sharey=True)
    axes = axes.flatten()
    
    # Hide unused axes
    for i in range(n_methods, len(axes)):
        axes[i].axis('off')
        
    for i, method in enumerate(methods):
        ax = axes[i]
        subset = df[df['Method'] == method]
        
        # We need to compute ROC for each strategy
        for strategy in strategies:
            strat_data = subset[subset['Strategy'] == strategy]
            
            if strat_data.empty:
                continue
                
            y_true = []
            y_scores = []
            
            for _, row in strat_data.iterrows():
                scene_id = row['Scene']
                score = row['Score']
                
                # Check if this scene is in ground truth outliers
                is_outlier = 1 if scene_id in gt_outliers else 0
                
                y_true.append(is_outlier)
                y_scores.append(score)
            
            if len(set(y_true)) < 2:
                # print(f"Skipping {strategy} for {method}: only one class present.")
                continue
                
            fpr, tpr, _ = roc_curve(y_true, y_scores)
            roc_auc = auc(fpr, tpr)
            
            ax.plot(fpr, tpr, lw=2, label=f'{strategy} (AUC = {roc_auc:.2f})')
        
        ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title(method)
        ax.legend(loc="lower right", fontsize='small')
        ax.grid(True, alpha=0.3)
        
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    print(f"ROC Plot saved to {output_path}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gt_metric", type=str, default="surprise_potential_p90", help="Ground truth metric to use (e.g., asil_ge_A)")
    parser.add_argument("--dataset", type=str, default="nuscenes_ego", help="Dataset name")
    args = parser.parse_args()

    # Define paths relative to script location
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir) # ../
    
    data_dir = os.path.join(project_root, 'analysis_results', 'outliers')

    if not os.path.exists(data_dir):
        # absolute path fallback
        data_dir = '/home/jadelynn/cf-reasoning/nvidia_dataset_demo/analysis_results/outliers'

    gt_outliers = load_ground_truth(args.dataset, args.gt_metric)
    if gt_outliers is None:
        return

    print(f"Loading scores from {data_dir}")
    df = load_scores(data_dir)
    
    if df.empty:
        print("No data found!")
        return

    output_file = os.path.join(data_dir, f'outlier_roc_curves_{args.gt_metric}.png')
    plot_roc_curves(df, gt_outliers, output_file)

if __name__ == "__main__":
    main()
