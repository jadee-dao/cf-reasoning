
import argparse
import os
import glob
import pickle
import json
import numpy as np
import matplotlib
matplotlib.use('Agg') # Force headless backend
import matplotlib.pyplot as plt
from sklearn.neighbors import LocalOutlierFactor
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score
import sys

# Define paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUTPUT_ROOT = os.path.join(BASE_DIR, "analysis_results")
EMBEDDINGS_DIR = os.path.join(OUTPUT_ROOT, "embeddings")
GT_OUTLIERS_PATH = os.path.join(BASE_DIR, "extracted_data", "nuscenes_ego", "calibration", "nuscenes_percentile_outliers.json")
ASIL_SCORES_PATH = os.path.join(OUTPUT_ROOT, "ground_truth", "asil_scores_nuscenes_ego.json")
RESULTS_DIR = os.path.join(OUTPUT_ROOT, "outlier_analysis")
OUTLIERS_DIR = os.path.join(OUTPUT_ROOT, "outliers")

def load_embeddings(strategy, dataset_filter=None):
    """
    Loads all embeddings for a given strategy.
    If dataset_filter is provided (e.g. 'nvidia_demo'), only loads matching files.
    """
    emb_dir = os.path.join(EMBEDDINGS_DIR, strategy)
    if not os.path.exists(emb_dir):
        print(f"Error: Embeddings directory not found: {emb_dir}")
        return None, None

    # Filter files if dataset specified
    pattern = "*.pkl"
    if dataset_filter:
        pattern = f"{dataset_filter}.pkl"
        
    embedding_files = glob.glob(os.path.join(emb_dir, pattern))
    if not embedding_files:
        print(f"No embedding files found for strategy '{strategy}' matching '{pattern}'.")
        return None, None

    all_ids = []
    all_embeddings = []

    print(f"Loading {len(embedding_files)} embedding files for strategy '{strategy}'...")
    for fpath in embedding_files:
        with open(fpath, "rb") as f:
            data = pickle.load(f)
        
        # Ensure data integrity
        if "ids" not in data or "embeddings" not in data:
            print(f"Skipping malformed file: {fpath}")
            continue
            
        ids = data["ids"]
        emb = data["embeddings"]
        
        all_ids.extend(ids)
        if len(all_embeddings) == 0:
            all_embeddings = emb
        else:
            all_embeddings = np.concatenate((all_embeddings, emb), axis=0)
            
    print(f"Total samples loaded from {dataset_filter if dataset_filter else 'all'}: {len(all_ids)}")
    if len(all_ids) == 0:
        return None, None
    return np.array(all_ids), all_embeddings

def load_ground_truth(gt_type='surprise'):
    """Loads the ground truth outliers from JSON based on gt_type."""
    if gt_type == 'surprise':
        if not os.path.exists(GT_OUTLIERS_PATH):
            print(f"Warning: Ground truth file not found at {GT_OUTLIERS_PATH}")
            return None
            
        with open(GT_OUTLIERS_PATH, "r") as f:
            data = json.load(f)
        
        outliers_map = data.get("generated_samples", {})
        return outliers_map
    
    elif gt_type in ['asil_ge_A', 'asil_ge_B']:
        if not os.path.exists(ASIL_SCORES_PATH):
            print(f"Warning: ASIL ground truth file not found at {ASIL_SCORES_PATH}")
            return None
            
        with open(ASIL_SCORES_PATH, "r") as f:
            asil_data = json.load(f)
            
        outliers_map = {}
        for uid, scores in asil_data.items():
            score = scores.get("asil_score", 0)
            if gt_type == 'asil_ge_A' and score >= 1:
                outliers_map[uid] = True
            elif gt_type == 'asil_ge_B' and score >= 2:
                outliers_map[uid] = True
                
        return outliers_map
    else:
        print(f"Unknown gt_type: {gt_type}")
        return None

def compute_outlier_scores(embeddings, method='lof', k=20, random_state=42):
    """
    Computes outlier scores using the specified method (Unsupervised).
    Returns scores where HIGHER value means MORE outlier-ish.
    """
    n_samples = len(embeddings)
    scores = np.zeros(n_samples)

    print(f"Computing scores using {method} (n={n_samples})...")
    
    if method == 'lof':
        # Local Outlier Factor
        if n_samples < k:
            print(f"Warning: Not enough samples ({n_samples}) for k={k}. Adjusting k.")
            k = max(1, n_samples - 1)
        
        lof = LocalOutlierFactor(n_neighbors=k)
        lof.fit(embeddings)
        # negative_outlier_factor_: higher is better (inlier).
        # We want higher = outlier. So multiply by -1.
        scores = -1 * lof.negative_outlier_factor_

    elif method == 'isolation_forest':
        # Isolation Forest
        model = IsolationForest(random_state=random_state, contamination='auto')
        model.fit(embeddings)
        # decision_function: average anomaly score. 
        # The lower, the more abnormal. Negative values are outliers, positive are inliers.
        # We want higher = outlier. So multiply by -1.
        scores = -1 * model.decision_function(embeddings)

    elif method == 'one_class_svm':
        # One-Class SVM
        # nu=0.1 is a common default for outlier fraction/regularization
        model = OneClassSVM(nu=0.1, kernel="rbf", gamma="auto")
        model.fit(embeddings)
        # decision_function: Signed distance to the separating hyperplane.
        # Positive for inliers, negative for outliers.
        # We want higher = outlier. So multiply by -1.
        scores = -1 * model.decision_function(embeddings)

    else:
        print(f"Unknown method strategy: {method}")
        return None

    return scores

def run_analysis_for_method(strategy_name, method_name, k, nuscenes_data, nvidia_data, gt_outliers, output_dir, gt_type):
    print(f"\n=== Running Analysis: {method_name} (GT: {gt_type}) ===")
    
    # Unpack data
    nuscenes_ids, nuscenes_emb = nuscenes_data
    nvidia_ids, nvidia_emb = nvidia_data
    
    # Special handling for MLP (Supervised)
    if method_name.startswith('mlp'):
        if nuscenes_ids is None or not gt_outliers:
            print("MLP requires NuScenes data and Ground Truth. Skipping.")
            return

        print("Running MLP Analysis with Scene-Based Split...")
        # Prepare targets
        y = np.array([1 if uid in gt_outliers else 0 for uid in nuscenes_ids])
        
        # Extract Scene IDs (Assumption: format scene-XXXX_sampleID)
        # We group by the part before the first underscore
        scene_ids = np.array([uid.split('_')[0] for uid in nuscenes_ids])
        unique_scenes = np.unique(scene_ids)
        
        # Determine which scenes have outliers for stratified splitting
        scene_has_outlier = {}
        for uid, label in zip(nuscenes_ids, y):
            s_id = uid.split('_')[0]
            if label == 1:
                scene_has_outlier[s_id] = True
            elif s_id not in scene_has_outlier:
                scene_has_outlier[s_id] = False
        
        outlier_scenes = [s for s, has in scene_has_outlier.items() if has]
        normal_scenes = [s for s, has in scene_has_outlier.items() if not has]
        
        print(f"  Total Scenes: {len(unique_scenes)}")
        print(f"  Outlier Scenes: {len(outlier_scenes)}, Normal Scenes: {len(normal_scenes)}")
        
        # Split scenes (aim for 70/30 split)
        # We try to put some outlier scenes in both train and test if possible
        train_scenes = []
        test_scenes = []
        
        # Helper to split list
        def split_list(lst, test_ratio=0.3):
            # Deterministic shuffle based on content/random_state concept
            # Using sklearn's train_test_split on the list of scenes
            if len(lst) < 2:
                # Cannot split effectively, put in train (or handle as edge case)
                return lst, []
            tr, te = train_test_split(lst, test_size=test_ratio, random_state=42)
            return tr, te

        if outlier_scenes:
            o_train, o_test = split_list(outlier_scenes, 0.3)
            # Ensure we have at least one outlier scene in train if possible to learn the class
            if len(outlier_scenes) > 1 and len(o_train) == 0:
                 # Force one into train if random split put all in test (unlikely with 0.3 but possible with small N)
                 o_train = [o_test.pop(0)]
            
            train_scenes.extend(o_train)
            test_scenes.extend(o_test)
            
        if normal_scenes:
            n_train, n_test = split_list(normal_scenes, 0.3)
            train_scenes.extend(n_train)
            test_scenes.extend(n_test)
            
        train_scenes_set = set(train_scenes)
        test_scenes_set = set(test_scenes)
        
        # Create Masks
        train_mask = np.isin(scene_ids, list(train_scenes_set))
        test_mask = np.isin(scene_ids, list(test_scenes_set))
        
        X_train = nuscenes_emb[train_mask]
        y_train = y[train_mask]
        X_test = nuscenes_emb[test_mask]
        y_test = y[test_mask]
        ids_test = nuscenes_ids[test_mask]
        
        print(f"  Train: {len(X_train)} samples, {sum(y_train)} outliers")
        print(f"  Test: {len(X_test)} samples, {sum(y_test)} outliers")
        
        if len(X_train) == 0 or len(X_test) == 0:
             print("  Error: Split resulted in empty train or test set. Skipping MLP.")
             return
             
        if sum(y_train) == 0:
             print("  Warning: No outliers in training set. MLP cannot learn the outlier class.")
             # Proceeding anyway usually results in predicting all zeros.
        
        # Train
        mlp = MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=500, random_state=42)
        mlp.fit(X_train, y_train)
        
        # Predict (Probabilities for class 1 = Outlier)
        # Check if model learned only one class
        if len(mlp.classes_) < 2:
             # Likely only 0s in train
             y_scores = np.zeros(len(X_test))
        else:
             y_scores = mlp.predict_proba(X_test)[:, 1]
        
        # Calculate Metrics on Test Set
        # Handle case where test set has no outliers (can calculate ROC but might warn)
        if sum(y_test) == 0:
             print("  Warning: No outliers in test set. AUROC undefined/not useful.")
             roc_auc = 0.0
             pr_auc = 0.0
             fpr, tpr = [0], [0] # dummy
        else:
            fpr, tpr, roc_thresholds = roc_curve(y_test, y_scores)
            roc_auc = auc(fpr, tpr)
            
            precision, recall, pr_thresholds = precision_recall_curve(y_test, y_scores)
            pr_auc = average_precision_score(y_test, y_scores)

        metrics = {
            "roc_auc": float(roc_auc),
            "pr_auc": float(pr_auc)
        }
        print(f"  MLP Test AUROC: {roc_auc:.4f}, AUPRC: {pr_auc:.4f}")
        
        # Save ROC Plot
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'ROC: {strategy_name} - {method_name} (GT: {gt_type})')
        plt.legend(loc="lower right")
        plt.grid(True, alpha=0.3)
        roc_path = os.path.join(output_dir, f"{method_name}_roc.png")
        plt.savefig(roc_path)
        plt.close()
        
        # Prediction on Nvidia (if available) - just for saving
        nvidia_scores = None
        if nvidia_ids is not None:
             if len(mlp.classes_) < 2:
                  nvidia_scores = np.zeros(len(nvidia_emb))
             else:
                  nvidia_scores = mlp.predict_proba(nvidia_emb)[:, 1]

        # Save Scores (Only for Test set of NuScenes + All Nvidia)
        scores_data = {}
        for i, uid in enumerate(ids_test):
            scores_data[uid] = {
                "score": float(y_scores[i]),
                "dataset": "nuscenes_ego" # implicitly test set
            }
            
        if nvidia_scores is not None:
             for i, uid in enumerate(nvidia_ids):
                scores_data[uid] = {
                    "score": float(nvidia_scores[i]),
                    "dataset": "nvidia_demo"
                }

        scores_out_path = os.path.join(OUTLIERS_DIR, f"{strategy_name}_{method_name}_scores.json")
        os.makedirs(os.path.dirname(scores_out_path), exist_ok=True)
        with open(scores_out_path, "w") as f:
            json.dump(scores_data, f, indent=2)
            
        metrics_out_path = os.path.join(OUTLIERS_DIR, f"{strategy_name}_{method_name}_metrics.json")
        with open(metrics_out_path, "w") as f:
            json.dump(metrics, f, indent=2)
            
        return

    # Unsupervised Methods (LOF, IF, OCSVM)
    
    # 1. Compute Scores
    nuscenes_scores = None
    if nuscenes_ids is not None:
        nuscenes_scores = compute_outlier_scores(nuscenes_emb, method=method_name.split('_k')[0], k=k)
        
    nvidia_scores = None
    if nvidia_ids is not None:
        nvidia_scores = compute_outlier_scores(nvidia_emb, method=method_name.split('_k')[0], k=k)

    if nuscenes_scores is None and nvidia_scores is None:
        print("No scores computed. Skipping.")
        return

    # 2. Separate GT (NuScenes only) & Compute Metrics
    nuscenes_gt_scores = []
    nuscenes_normal_scores = []
    metrics = {}
    
    if nuscenes_scores is not None and gt_outliers:
        y_true = []
        y_scores = []
        
        for i, file_id in enumerate(nuscenes_ids):
            is_outlier = file_id in gt_outliers
            y_true.append(1 if is_outlier else 0)
            y_scores.append(nuscenes_scores[i])
            
            if is_outlier:
                nuscenes_gt_scores.append(nuscenes_scores[i])
            else:
                nuscenes_normal_scores.append(nuscenes_scores[i])
                
        # --- Metrics Calculation ---
        if len(nuscenes_gt_scores) > 0 and len(nuscenes_normal_scores) > 0:
            fpr, tpr, roc_thresholds = roc_curve(y_true, y_scores)
            roc_auc = auc(fpr, tpr)
            
            precision, recall, pr_thresholds = precision_recall_curve(y_true, y_scores)
            pr_auc = average_precision_score(y_true, y_scores)
            
            metrics = {
                "roc_auc": float(roc_auc),
                "pr_auc": float(pr_auc)
            }
            print(f"  AUROC: {roc_auc:.4f}, AUPRC: {pr_auc:.4f}")
            
            # Plot ROC
            plt.figure(figsize=(8, 6))
            plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
            plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title(f'ROC: {strategy_name} - {method_name} (GT: {gt_type})')
            plt.legend(loc="lower right")
            plt.grid(True, alpha=0.3)
            roc_path = os.path.join(output_dir, f"{method_name}_roc.png")
            plt.savefig(roc_path)
            plt.close()
            print(f"  ROC Plot saved to {roc_path}")
            
    elif nuscenes_scores is not None:
        # Fallback if no GT loaded
        nuscenes_normal_scores = list(nuscenes_scores)

    # 3. Plotting Distribution
    plot_path = os.path.join(output_dir, f"{method_name}_distribution.png")
    
    plt.figure(figsize=(12, 7))
    bins = 50
    hist_type = 'step'
    line_width = 2
    
    if nvidia_scores is not None:
        plt.hist(nvidia_scores, bins=bins, density=True, histtype=hist_type, linewidth=line_width,
                 label=f'NVIDIA Demo (N={len(nvidia_scores)})', color='orange')
        
    if nuscenes_normal_scores:
        plt.hist(nuscenes_normal_scores, bins=bins, density=True, histtype=hist_type, linewidth=line_width,
                 label=f'nuScenes Normal (N={len(nuscenes_normal_scores)})', color='green')
        
    if nuscenes_gt_scores:
        plt.hist(nuscenes_gt_scores, bins=bins, density=True, histtype=hist_type, linewidth=line_width,
                 label=f'nuScenes GT (N={len(nuscenes_gt_scores)})', color='red')

    plt.title(f"Outlier Scores: {strategy_name} - {method_name} (GT: {gt_type}, N={len(nuscenes_normal_scores) + len(nuscenes_gt_scores)})\n(Scores computed within each dataset context)")
    plt.xlabel(f"{method_name} Score (Higher = More Outlier)")
    plt.ylabel("Density")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.savefig(plot_path)
    plt.close()
    print(f"Distribution Plot saved to {plot_path}")

    # 4. Save Scores to JSON
    scores_data = {}
    
    if nuscenes_scores is not None:
        for i, uid in enumerate(nuscenes_ids):
            scores_data[uid] = {
                "score": float(nuscenes_scores[i]),
                "dataset": "nuscenes_ego"
            }
            
    if nvidia_scores is not None:
        for i, uid in enumerate(nvidia_ids):
            scores_data[uid] = {
                "score": float(nvidia_scores[i]),
                "dataset": "nvidia_demo"
            }
            
    scores_out_path = os.path.join(OUTLIERS_DIR, f"{strategy_name}_{method_name}_scores.json")
    os.makedirs(os.path.dirname(scores_out_path), exist_ok=True)
    
    with open(scores_out_path, "w") as f:
        json.dump(scores_data, f, indent=2)
    print(f"Saved scores to {scores_out_path}")
    
    # 5. Save Metrics to JSON
    if metrics:
        metrics_out_path = os.path.join(OUTLIERS_DIR, f"{strategy_name}_{method_name}_metrics.json")
        with open(metrics_out_path, "w") as f:
            json.dump(metrics, f, indent=2)
        print(f"Saved metrics to {metrics_out_path}")


def main():
    parser = argparse.ArgumentParser(description="Analyze Comparative Embedding Outliers")
    parser.add_argument("--strategy", type=str, required=True, help="Strategy name (e.g., fastvit_attention)")
    parser.add_argument("--k", type=int, default=20, help="Number of neighbors for LOF (default, can be overridden by loop)")
    parser.add_argument("--gt_type", type=str, default="surprise", choices=["surprise", "asil_ge_A", "asil_ge_B"], help="Ground truth outliers to use")
    args = parser.parse_args()

    # Methods to run
    # For LOF, we run multiple k values
    lof_k_values = [3, 5, 10, 20]
    
    # Generate method/k pairs
    methods_config = []
    
    # LOF Variations
    for k in lof_k_values:
        methods_config.append(("lof", f"lof_k{k}", k))
        
    # Isolation Forest & SVM (Fixed k doesn't matter, pass default)
    methods_config.append(("isolation_forest", "isolation_forest", args.k))
    methods_config.append(("one_class_svm", "one_class_svm", args.k))
    
    # MLP (Supervised)
    methods_config.append(("mlp", "mlp", args.k))
    
    # Output Dir Specific to Strategy
    strategy_output_dir = os.path.join(RESULTS_DIR, args.strategy)
    os.makedirs(strategy_output_dir, exist_ok=True)
    print(f"Results will be saved to: {strategy_output_dir}")

    # 1. Load Data Once
    print("Loading data...")
    try:
        nuscenes_data = load_embeddings(args.strategy, "nuscenes_ego")
        nvidia_data = load_embeddings(args.strategy, "nvidia_demo")
        gt_outliers = load_ground_truth(args.gt_type)
    except Exception as e:
        print(f"Error loading data: {e}")
        return

    if nuscenes_data[0] is None and nvidia_data[0] is None:
        print("No embeddings found for any dataset.")
        return

    # 2. Loop Through Configs
    for method_type, method_name, k_val in methods_config:
        try:
            # We pass 'method_name' which includes the k suffix if applicable for filenames
            # But inside for calculation, we might need to parse it back or pass method_type.
            # actually run_analysis_for_method calls compute_outlier_scores with method_name.
            # compute_outlier_scores handles 'lof' but not 'lof_k20'.
            # So inside run_analysis_for_method, we need to handle this.
            # I updated run_analysis_for_method to split('_k')[0].
            run_analysis_for_method(args.strategy, method_name, k_val, nuscenes_data, nvidia_data, gt_outliers, strategy_output_dir, args.gt_type)
        except Exception as e:
            print(f"Failed to run analysis for {method_name}: {e}")

if __name__ == "__main__":
    main()
