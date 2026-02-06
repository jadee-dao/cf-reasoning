import argparse
import os
import glob
import pickle
import json
import numpy as np
import sys

# Optional optimization
try:
    from sklearn.metrics.pairwise import cosine_similarity
except ImportError:
    print("Error: sklearn is required for similarity calculation.")
    sys.exit(1)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_ROOT = os.path.join(BASE_DIR, "../../analysis_results")

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NumpyEncoder, self).default(obj)

def main():
    parser = argparse.ArgumentParser(description="Compute Cross-Dataset Similarities")
    parser.add_argument("--strategy", type=str, required=True, help="Strategy name")
    args = parser.parse_args()

    emb_dir = os.path.join(OUTPUT_ROOT, "embeddings", args.strategy)
    sim_dir = os.path.join(OUTPUT_ROOT, "similarities", args.strategy)
    os.makedirs(sim_dir, exist_ok=True)

    if not os.path.exists(emb_dir):
        print(f"Error: Embeddings directory not found: {emb_dir}")
        return

    # 1. Load All Embeddings
    embedding_files = glob.glob(os.path.join(emb_dir, "*.pkl"))
    if not embedding_files:
        print("No embedding files found.")
        return

    all_ids = []
    all_embeddings = []
    dataset_map = {} # id -> dataset_name

    print(f"Found {len(embedding_files)} datasets. Loading...")
    
    datasets_found = []

    for fpath in embedding_files:
        dataset_name = os.path.splitext(os.path.basename(fpath))[0]
        datasets_found.append(dataset_name)
        with open(fpath, "rb") as f:
            data = pickle.load(f)
            
        ids = data["ids"]
        emb = data["embeddings"]
        
        all_ids.extend(ids)
        if len(all_embeddings) == 0:
            all_embeddings = emb
        else:
            all_embeddings = np.concatenate((all_embeddings, emb), axis=0)
            
        for i_id in ids:
            dataset_map[i_id] = dataset_name
            
    print(f"Total samples loaded: {len(all_ids)}")

    # 2. Compute Global Similarity Matrix
    print("Computing global cosine similarity matrix...")
    sim_matrix = cosine_similarity(all_embeddings)

    # 3. Extract Top-K Pairs
    # For Viewer compatibility, we want to save per-dataset similarity files.
    # Each file should contain pairs relevant to that dataset.
    # Structure: [ {"pair": [idA, idB], "score": 0.9}, ... ]
    # A pair (A, B) is "relevant" to Dataset D if A belongs to D (or B belongs to D).
    # Since specific order doesn't matter for "similarity", we can store it in both?
    # Actually, Viewer lists "Most Similar" by iterating this list.
    # If I am viewing Dataset A, I want to see pairs involving A.
    
    # Let's verify existing logic in Viewer or previous run_strategy:
    # "unique_pairs" was used.
    # If we split by dataset, we might duplicate pairs (A from D1, B from D2) -> stored in D1.json AND D2.json
    # This is fine. The Viewer loads ONE dataset configuration.
    
    per_dataset_pairs = {d: [] for d in datasets_found}
    
    TOP_K = 50
    
    # Iterate row by row (faster than creating huge unique_pairs dict first?)
    # Matrix is symmetric.
    
    print("Extracting top pairs...")
    
    # Using a global set to avoid processing same pair twice?
    # Actually, for the per-dataset files, we WANT duplicates if they cross boundaries.
    # (A, B) goes to A's list. (B, A) goes to B's list?
    # Viewer uses: for item in similarities: render(item)
    # So if A is current dataset, we want (A, B) pairs.
    
    for i in range(len(all_ids)):
        row = sim_matrix[i]
        id_i = all_ids[i]
        dataset_i = dataset_map[id_i]
        
        # Sort row
        sorted_indices = np.argsort(row)[::-1]
        
        # Identify indices of interest (Top K and Bottom K)
        # Handle overlap if dataset is small
        n_samples = len(sorted_indices)
        if n_samples <= 2 * TOP_K:
            indices_of_interest = sorted_indices
        else:
            indices_of_interest = np.concatenate((sorted_indices[:TOP_K+1], sorted_indices[-TOP_K:]))
            
        indices_of_interest = np.unique(indices_of_interest)

        for idx in indices_of_interest:
            if idx == i:
                continue
                
            id_j = all_ids[idx]
            score = float(row[idx])
            
            per_dataset_pairs[dataset_i].append({
                "pair": [id_i, id_j],
                "score": score
            })

    # 4. Save results
    for dname, pairs in per_dataset_pairs.items():
        out_json = os.path.join(sim_dir, f"{dname}.json")
        # Sort locally just in case
        pairs.sort(key=lambda x: x["score"], reverse=True)
        
        with open(out_json, "w") as f:
            json.dump(pairs, f, indent=2, cls=NumpyEncoder)
        print(f"Saved {len(pairs)} pairs to {out_json}")

if __name__ == "__main__":
    main()
