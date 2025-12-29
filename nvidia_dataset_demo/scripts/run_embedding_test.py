import argparse
import glob
import os
import json
import numpy as np
import matplotlib.pyplot as plt
import cv2
from sklearn.metrics.pairwise import cosine_similarity
from embeddings.strategies import (
    NaiveStrategy, 
    ForegroundStrictStrategy, 
    ForegroundLooseStrategy,
    TextDescriptionStrategy,
    VLMCaptionStrategy,
    VideoXClipStrategy,
    ObjectSemanticsStrategy,
    FastViTAttentionStrategy,
    FastVLMDescriptionStrategy,
    FastVLMHazardStrategy
)

# --- Configuration ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "../extracted_data")
OUTPUT_DIR = os.path.join(BASE_DIR, "analysis_results")

STRATEGIES = {
    "naive": NaiveStrategy,
    "foreground_strict": ForegroundStrictStrategy,
    "foreground_loose": ForegroundLooseStrategy,
    "text": TextDescriptionStrategy,
    "vlm": VLMCaptionStrategy,
    "video": VideoXClipStrategy,
    "object_semantics": ObjectSemanticsStrategy,
    "fastvit_attention": FastViTAttentionStrategy,
    "fastvlm_description": FastVLMDescriptionStrategy,
    "fastvlm_hazard": FastVLMHazardStrategy
}

def load_samples(limit=None):
    """Finds all video folders and identifies the middle frame."""
    samples = []
    # Matching the folder structure: extracted_data/camera_front_wide_120fov/UUID.camera_front...
    # We look for the folder, then the video inside is usually standard.
    # Actually, based on previous scripts, the folders ARE the samples.
    # Inside each folder, there are frames if extracted, or we extract the middle frame.
    
    # Let's check how inspect_data.py yielded samples. 
    # It seems we have .mp4 files directly in the root of extracted_data? 
    # Or folders? 
    # Let's rely on finding .mp4 files.
    
    mp4_files = glob.glob(os.path.join(DATA_DIR, "**/*.mp4"), recursive=True)
    mp4_files.sort()
    
    if limit:
        mp4_files = mp4_files[:limit]
        
    for video_path in mp4_files:
        # We need to extract the middle frame to a temp location for the strategy to read
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        mid_frame_idx = total_frames // 2
        cap.set(cv2.CAP_PROP_POS_FRAMES, mid_frame_idx)
        ret, frame = cap.read()
        cap.release()
        
        if ret:
            # Save frame temporarily
            filename = os.path.basename(video_path)
            sample_id = filename.split('.')[0] # Basic ID
            temp_img_path = f"temp_frame_{sample_id}.jpg"
            cv2.imwrite(temp_img_path, frame)
            
            samples.append({
                "id": sample_id,
                "video_path": video_path,
                "image_path": temp_img_path
            })
            
    return samples

def clean_temp_files(samples):
    for s in samples:
        if os.path.exists(s["image_path"]):
            os.remove(s["image_path"])

def main():
    parser = argparse.ArgumentParser(description="Run Embedding Analysis")
    parser.add_argument("--strategy", type=str, required=True, choices=STRATEGIES.keys(), help="Embedding strategy to use")
    parser.add_argument("--limit", type=int, default=10, help="Number of samples to process (default 10)")
    args = parser.parse_args()
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    print(f"--- Running Analysis with Strategy: {args.strategy.upper()} ---")
    
    # 1. Initialize Strategy
    StrategyClass = STRATEGIES[args.strategy]
    strategy_instance = StrategyClass()
    
    # 2. Load Data
    print(f"Loading {args.limit} samples...")
    samples = load_samples(args.limit)
    print(f"Loaded {len(samples)} samples.")
    
    # 3. Generate Embeddings
    embeddings = []
    ids = []
    
    # Create debug images directory
    debug_dir = os.path.join(OUTPUT_DIR, "debug_images", args.strategy)
    os.makedirs(debug_dir, exist_ok=True)
    
    print(f"Generating embeddings and saving debug images to {debug_dir}...")
    for i, s in enumerate(samples):
        print(f"[{i+1}/{len(samples)}] Processing {s['id']}...")
        
        debug_output_path = os.path.join(debug_dir, f"{s['id']}.jpg")
        emb = strategy_instance.generate_embedding(
            s["image_path"], 
            debug_output_path=debug_output_path,
            video_path=s["video_path"] # Pass video path for Video Strategy
        )
        
        embeddings.append(emb)
        ids.append(s["id"])
        
    embeddings = np.array(embeddings)
    
    # 4. Compute Similarity Matrix
    print("Computing Cosine Similarity...")
    sim_matrix = cosine_similarity(embeddings)

    # 4.5. Dimensionality Reduction & Export
    import pandas as pd
    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE
    try:
        import umap
        HAS_UMAP = True
    except ImportError:
        HAS_UMAP = False
        print("UMAP not found, skipping UMAP reduction.")

    n_samples = len(embeddings)
    
    # --- 2D Projections (For Viewer) ---
    print("Computing 2D Projections for Viewer...")
    projections_2d = {}
    
    # PCA-2D
    if n_samples >= 2:
        pca_2d = PCA(n_components=2)
        projections_2d['pca'] = pca_2d.fit_transform(embeddings)
    else:
        projections_2d['pca'] = np.zeros((n_samples, 2))

    # t-SNE-2D
    if n_samples > 1:
        perp = min(30, n_samples - 1)
        tsne_2d = TSNE(n_components=2, perplexity=perp, random_state=42, init='random', learning_rate='auto')
        projections_2d['tsne'] = tsne_2d.fit_transform(embeddings)
    else:
        projections_2d['tsne'] = np.zeros((n_samples, 2))

    # UMAP-2D
    if HAS_UMAP and n_samples > 1:
        # n_neighbors must be < n_samples
        n_neigh = min(15, n_samples - 1)
        if n_neigh < 2: n_neigh = 2
        umap_2d = umap.UMAP(n_components=2, n_neighbors=n_neigh, random_state=42)
        projections_2d['umap'] = umap_2d.fit_transform(embeddings)
    else:
        projections_2d['umap'] = np.zeros((n_samples, 2))

    # --- High-Dim Projections (For CSV) ---
    print("Computing High-Dim Projections (5D) for CSV...")
    projections_hd = {} # High Dim
    
    # PCA-5D
    n_comp_pca = min(5, n_samples)
    if n_comp_pca > 0:
        pca_hd = PCA(n_components=n_comp_pca)
        projections_hd['pca'] = pca_hd.fit_transform(embeddings)
    else:
        projections_hd['pca'] = np.zeros((n_samples, 5))

    # t-SNE-3D (Max for Barnes-Hut is 3)
    # If user really wants 5, we'd need method='exact' which is slow, but let's stick to 3 for stability/speed.
    n_comp_tsne = min(3, n_samples)
    if n_samples > 1:
        perp = min(30, n_samples - 1)
        tsne_hd = TSNE(n_components=n_comp_tsne, perplexity=perp, random_state=42, init='random', learning_rate='auto')
        projections_hd['tsne'] = tsne_hd.fit_transform(embeddings)
    else:
        projections_hd['tsne'] = np.zeros((n_samples, 3))

    # UMAP-5D
    if HAS_UMAP and n_samples > 1:
        n_neigh = min(15, n_samples - 1)
        if n_neigh < 2: n_neigh = 2
        umap_hd = umap.UMAP(n_components=5, n_neighbors=n_neigh, random_state=42)
        projections_hd['umap'] = umap_hd.fit_transform(embeddings)
    else:
        projections_hd['umap'] = np.zeros((n_samples, 5))

    # --- Prepare Data Structures ---
    
    # 1. JSON Points (for Viewer) - Include all 2D variants
    points = []
    for i, id_val in enumerate(ids):
        pt = {"id": id_val}
        # Add 2D coords
        pt["pca"] = projections_2d['pca'][i].tolist()
        pt["tsne"] = projections_2d['tsne'][i].tolist()
        pt["umap"] = projections_2d['umap'][i].tolist() if HAS_UMAP else [0, 0]
        
        # Default x/y to t-SNE for backward compat? Or make viewer smart.
        # Let's verify 'tsne' exists
        pt["x"] = pt["tsne"][0]
        pt["y"] = pt["tsne"][1]
        
        points.append(pt)

    # 2. DataFrame (for CSV) - Include High-Dim variants
    df_data = {"id": ids}
    
    # Add PCA-5D
    for dim in range(projections_hd['pca'].shape[1]):
        df_data[f"pca_{dim+1}"] = projections_hd['pca'][:, dim]
        
    # Add t-SNE-3D
    for dim in range(projections_hd['tsne'].shape[1]):
        df_data[f"tsne_{dim+1}"] = projections_hd['tsne'][:, dim]
        
    # Add UMAP-5D
    if HAS_UMAP:
        for dim in range(projections_hd['umap'].shape[1]):
            df_data[f"umap_{dim+1}"] = projections_hd['umap'][:, dim]
            
    # Save CSV
    csv_file = os.path.join(OUTPUT_DIR, f"projections_{args.strategy}.csv")
    pd.DataFrame(df_data).to_csv(csv_file, index=False)
    print(f"Projections CSV saved to {csv_file}")

    
    # 5. Find Extremes (Most/Least Similar Pairs)
    # Mask diagonal
    np.fill_diagonal(sim_matrix, -1) # specific low value to avoid picking diagonal
    
    # Flatten to sort
    flat_indices = np.argsort(sim_matrix.flatten())
    # Top 5 (End of the sorted list)
    top_indices = flat_indices[-5:][::-1] 
    # Bottom 5 (Start of the sorted list, ignoring -1s if possible, but here lowest real similarity)
    # We should filter out the -1 diagonal entries first for correctness
    
    pairs = []
    rows, cols = sim_matrix.shape
    for r in range(rows):
        for c in range(r + 1, cols): # Upper triangle only
            pairs.append(((r, c), sim_matrix[r, c]))
            
    pairs.sort(key=lambda x: x[1], reverse=True)
    
    # Store all pairs for the viewer to handle slicing and global ranking
    results = {
        "strategy": args.strategy,
        "total_pairs": len(pairs),
        "points": points,
        "all_pairs": [{"pair": [ids[p[0][0]], ids[p[0][1]]], "score": float(p[1])} for p in pairs]
    }
    
    # Save Results
    out_file = os.path.join(OUTPUT_DIR, f"results_{args.strategy}.json")
    with open(out_file, "w") as f:
        json.dump(results, f, indent=4)
    print(f"Results saved to {out_file} (Total Pairs: {len(pairs)})")
    
    # Clean up
    clean_temp_files(samples)
    
    # Visualize Top 1 Pair
    if pairs:
        best_pair_idx = pairs[0][0]
        id1, id2 = ids[best_pair_idx[0]], ids[best_pair_idx[1]]
        score = pairs[0][1]
        print(f"Most Similar Pair: {id1} vs {id2} (Score: {score:.4f})")

if __name__ == "__main__":
    main()
