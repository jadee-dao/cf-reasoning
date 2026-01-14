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
    FastVLMHazardStrategy,
    OpenRouterDescriptionStrategy,
    FastVLMHazardStrategy,
    OpenRouterDescriptionStrategy,
    OpenRouterHazardStrategy,
    OpenRouterStoryboardStrategy
)
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), '.env'))
# Also try a default load in case it's elsewhere
load_dotenv()

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
    "fastvlm_hazard": FastVLMHazardStrategy,
    "openrouter_description": OpenRouterDescriptionStrategy,
    "openrouter_hazard": OpenRouterHazardStrategy,
    "openrouter_storyboard": OpenRouterStoryboardStrategy
}

def load_samples(limit=None, interval_sec=1.5):
    """Finds all video folders and identifies frames at fixed intervals."""
    import subprocess
    samples = []
    
    # Create persistent base dir for this interval
    interval_dir_name = f"{interval_sec}s"
    persistent_base_dir = os.path.join(DATA_DIR, "interval_samples", interval_dir_name)
    os.makedirs(persistent_base_dir, exist_ok=True)
    
    mp4_files = glob.glob(os.path.join(DATA_DIR, "**/*.mp4"), recursive=True)
    mp4_files.sort()
    
    if limit:
        mp4_files = mp4_files[:limit]
        
    for video_path in mp4_files:
        filename = os.path.basename(video_path)
        # Extract UUID (everything before the first dot)
        uuid = filename.split('.')[0]
        
        # Create persistent dir for this video
        sample_dir = os.path.join(persistent_base_dir, uuid)
        os.makedirs(sample_dir, exist_ok=True)
        
        # Get video duration using cv2 (simple way)
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Warning: Could not open {video_path}")
            continue
            
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration_sec = total_frames / fps if fps > 0 else 0
        cap.release()
        
        if duration_sec <= 0:
            print(f"Warning: Invalid duration for {video_path}")
            continue

        # Iterate by time intervals
        current_time = 0.0
        while current_time < duration_sec:
            ts_us = int(current_time * 1_000_000)
            sample_id = f"{uuid}_{ts_us}"
            
            sub_video_path = os.path.join(sample_dir, f"{ts_us}.mp4")
            img_path = os.path.join(sample_dir, f"{ts_us}.jpg")
            
            # Check if files exist
            if not os.path.exists(sub_video_path) or not os.path.exists(img_path):
                # FFmpeg command to extract sub-video and frame
                # -ss before -i is faster processing
                try:
                    # Extract sub-video (-t duration, -c copy might fail if keyframes don't align, so re-encode is safer but slower. 
                    # For accuracy we re-encode or use fast splitting. Let's try re-encoding for exact cut.)
                    # Actually -c copy is bad for arbitrary cuts.
                    cmd_video = [
                        "ffmpeg", "-y", "-v", "error",
                        "-ss", str(current_time),
                        "-t", str(interval_sec),
                        "-i", video_path,
                        "-c:v", "libx264", "-c:a", "aac", # Re-encode to ensure playable subclip
                        sub_video_path
                    ]
                    subprocess.run(cmd_video, check=True)
                    
                    # Extract frame
                    cmd_frame = [
                        "ffmpeg", "-y", "-v", "error",
                        "-ss", str(current_time),
                        "-i", video_path,
                        "-frames:v", "1",
                        "-q:v", "2", # High quality jpg
                        img_path
                    ]
                    subprocess.run(cmd_frame, check=True)
                    
                except subprocess.CalledProcessError as e:
                    print(f"Error processing {sample_id}: {e}")
                    current_time += interval_sec
                    continue

            samples.append({
                "id": sample_id,
                "video_path": sub_video_path, # POINT TO SUB-VIDEO
                "image_path": img_path,
                "timestamp_us": ts_us,
                "uuid": uuid
            })
            
            current_time += interval_sec
            
    return samples

def clean_temp_files(samples):
    # Frames are now persistent in extracted_data/interval_samples, so we do NOT delete them.
    pass

def main():
    parser = argparse.ArgumentParser(description="Run Embedding Analysis")
    parser.add_argument("--strategy", type=str, required=True, choices=STRATEGIES.keys(), help="Embedding strategy to use")
    parser.add_argument("--limit", type=int, default=10, help="Number of samples to process (default 10)")
    parser.add_argument("--interval", type=float, default=1.5, help="Time interval in seconds for sampling (default 1.5)")
    args = parser.parse_args()
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    print(f"--- Running Analysis with Strategy: {args.strategy.upper()} ---")
    
    # 1. Initialize Strategy
    StrategyClass = STRATEGIES[args.strategy]
    strategy_instance = StrategyClass()
    
    # Determine configuration name for outputs
    config_name = strategy_instance.get_config_name() or args.strategy
    print(f"Using Output Config Name: {config_name}")
    
    # 2. Load Data
    print(f"Loading {args.limit} samples (Interval: {args.interval}s)...")
    samples = load_samples(limit=args.limit, interval_sec=args.interval)
    print(f"Loaded {len(samples)} samples.")
    
    # 3. Generate Embeddings
    embeddings = []
    ids = []
    
    # Create debug images directory
    debug_dir = os.path.join(OUTPUT_DIR, "debug_images", config_name)
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
    if HAS_UMAP and n_samples > 2:
        try:
            # n_neighbors must be < n_samples
            n_neigh = min(15, n_samples - 1)
            if n_neigh < 2: n_neigh = 2
            umap_2d = umap.UMAP(n_components=2, n_neighbors=n_neigh, random_state=42)
            projections_2d['umap'] = umap_2d.fit_transform(embeddings)
        except Exception as e:
            print(f"UMAP 2D failed: {e}")
            projections_2d['umap'] = np.zeros((n_samples, 2))
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
    if HAS_UMAP and n_samples > 2:
        try:
            n_neigh = min(15, n_samples - 1)
            if n_neigh < 2: n_neigh = 2
            umap_hd = umap.UMAP(n_components=5, n_neighbors=n_neigh, random_state=42)
            projections_hd['umap'] = umap_hd.fit_transform(embeddings)
        except Exception as e:
            print(f"UMAP 5D failed: {e}")
            projections_hd['umap'] = np.zeros((n_samples, 5))
    else:
        projections_hd['umap'] = np.zeros((n_samples, 5))

    # --- Clustering & Outlier Detection ---
    print("Computing Clusters and Outliers...")
    from sklearn.cluster import KMeans
    from sklearn.ensemble import IsolationForest
    
    # 1. Outlier Detection
    # Contamination='auto' is usually good
    iso = IsolationForest(contamination='auto', random_state=42)
    # -1 is outlier, 1 is inlier. We'll store boolean is_outlier
    outlier_preds = iso.fit_predict(embeddings)
    
    # 2. Clusters
    # We'll pre-compute a few k values for the viewer
    k_values = [3, 5, 8]
    cluster_labels = {}
    
    for k in k_values:
        if n_samples >= k:
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            cluster_labels[f"cluster_k{k}"] = kmeans.fit_predict(embeddings)
        else:
            cluster_labels[f"cluster_k{k}"] = np.zeros(n_samples)

    # --- Prepare Data Structures ---
    
    # 1. JSON Points (for Viewer) - Include all 2D variants + Clusters
    points = []
    for i, id_val in enumerate(ids):
        pt = {"id": id_val}
        # Add 2D coords
        pt["pca"] = projections_2d['pca'][i].tolist()
        pt["tsne"] = projections_2d['tsne'][i].tolist()
        pt["umap"] = projections_2d['umap'][i].tolist() if HAS_UMAP else [0, 0]
        
        # Default x/y
        pt["x"] = pt["tsne"][0]
        pt["y"] = pt["tsne"][1]
        
        # Add Analysis Tags
        pt["is_outlier"] = bool(outlier_preds[i] == -1)
        for k in k_values:
             pt[f"cluster_k{k}"] = int(cluster_labels[f"cluster_k{k}"][i])
        
        points.append(pt)

    # 2. DataFrame (for CSV) - Include High-Dim variants + Analysis
    df_data = {"id": ids}
    
    # Analysis columns
    df_data["is_outlier"] = outlier_preds
    for k in k_values:
        df_data[f"cluster_k{k}"] = cluster_labels[f"cluster_k{k}"]
    
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
    csv_file = os.path.join(OUTPUT_DIR, f"projections_{config_name}.csv")
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
        "strategy": config_name,
        "total_pairs": len(pairs),
        "points": points,
        "all_pairs": [{"pair": [ids[p[0][0]], ids[p[0][1]]], "score": float(p[1])} for p in pairs]
    }
    
    # Save Results
    out_file = os.path.join(OUTPUT_DIR, f"results_{config_name}.json")
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
