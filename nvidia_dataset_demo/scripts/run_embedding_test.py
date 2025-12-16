import argparse
import glob
import os
import json
import numpy as np
import matplotlib.pyplot as plt
import cv2
from sklearn.metrics.pairwise import cosine_similarity
from embeddings.strategies import NaiveStrategy, ForegroundStrictStrategy, ForegroundLooseStrategy, TextDescriptionStrategy, VLMCaptionStrategy, VideoXClipStrategy, ObjectSemanticsStrategy

# --- Configuration ---
DATA_DIR = "../extracted_data"
OUTPUT_DIR = "analysis_results"

STRATEGIES = {
    "naive": NaiveStrategy,
    "foreground_strict": ForegroundStrictStrategy,
    "foreground_loose": ForegroundLooseStrategy,
    "text": TextDescriptionStrategy,
    "vlm": VLMCaptionStrategy,
    "video": VideoXClipStrategy,
    "object_semantics": ObjectSemanticsStrategy
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
