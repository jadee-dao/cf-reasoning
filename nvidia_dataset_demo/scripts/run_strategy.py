import argparse
import glob
import os
import json
import pickle
import numpy as np
import cv2
import sys
from tqdm import tqdm

# Ensure we can import embeddings
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.append(BASE_DIR)

from embeddings.strategies import (
    NaiveStrategy, 
    ForegroundStrictStrategy, 
    ForegroundLooseStrategy,
    VideoMAEStrategy,
    ObjectSemanticsStrategy,
    FastViTAttentionStrategy,
    ViViTStrategy,
    OpenRouterDescriptionStrategy,
    OpenRouterHazardStrategy,
    OpenRouterStoryboardStrategy
)

from dotenv import load_dotenv
load_dotenv()

DATA_ROOT = os.path.join(BASE_DIR, "../extracted_data")
OUTPUT_ROOT = os.path.join(BASE_DIR, "analysis_results")

STRATEGIES = {
    "naive": NaiveStrategy,
    "foreground_strict": ForegroundStrictStrategy,
    "foreground_loose": ForegroundLooseStrategy,
    "video_mae": VideoMAEStrategy,
    "object_semantics": ObjectSemanticsStrategy,
    "fastvit_attention": FastViTAttentionStrategy,
    "video_vit": ViViTStrategy,
    "openrouter_description": OpenRouterDescriptionStrategy,
    "openrouter_hazard": OpenRouterHazardStrategy,
    "openrouter_storyboard": OpenRouterStoryboardStrategy
}

def load_processed_samples(dataset_name, limit=None):
    """Loads processed samples from extracted_data/{dataset}/samples"""
    samples_dir = os.path.join(DATA_ROOT, dataset_name, "samples")
    if not os.path.exists(samples_dir):
        print(f"Error: Samples directory not found: {samples_dir}")
        return []
        
    pattern = os.path.join(samples_dir, "*.mp4") # Or .jpg?
    # We need both image and video usually.
    # We iterating over IDs.
    
    # Implementation: Find all .jpg files (easier to count)
    files = glob.glob(os.path.join(samples_dir, "*.jpg"))
    files.sort()
    
    if limit:
        files = files[:limit]
        
    samples = []
    for img_path in files:
        # ID is filename without extension
        filename = os.path.basename(img_path)
        sample_id = os.path.splitext(filename)[0]
        video_path = os.path.join(samples_dir, f"{sample_id}.mp4")
        
        samples.append({
            "id": sample_id,
            "image_path": img_path,
            "video_path": video_path if os.path.exists(video_path) else None
        })
        
    return samples

def main():
    parser = argparse.ArgumentParser(description="Run Embedding Strategy")
    parser.add_argument("--strategy", type=str, required=True, choices=STRATEGIES.keys(), help="Embedding strategy")
    parser.add_argument("--dataset", type=str, default="nvidia_demo", help="Dataset name")
    parser.add_argument("--limit", type=int, default=None, help="Limit samples")
    parser.add_argument("--force", action="store_true", help="Force re-generation of embeddings")
    args = parser.parse_args()
    
    # 1. Paths
    viz_dir = os.path.join(OUTPUT_ROOT, "visualize_samples", args.strategy, args.dataset)
    emb_dir = os.path.join(OUTPUT_ROOT, "embeddings", args.strategy)
    sim_dir = os.path.join(OUTPUT_ROOT, "similarities", args.strategy)
    
    os.makedirs(viz_dir, exist_ok=True)
    os.makedirs(emb_dir, exist_ok=True)
    os.makedirs(sim_dir, exist_ok=True)
    
    # 2. Load Samples
    print(f"Loading samples for {args.dataset}...")
    samples = load_processed_samples(args.dataset, limit=args.limit)
    print(f"Loaded {len(samples)} samples.")
    
    if not samples:
        return

    # Check if embeddings already exist
    out_pkl = os.path.join(emb_dir, f"{args.dataset}.pkl")
    if os.path.exists(out_pkl) and not args.force:
        print(f"Embeddings already exist at {out_pkl}. Use --force to overwrite. Skipping.")
        return

    # 3. Initialize Strategy
    print(f"Initializing {args.strategy}...")
    StrategyClass = STRATEGIES[args.strategy]
    strategy = StrategyClass()
    # Ensure model loaded? Some strategies do lazy load.
    if hasattr(strategy, 'load_model'):
        strategy.load_model()

    embeddings = []
    ids = []
    
    # 4. Process
    pbar = tqdm(samples, desc="Generating Embeddings")
    for s in pbar:
        pbar.set_description(f"Processing {s['id']}")
        
        # Debug Output Path
        debug_img_path = os.path.join(viz_dir, f"{s['id']}.jpg")
        
        # Resume Logic (Check if debug text exists?)
        # Simplification: Just check if we want to skip? 
        # For now, let's run it.
        
        try:
            emb = strategy.generate_embedding(
                s["image_path"],
                debug_output_path=debug_img_path,
                video_path=s["video_path"]
            )
            embeddings.append(emb)
            ids.append(s["id"])
        except Exception as e:
            print(f"Error processing {s['id']}: {e}")
            # Append zero vector or skip?
            # Append None and filter later?
            # Let's append None
            embeddings.append(None)
            ids.append(s['id'])

    # Filter failures
    valid_indices = [i for i, e in enumerate(embeddings) if e is not None]
    embeddings = np.array([embeddings[i] for i in valid_indices])
    ids = [ids[i] for i in valid_indices]
    
    if len(embeddings) == 0:
        print("No valid embeddings generated.")
        return

    # 5. Save Embeddings Dictionary
    out_pkl = os.path.join(emb_dir, f"{args.dataset}.pkl")
    data = {
        "ids": ids,
        "embeddings": embeddings
    }
    with open(out_pkl, "wb") as f:
        pickle.dump(data, f)
    print(f"Saved embeddings to {out_pkl}")
    
    print("Embedding generation complete. Run compute_similarities.py to calculate matches.")

if __name__ == "__main__":
    main()
