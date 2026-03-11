import argparse
import glob
import os
import sys
from tqdm import tqdm

# Ensure we can import modules from src
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.join(BASE_DIR, "../src")
if SRC_DIR not in sys.path:
    sys.path.append(SRC_DIR)

from embeddings.strategies import ObjectGraphStrategy

def load_shard_samples(samples_root, shard_name):
    """Loads samples from a specific shard folder."""
    shard_dir = os.path.join(samples_root, shard_name)
    if not os.path.exists(shard_dir):
        print(f"Error: Shard directory not found: {shard_dir}")
        return []
        
    files = glob.glob(os.path.join(shard_dir, "*.jpg"))
    files.sort()
    
    samples = []
    for img_path in files:
        filename = os.path.basename(img_path)
        sample_id = os.path.splitext(filename)[0]
        video_path = img_path.replace(".jpg", ".mp4")
        
        samples.append({
            "id": sample_id,
            "image_path": img_path,
            "video_path": video_path if os.path.exists(video_path) else None,
            "shard": shard_name
        })
    return samples

def main():
    parser = argparse.ArgumentParser(description="Run Object Graph Strategy on Specific Shards")
    parser.add_argument("--shards", type=str, nargs="+", default=["shard_00096", "shard_00097", "shard_00098"], help="Shard folder names")
    parser.add_argument("--dataset", type=str, default="nvidia_demo", help="Dataset name")
    parser.add_argument("--force", action="store_true", help="Force re-generation")
    args = parser.parse_args()
    
    samples_root = os.path.abspath(os.path.join(BASE_DIR, f"../extracted_data/{args.dataset}/samples"))
    # Save directly to the dataset's object_graphs folder as requested
    output_root = os.path.abspath(os.path.join(BASE_DIR, f"../extracted_data/{args.dataset}/object_graphs"))
    
    print(f"Initializing ObjectGraphStrategy...")
    strategy = ObjectGraphStrategy()
    strategy.load_model()
    
    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    for shard_name in args.shards:
        print(f"\nProcessing {shard_name}...")
        samples = load_shard_samples(samples_root, shard_name)
        print(f"Loaded {len(samples)} samples from {shard_name}.")
        
        if not samples:
            continue
            
        shard_out_dir = os.path.join(output_root, shard_name)
        os.makedirs(shard_out_dir, exist_ok=True)
        
        pbar = tqdm(samples, desc=f"Shard {shard_name}")
        for s in pbar:
            graph_json_path = os.path.join(shard_out_dir, f"{s['id']}_graph.json")
            
            if os.path.exists(graph_json_path) and not args.force:
                continue
                
            try:
                # Save JUST the graph JSON for now
                strategy.generate_embedding(
                    s["image_path"],
                    graph_output_path=graph_json_path,
                    video_path=s["video_path"]
                )
            except Exception as e:
                # tqdm.write(f"Error processing {s['id']}: {e}")
                pass
                
    print("\nShard-based strategy processing complete.")

if __name__ == "__main__":
    main()
