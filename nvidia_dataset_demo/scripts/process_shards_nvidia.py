import argparse
import os
import sys

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../src/processing")))
from dataset_processors import NvidiaDemoProcessor

def main():
    parser = argparse.ArgumentParser(description="Process specific Nvidia calibration shards.")
    parser.add_argument("--shards", type=int, nargs="+", default=[96, 97, 98], help="Shard indices to process.")
    parser.add_argument("--limit", type=int, default=None, help="Limit samples per shard.")
    parser.add_argument("--workers", type=int, default=8, help="Number of parallel workers.")
    parser.add_argument("--no-video", action="store_true", help="Only extract frames, skip video clips.")
    parser.add_argument("--output-dir", type=str, default=None, help="Custom output directory for samples.")
    
    args = parser.parse_args()
    
    # Calibration shard path convention
    # /home/jadelynn/cf-reasoning/nvidia_dataset_demo/extracted_data/nvidia_demo/calibration/shard_000XX.jsonl
    base_calib = os.path.abspath(os.path.join(os.path.dirname(__file__), "../extracted_data/nvidia_demo/calibration"))
    
    shard_paths = []
    for s in args.shards:
        path = os.path.join(base_calib, f"shard_{s:05d}.jsonl")
        shard_paths.append(path)
        
    processor = NvidiaDemoProcessor()
    processor.process_shards(shard_paths, limit=args.limit, max_workers=args.workers, 
                            extract_video=not args.no_video, output_dir=args.output_dir)

if __name__ == "__main__":
    main()
