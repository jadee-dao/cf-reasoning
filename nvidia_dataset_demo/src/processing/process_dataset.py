import argparse
import os
import sys
from dataset_processors import PROCESSORS

def main():
    parser = argparse.ArgumentParser(description="Process a dataset into standardized samples.")
    parser.add_argument("--dataset_name", type=str, required=True, help="Name of the dataset (e.g., nvidia_demo)")
    parser.add_argument("--raw_path", type=str, default=None, help="Path to raw data. If provided, creates a symlink in extracted_data/raw_data.")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of raw videos to process")
    parser.add_argument("--interval", type=float, default=1.5, help="Sampling interval in seconds")
    
    args = parser.parse_args()
    
    if args.dataset_name not in PROCESSORS:
        print(f"Error: Unknown dataset '{args.dataset_name}'. Available: {list(PROCESSORS.keys())}")
        sys.exit(1)
        
    ProcessorClass = PROCESSORS[args.dataset_name]
    
    # Handle Raw Path Linkage
    base_extracted = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../extracted_data"))
    raw_dest = os.path.join(base_extracted, "raw_data", args.dataset_name)
    
    if args.raw_path:
        if not os.path.exists(args.raw_path):
            print(f"Error: Provided raw path does not exist: {args.raw_path}")
            sys.exit(1)
            
        # Create raw_data parent if needed
        os.makedirs(os.path.dirname(raw_dest), exist_ok=True)
        
        # Create symlink
        if os.path.islink(raw_dest):
            os.remove(raw_dest)
        elif os.path.exists(raw_dest) and len(os.listdir(raw_dest)) == 0:
            os.rmdir(raw_dest)
        
        if not os.path.exists(raw_dest):
            os.symlink(args.raw_path, raw_dest)
            print(f"Linked {raw_dest} -> {args.raw_path}")
        else:
            print(f"Warning: {raw_dest} already exists and is not a symlink/empty. Using existing data.")
    
    # Run Processor
    processor = ProcessorClass(dataset_name=args.dataset_name)
    processor.process(interval_sec=args.interval, limit=args.limit)

if __name__ == "__main__":
    main()
