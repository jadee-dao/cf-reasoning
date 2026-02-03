import os
import json
import argparse
import numpy as np

def main():
    parser = argparse.ArgumentParser(description="Filter calibration log to keep only the worst performing scenes (top percentile).")
    parser.add_argument("--input", type=str, required=True, help="Path to input JSON file.")
    parser.add_argument("--output", type=str, required=True, help="Path to output JSON file.")
    parser.add_argument("--percentile", type=float, default=10.0, help="Top percentile to keep (e.g. 10 for top 10%%).")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.input):
        print(f"Error: Input file {args.input} does not exist.")
        return

    print(f"Loading {args.input}...")
    with open(args.input, 'r') as f:
        data = json.load(f)
        
    if "results" not in data:
        print("Error: JSON must contain a 'results' key.")
        return
        
    scenes = data["results"]
    print(f"Found {len(scenes)} scenes.")
    
    # Collect all frame scores
    all_frame_scores = []
    for key, val in scenes.items():
        if "top3_worst" in val and isinstance(val["top3_worst"], list):
            for frame in val["top3_worst"]:
                ade = frame.get("ade_xy", 0.0)
                all_frame_scores.append(ade)
    
    # Sort scores descending
    all_frame_scores.sort(reverse=True)
    
    total_frames = len(all_frame_scores)
    if total_frames == 0:
        print("No frames found in input data.")
        return

    # Calculate cutoff
    keep_count = int(np.ceil(total_frames * (args.percentile / 100.0)))
    keep_count = max(1, keep_count) # Keep at least one
    
    cutoff_score = all_frame_scores[keep_count - 1]
    print(f"Keeping top {args.percentile}% frames: {keep_count} out of {total_frames}")
    print(f"Score cutoff: >= {cutoff_score:.4f} (Max: {all_frame_scores[0]:.4f})")
    
    # Filter scenes and frames
    new_results = {}
    kept_frame_count = 0
    
    for key, val in scenes.items():
        if "top3_worst" not in val or not isinstance(val["top3_worst"], list):
            continue
            
        kept_frames = []
        for frame in val["top3_worst"]:
            if frame.get("ade_xy", 0.0) >= cutoff_score:
                kept_frames.append(frame)
        
        if kept_frames:
            new_entry = val.copy()
            new_entry["top3_worst"] = kept_frames
            new_results[key] = new_entry
            kept_frame_count += len(kept_frames)
            
    print(f"Retained {len(new_results)} scenes containing {kept_frame_count} frames.")
    
    new_data = data.copy()
    new_data["results"] = new_results
    new_data["num_scenes"] = len(new_results)
    
    print(f"Saving to {args.output}...")
    with open(args.output, 'w') as f:
        json.dump(new_data, f, indent=2)

    # --- Plotting ---
    import matplotlib.pyplot as plt
    
    all_scores = all_frame_scores
    
    plt.figure(figsize=(10, 6))
    plt.hist(all_scores, bins=50, color='skyblue', edgecolor='black', alpha=0.7, label='ADE Scores (Frames)')
    
    if kept_frame_count > 0:
        plt.axvline(cutoff_score, color='red', linestyle='dashed', linewidth=2, label=f'Cutoff (Top {args.percentile}%: {cutoff_score:.4f})')
    
    plt.title('ADE Score Distribution')
    plt.xlabel('ADE Score')
    plt.ylabel('Frequency')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Save plot to the same directory as output, but with .png extension
    output_dir = os.path.dirname(os.path.abspath(args.output))
    base_name = os.path.basename(args.output).replace('.json', '')
    plot_path = os.path.join(output_dir, f"{base_name}_distribution.png")
    
    plt.savefig(plot_path)
    print(f"Distribution plot saved to {plot_path}")
        
    print("Done.")

if __name__ == "__main__":
    main()
