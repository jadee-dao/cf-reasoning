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
    
    # Calculate score for each scene
    # We use the maximum 'ade_xy' from the 'top3_worst' frames as the scene score.
    scene_scores = []
    
    for key, val in scenes.items():
        max_ade = 0.0
        if "top3_worst" in val and isinstance(val["top3_worst"], list):
            for frame in val["top3_worst"]:
                ade = frame.get("ade_xy", 0.0)
                if ade > max_ade:
                    max_ade = ade
        
        scene_scores.append({
            "key": key,
            "data": val,
            "score": max_ade
        })
        
    # Sort by score descending (worst first)
    scene_scores.sort(key=lambda x: x["score"], reverse=True)
    
    # Calculate cutoff
    count = len(scene_scores)
    keep_count = int(np.ceil(count * (args.percentile / 100.0)))
    keep_count = max(1, keep_count) # Keep at least one
    
    print(f"Keeping top {args.percentile}% scenes: {keep_count} out of {count}")
    
    top_scenes = scene_scores[:keep_count]
    if top_scenes:
        cutoff_score = top_scenes[-1]["score"]
        print(f"Score cutoff: >= {cutoff_score:.4f} (Max: {top_scenes[0]['score']:.4f})")
    
    # Construct new data
    new_results = {item["key"]: item["data"] for item in top_scenes}
    
    new_data = data.copy()
    new_data["results"] = new_results
    new_data["num_scenes"] = len(new_results) # Update count if it exists
    
    print(f"Saving to {args.output}...")
    with open(args.output, 'w') as f:
        json.dump(new_data, f, indent=2)

    # --- Plotting ---
    import matplotlib.pyplot as plt
    
    all_scores = [x["score"] for x in scene_scores]
    
    plt.figure(figsize=(10, 6))
    plt.hist(all_scores, bins=50, color='skyblue', edgecolor='black', alpha=0.7, label='ADE Scores')
    
    if top_scenes:
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
