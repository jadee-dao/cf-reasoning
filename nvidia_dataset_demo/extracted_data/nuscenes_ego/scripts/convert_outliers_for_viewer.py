import numpy as np
import os
import json

# Constants
TRAIN_FILE = "/home/jadelynn/cf-reasoning/nvidia_dataset_demo/extracted_data/nuscenes_ego/calibration/nuscenes_train.npy"
VAL_FILE = "/home/jadelynn/cf-reasoning/nvidia_dataset_demo/extracted_data/nuscenes_ego/calibration/nuscenes_val.npy"
OUTPUT_FILE = "/home/jadelynn/cf-reasoning/nvidia_dataset_demo/extracted_data/nuscenes_ego/calibration/nuscenes_percentile_outliers.json"

def load_data(filepath):
    try:
        data = np.load(filepath, allow_pickle=True)
        if data.shape == ():
            data = data.item()
        return data
    except Exception as e:
        print(f"Error loading {filepath}: {e}")
        return None

def main():
    print("Loading NPY data...")
    train_data = load_data(TRAIN_FILE)
    val_data = load_data(VAL_FILE)

    if not train_data or not val_data:
        print("Failed to load data.")
        return

    # Aggregate global scores to find P99
    all_scores = []
    
    # We also need to iterate to build the map
    # Let's collect everything first
    items = []

    for d in [train_data, val_data]:
        scores = d['dist']
        idxs = d['idx']
        data_idxs = d['data_idx']
        
        # Validation: check lengths
        if not (len(scores) == len(idxs) == len(data_idxs)):
            print("Warning: Data arrays length mismatch.")
            continue
            
        all_scores.extend(scores)
        
        for i in range(len(scores)):
            items.append({
                "scene_id": idxs[i],
                "data_idx": int(data_idxs[i]),
                "score": float(scores[i])
            })

    all_scores = np.array(all_scores)
    p90 = np.percentile(all_scores, 90)
    p99 = np.percentile(all_scores, 99)
    print(f"Global P90 Score: {p90:.4f}")
    print(f"Global P99 Score: {p99:.4f}")

    # Build Output - Includes anything > P90
    outliers_map = {}
    count_90 = 0
    count_99 = 0
    
    for item in items:
        if item["score"] > p90:
            sample_id = f"{item['scene_id']}_{item['data_idx']}"
            
            is_p99 = bool(item["score"] > p99)
            
            outliers_map[sample_id] = {
                "score": item["score"],
                "scene_id": item["scene_id"],
                "data_idx": item["data_idx"],
                "is_p90": True,
                "is_p99": is_p99
            }
            count_90 += 1
            if is_p99:
                count_99 += 1
            
    print(f"Identified {count_90} global P90 outliers.")
    print(f"Identified {count_99} global P99 outliers.")
    
    output_data = {
        "global_p90": float(p90),
        "global_p99": float(p99),
        "generated_samples": outliers_map
    }
    
    # Ensure dir exists
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    
    with open(OUTPUT_FILE, "w") as f:
        json.dump(output_data, f, indent=2)
        
    print(f"Saved to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
