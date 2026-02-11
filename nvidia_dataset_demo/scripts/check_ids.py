
import pickle
import json
import os

PKL_PATH = "analysis_results/embeddings/semantic_counts/nuscenes_ego.pkl"
GT_PATH = "extracted_data/nuscenes_ego/calibration/nuscenes_percentile_outliers.json"

def main():
    if not os.path.exists(PKL_PATH):
        print("Pickle not found.")
        return
        
    with open(PKL_PATH, "rb") as f:
        data = pickle.load(f)
    pkl_ids = data["ids"]
    print(f"Loaded {len(pkl_ids)} IDs from pickle.")
    print(f"Sample Pickle IDs: {pkl_ids[:5]}")
    
    with open(GT_PATH, "r") as f:
        gt_data = json.load(f)
    gt_ids = list(gt_data.get("generated_samples", {}).keys())
    print(f"Loaded {len(gt_ids)} IDs from GT JSON.")
    print(f"Sample GT IDs: {gt_ids[:5]}")
    
    # Check intersection
    pkl_set = set(pkl_ids)
    intersection = [uid for uid in gt_ids if uid in pkl_set]
    print(f"\nIntersection count: {len(intersection)}")
    if intersection:
        print(f"Sample Intersection: {intersection[:5]}")

if __name__ == "__main__":
    main()
