import os
import json
import glob
import argparse

def main():
    parser = argparse.ArgumentParser(description="Merge ASIL scores into projection JSONs.")
    parser.add_argument("--dataset", type=str, required=True, help="Dataset name, e.g., nuscenes_ego")
    args = parser.parse_args()
    
    base_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.join(base_dir, "..")
    
    # Load ASIL Scores
    asil_file = os.path.join(project_root, "analysis_results", "ground_truth", f"asil_scores_{args.dataset}.json")
    if not os.path.exists(asil_file):
        print(f"ASIL scores file not found: {asil_file}")
        return
        
    with open(asil_file, 'r') as f:
        asil_data = json.load(f)
        
    print(f"Loaded {len(asil_data)} ASIL scores.")
    
    # Iterate over all strategies in projections
    projections_dir = os.path.join(project_root, "analysis_results", "projections")
    strategies = [d for d in os.listdir(projections_dir) if os.path.isdir(os.path.join(projections_dir, d))]
    
    for strategy in strategies:
        target_json = os.path.join(projections_dir, strategy, f"{args.dataset}.json")
        if not os.path.exists(target_json):
            continue
            
        print(f"Updating {strategy}/{args.dataset}.json...")
        
        with open(target_json, 'r') as f:
            data = json.load(f)
            
        updated_count = 0
        
        # Data is usually {"points": [...]}
        if "points" in data:
            for point in data["points"]:
                pid = point.get("id")
                # Handle possible ID mismatch (e.g. filename vs UUID)
                # ASIL keys are likely "scene-XXXX_TIMESTAMP"
                if pid in asil_data:
                    scores = asil_data[pid]
                    
                    if "gt_outliers" not in point:
                        point["gt_outliers"] = {}
                        
                    score = scores.get("asil_score", 0)
                    point["gt_outliers"]["asil_ge_A"] = bool(score >= 1)
                    point["gt_outliers"]["asil_ge_B"] = bool(score >= 2)
                    
                    point["asil_severity"] = scores.get("severity")
                    point["asil_exposure"] = scores.get("exposure")
                    point["asil_controllability"] = scores.get("controllability")
                    point["asil_score"] = score
                    point["asil_reasoning"] = scores.get("reasoning")
                    
                    updated_count += 1
        
        # Save back
        with open(target_json, 'w') as f:
            json.dump(data, f, indent=2)
            
        print(f"  Updated {updated_count} points.")

if __name__ == "__main__":
    main()
