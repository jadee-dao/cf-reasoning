import numpy as np
import os
import json
import matplotlib.pyplot as plt

train_path = 'nvidia_dataset_demo/extracted_data/calibration_set/nuscenes_train.npy'
val_path = 'nvidia_dataset_demo/extracted_data/calibration_set/nuscenes_val.npy'
output_json_path = 'nvidia_dataset_demo/extracted_data/calibration_set/percentile_outliers.json'
output_plot_path = 'nvidia_dataset_demo/extracted_data/calibration_set/percentile_distribution.png'

def calculate_and_export():
    try:
        # Load data
        print("Loading data...")
        train_data = np.load(train_path, allow_pickle=True).item()
        val_data = np.load(val_path, allow_pickle=True).item()
        
        # Extract distributions and metadata
        # Assuming structure based on previous inspection
        # train_data keys: ['idx', 'ts', 'data_idx', 'dist']
        # idx is list of scene IDs
        # dist is array of float values
        
        train_dist = np.array(train_data['dist'])
        val_dist = np.array(val_data['dist'])
        
        train_idx = train_data['idx']
        val_idx = val_data['idx']
        
        # Combine for percentile calculation
        combined_dist = np.concatenate([train_dist, val_dist])
        total_samples = len(combined_dist)
        print(f"Total samples: {total_samples}")
        
        p90 = np.percentile(combined_dist, 90)
        p99 = np.percentile(combined_dist, 99)
        
        print(f"90th Percentile: {p90:.4f}")
        print(f"99th Percentile: {p99:.4f}")
        
        # Prepare JSON results
        results = {}
        
        def process_dataset(dataset_name, dist_arr, idx_arr, data_obj):
            count = 0
            for i in range(len(dist_arr)):
                val = dist_arr[i]
                if val > p99:
                    scene_id = idx_arr[i]
                    # Create a unique key
                    key = f"{dataset_name}.{scene_id}.{i}" 
                    
                    # Entry structure similar to reference
                    entry = {
                        "chunk_name": dataset_name,
                        "scene_id": scene_id,
                        "num_eval_points": 1, # Placeholder
                        "top3_worst": [
                            {
                                "dist": float(val),
                                "data_idx": int(data_obj['data_idx'][i]) if 'data_idx' in data_obj else i,
                                "frame_index": i # approximate
                            }
                        ]
                    }
                    results[key] = entry
                    count += 1
            return count

        print("Processing Train set...")
        train_outliers = process_dataset("train", train_dist, train_idx, train_data)
        print(f"Found {train_outliers} outliers in Train.")
        
        print("Processing Val set...")
        val_outliers = process_dataset("val", val_dist, val_idx, val_data)
        print(f"Found {val_outliers} outliers in Val.")
        
        output_data = {
            "args": {
                "description": "99th percentile outliers from calibration set",
                "p90": float(p90),
                "p99": float(p99)
            },
            "num_scenes": len(results),
            "results": results
        }
        
        # Save JSON
        print(f"Saving JSON to {output_json_path}...")
        with open(output_json_path, 'w') as f:
            json.dump(output_data, f, indent=2)
            
        # Plotting
        print("Generating plot...")
        plt.figure(figsize=(12, 6))
        # Use log scale for y axis if distribution is skewed (implied by 'log-10' in reference filename)
        # But 'dist' values seen were 5.6, 9.4 etc. linear hist is probably fine, maybe log x?
        # Reference implies "log-10-90pctl", maybe log10(ADE)?
        # For now, I'll plot the raw distribution as requested "distribution.png" usually implies hist.
        
        plt.hist(combined_dist, bins=100, alpha=0.7, color='skyblue', label='Distribution')
        
        plt.axvline(p90, color='green', linestyle='dashed', linewidth=2, label=f'90th % ({p90:.2f})')
        plt.axvline(p99, color='red', linestyle='dashed', linewidth=2, label=f'99th % ({p99:.2f})')
        
        plt.title('Distribution of Calibration Set Values')
        plt.xlabel('Value')
        plt.ylabel('Frequency')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        print(f"Saving plot to {output_plot_path}...")
        plt.savefig(output_plot_path)
        print("Done.")

    except Exception as e:
        print(f"Error: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    calculate_and_export()
