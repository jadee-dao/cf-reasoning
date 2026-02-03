import numpy as np
import os

TRAIN_FILE = "/home/jadelynn/cf-reasoning/nvidia_dataset_demo/extracted_data/nuscenes_ego/calibration/nuscenes_train.npy"
VAL_FILE = "/home/jadelynn/cf-reasoning/nvidia_dataset_demo/extracted_data/nuscenes_ego/calibration/nuscenes_val.npy"

def inspect_npy(filepath):
    print(f"--- Inspecting {os.path.basename(filepath)} ---")
    if not os.path.exists(filepath):
        print("File not found.")
        return

    try:
        data = np.load(filepath, allow_pickle=True)
        print(f"Type: {type(data)}")
        
        # If it's a 0-d array wrapping a dict (common for saving dicts in npy)
        if data.shape == ():
            data = data.item()
            
        if isinstance(data, dict):
            print(f"Keys: {list(data.keys())}")
            for key in data:
                val = data[key]
                print(f"  Key '{key}': type={type(val)}, shape={getattr(val, 'shape', 'N/A')}")
                if hasattr(val, '__getitem__') and len(val) > 0:
                    print(f"    Sample[0]: {val[0]}")
        else:
            print(f"Shape: {data.shape}")
            print(f"Sample: {data[:1]}")
            
    except Exception as e:
        print(f"Error loading: {e}")

inspect_npy(TRAIN_FILE)
inspect_npy(VAL_FILE)
