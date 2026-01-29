import numpy as np

train_path = 'nvidia_dataset_demo/extracted_data/calibration_set/nuscenes_train.npy'

def check_keys():
    data = np.load(train_path, allow_pickle=True).item()
    print(f"First 5 idxs: {data['idx'][:5]}")
    print(f"First 5 data_idxs: {data['data_idx'][:5]}")
    print(f"First 5 ts: {data['ts'][:5]}")

if __name__ == "__main__":
    check_keys()
