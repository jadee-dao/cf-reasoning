
import numpy as np

npy_path = 'extracted_data/nuscenes_ego/calibration/nuscenes_train.npy'
data = np.load(npy_path, allow_pickle=True).item()
scores = list(data['dist'])
print(f"Min score: {np.min(scores)}")
print(f"Max score: {np.max(scores)}")
print(f"Mean score: {np.mean(scores)}")
print(f"Median score: {np.median(scores)}")
