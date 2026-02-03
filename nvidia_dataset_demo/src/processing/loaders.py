import os
import cv2
import json
import torch
import numpy as np
from torch.utils.data import Dataset
from typing import Optional, Callable, Dict, Any

from .inputs import InputLoader, VideoLoader

class NuScenesDataset(Dataset):
    """
    Dataset loader for nuScenes video clips and labels.
    """

    def __init__(self, 
                 data_dir: str, 
                 json_path: str, 
                 npy_path: str,
                 transform: Optional[Callable] = None,
                 target_type: str = 'p90', # 'p90', 'p99', 'score', 'log_score', 'bin_class'
                 score_norm_factor: float = 1000.0,
                 mode: str = 'train',
                 split_ratio: float = 0.8,
                 input_loader: Optional[InputLoader] = None):
        """
        Args:
            data_dir (str): Path to video samples.
            json_path (str): Path to outliers JSON.
            npy_path (str): Path to train npy file (scores).
            transform (callable): Transform pipeline.
            target_type (str): 'p90', 'p99', 'score', 'log_score', 'bin_class'.
            score_norm_factor (float): Factor to divide scores by.
            mode (str): 'train' or 'val'.
            split_ratio (float): Split ratio.
            input_loader (InputLoader): Strategy for loading input (Video, Image, etc).
        """
        self.data_dir = data_dir
        self.transform = transform
        self.target_type = target_type
        self.score_norm_factor = score_norm_factor
        self.mode = mode
        self.input_loader = input_loader or VideoLoader() # Default to VideoLoader

        # Load Outliers JSON (for classification)
        with open(json_path, 'r') as f:
            outliers_data = json.load(f)
        
        # Map (scene_id, data_idx) -> {is_p90, is_p99}
        self.outliers_map = {}
        for key, val in outliers_data.get('generated_samples', {}).items():
            # key is like scene-0303_44
            # We can also rely on val['scene_id'] and val['data_idx']
            sid = val['scene_id']
            didx = int(val['data_idx'])
            self.outliers_map[(sid, didx)] = {
                'is_p90': val.get('is_p90', False),
                'is_p99': val.get('is_p99', False)
            }
            
        # Load Scores NPY (for regression and valid sample filtering)
        # NPY structure: dict with keys 'idx' (scene_id), 'data_idx', 'dist' (score)
        try:
            train_data = np.load(npy_path, allow_pickle=True).item()
            self.scores_map = {}
            # Verify lists are same length
            num_items = len(train_data['idx'])
            for i in range(num_items):
                sid = train_data['idx'][i]
                didx = int(train_data['data_idx'][i])
                score = float(train_data['dist'][i])
                self.scores_map[(sid, didx)] = score
        except Exception as e:
            print(f"Error loading {npy_path}: {e}")
            self.scores_map = {}

        # Compute Percentile Bins if needed
        if self.target_type == 'bin_class':
            all_scores = list(self.scores_map.values())
            # Create 10 bins (deciles)
            # thresholds will have 9 values: 10th, 20th, ... 90th percentile
            self.bin_thresholds = np.percentile(all_scores, np.linspace(10, 90, 9))
            print(f"Decile Thresholds: {self.bin_thresholds}")

        # Scan directory for available videos
        video_files = [f for f in os.listdir(data_dir) if f.endswith('.mp4')]
        self.sample_list = []
        
        for vid_file in video_files:
            # Parse filename: scene-XXXX_YYYY.mp4
            try:
                base = os.path.splitext(vid_file)[0]
                parts = base.split('_')
                if len(parts) != 2:
                    continue
                sid = parts[0]
                didx = int(parts[1])
                
                # Check if we have a score for it (valid training sample)
                if (sid, didx) in self.scores_map:
                    self.sample_list.append({
                        'video_filename': vid_file,
                        'scene_id': sid,
                        'data_idx': didx
                    })
            except Exception as e:
                continue
                
        # Split
        self.sample_list.sort(key=lambda x: (x['scene_id'], x['data_idx']))
        split_idx = int(len(self.sample_list) * split_ratio)
        
        if mode == 'train':
            self.sample_list = self.sample_list[:split_idx]
        else:
            self.sample_list = self.sample_list[split_idx:]
            
        print(f"Dataset ({mode}): {len(self.sample_list)} samples loaded. Modality: {self.input_loader.modality}")

    def __len__(self):
        return len(self.sample_list)

    def __getitem__(self, idx):
        sample_info = self.sample_list[idx]
        
        # Use Strategy to load input
        video = self.input_loader.load(self.data_dir, sample_info)
        
        if self.transform:
            video = self.transform(video)

        # Labels
        sid = sample_info['scene_id']
        didx = sample_info['data_idx']
        key = (sid, didx)
        
        if self.target_type == 'score':
            target = self.scores_map.get(key, 0.0)
            target = target / self.score_norm_factor
            target = torch.tensor(target, dtype=torch.float32)
            
        elif self.target_type == 'log_score':
            # Log Normalization: log(1 + score)
            raw_score = self.scores_map.get(key, 0.0)
            target = np.log1p(raw_score)
            target = torch.tensor(target, dtype=torch.float32)
            
        elif self.target_type == 'bin_class':
            # Dynamic Percentile Binning (Deciles)
            raw_score = self.scores_map.get(key, 0.0)
            # np.digitize returns index of the bin the value belongs to.
            # bins[i-1] <= x < bins[i]
            # Since self.bin_thresholds has 9 values, digitize returns 0..9
            label = np.digitize(raw_score, self.bin_thresholds)
            target = torch.tensor(label, dtype=torch.long)

        elif self.target_type == 'p90':
            outlier_info = self.outliers_map.get(key, {'is_p90': False})
            target = 1.0 if outlier_info['is_p90'] else 0.0
            target = torch.tensor(target, dtype=torch.float32)
        elif self.target_type == 'p99':
            outlier_info = self.outliers_map.get(key, {'is_p99': False})
            target = 1.0 if outlier_info['is_p99'] else 0.0
            target = torch.tensor(target, dtype=torch.float32)
        else:
            raise ValueError(f"Unknown target_type: {self.target_type}")
            
        return video, target

