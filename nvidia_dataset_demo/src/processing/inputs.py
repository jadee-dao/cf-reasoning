import os
import cv2
import torch
import numpy as np
from abc import ABC, abstractmethod

class InputLoader(ABC):
    """
    Abstract strategy for loading input data from a sample info dict.
    """
    @abstractmethod
    def load(self, data_dir: str, sample_info: dict) -> torch.Tensor:
        """
        Args:
            data_dir: Root directory of data (e.g., .../samples)
            sample_info: Dict with 'video_filename', 'scene_id', etc.
        Returns:
            Tensor representing the input (e.g., [3, T, H, W] or [3, H, W])
        """
        pass
    
    @property
    @abstractmethod
    def modality(self) -> str:
        """Returns 'video', 'image', etc."""
        pass

class VideoLoader(InputLoader):
    """
    Loads video clips, resizes, and interpolates to fixed temporal size.
    Output: (3, 16, 224, 224)
    """
    def __init__(self, num_frames=16, height=224, width=224):
        self.num_frames = num_frames
        self.height = height
        self.width = width
        
    @property
    def modality(self) -> str:
        return 'video'
        
    def load(self, data_dir: str, sample_info: dict) -> torch.Tensor:
        video_path = os.path.join(data_dir, sample_info['video_filename'])
        cap = cv2.VideoCapture(video_path)
        frames = []
        
        if not cap.isOpened():
            return torch.zeros((3, self.num_frames, self.height, self.width))
            
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            # Resize early to save memory if needed, but doing it later is fine for short clips
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)
        cap.release()
        
        if not frames:
             return torch.zeros((3, self.num_frames, self.height, self.width))
             
        video = np.array(frames) # (T, H, W, 3)
        video = torch.from_numpy(video).float() / 255.0
        video = video.permute(3, 0, 1, 2) # (C, T, H, W)
        
        # Interpolate to fixed size
        video = torch.nn.functional.interpolate(
            video.unsqueeze(0), 
            size=(self.num_frames, self.height, self.width), 
            mode='trilinear', 
            align_corners=False
        ).squeeze(0)
        
        return video

class ImageLoader(InputLoader):
    """
    Loads a single frame (middle frame) from the video file.
    Output: (3, 224, 224)
    """
    def __init__(self, height=224, width=224):
        self.height = height
        self.width = width
        
    @property
    def modality(self) -> str:
        return 'image'
        
    def load(self, data_dir: str, sample_info: dict) -> torch.Tensor:
        video_path = os.path.join(data_dir, sample_info['video_filename'])
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            return torch.zeros((3, self.height, self.width))
            
        # Get total frames to find middle
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames > 0:
            middle_idx = total_frames // 2
            cap.set(cv2.CAP_PROP_POS_FRAMES, middle_idx)
            
        ret, frame = cap.read()
        cap.release()
        
        if not ret:
             return torch.zeros((3, self.height, self.width))
             
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.resize(frame, (self.width, self.height))
        
        image = torch.from_numpy(frame).float() / 255.0
        image = image.permute(2, 0, 1) # (C, H, W)
        
        return image
