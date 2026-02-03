import os
import glob
import shutil
import subprocess
try:
    import cv2
except ImportError:
    cv2 = None
import json
from tqdm import tqdm
from abc import ABC, abstractmethod

class DatasetProcessor(ABC):
    def __init__(self, dataset_name, raw_data_path, output_path):
        self.dataset_name = dataset_name
        self.raw_data_path = raw_data_path
        self.output_path = output_path
        self.samples_dir = os.path.join(output_path, "samples")
        self.calibration_dir = os.path.join(output_path, "calibration")
        
        os.makedirs(self.samples_dir, exist_ok=True)
        os.makedirs(self.calibration_dir, exist_ok=True)

    @abstractmethod
    def process(self):
        pass

    def _extract_interval(self, video_path, output_video_path, output_image_path, start_time, duration):
        """Extracts a sub-clip and a representative frame."""
        try:
            # Extract sub-video
            cmd_video = [
                "ffmpeg", "-y", "-v", "error",
                "-ss", str(start_time),
                "-t", str(duration),
                "-i", video_path,
                "-c:v", "libx264", "-c:a", "aac",
                output_video_path
            ]
            subprocess.run(cmd_video, check=True)
            
            # Extract frame (middle of clip or start?) 
            # Original script used start_time. Let's stick to start_time for consistency with {timestamp} naming.
            cmd_frame = [
                "ffmpeg", "-y", "-v", "error",
                "-ss", str(start_time),
                "-i", video_path,
                "-frames:v", "1",
                "-q:v", "2",
                output_image_path
            ]
            subprocess.run(cmd_frame, check=True)
            return True
        except subprocess.CalledProcessError as e:
            print(f"Error processing video chunk {output_video_path}: {e}")
            return False

class NvidiaDemoProcessor(DatasetProcessor):
    def __init__(self, dataset_name="nvidia_demo", raw_data_path=None, output_path=None):
        # Default paths if not provided
        base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../extracted_data"))
        if raw_data_path is None:
            raw_data_path = os.path.join(base_dir, "raw_data", dataset_name)
        if output_path is None:
            output_path = os.path.join(base_dir, dataset_name)
            
        super().__init__(dataset_name, raw_data_path, output_path)
    
    def process(self, interval_sec=1.5, limit=None):
        print(f"Processing {self.dataset_name} from {self.raw_data_path}...")
        
        # Find raw videos
        pattern = os.path.join(self.raw_data_path, "**/*.mp4")
        videos = glob.glob(pattern, recursive=True)
        videos.sort()
        if limit:
            videos = videos[:limit]
            
        processed_count = 0
        
        # Merging calibration videos is removed as per user clarification (no videos in calibration dir)
        all_videos = videos
        
        pbar = tqdm(all_videos, desc="Processing Videos")
        for video_path in pbar:
            filename = os.path.basename(video_path)
            # Nvidia Demo convention: UUID.camera_front_wide_120fov.mp4
            if '.' in filename:
                uuid = filename.split('.')[0]
            else:
                uuid = filename
                
            pbar.set_description(f"Video: {uuid}")
            
            # is_calib check removed
            
            # If calibration, maybe keep original name or prefix?
            # User said: {sample_id}_{frame}
            # scene-0094.mp4 -> scene-0094_{timestamp}
            
            # Get Duration
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                # tqdm.write avoid breaking bar
                tqdm.write(f"Warning: Could not open {video_path}")
                continue
                
            fps = cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            duration_sec = total_frames / fps if fps > 0 else 0
            cap.release()
            
            if duration_sec <= 0:
                continue

            current_time = 0.0
            while current_time < duration_sec:
                ts_us = int(current_time * 1_000_000)
                sample_id = f"{uuid}_{ts_us}"
                
                # Output filenames
                out_vid = os.path.join(self.samples_dir, f"{sample_id}.mp4")
                out_img = os.path.join(self.samples_dir, f"{sample_id}.jpg")
                
                if not os.path.exists(out_vid) or not os.path.exists(out_img):
                    # self._extract_interval(video_path, out_vid, out_img, current_time, interval_sec)
                    # Removing print per user request
                    self._extract_interval(video_path, out_vid, out_img, current_time, interval_sec)
                
                processed_count += 1
                current_time += interval_sec
                
        print(f"Finished processing. Total samples processed/checked: {processed_count}")

class NuScenesEgoProcessor(DatasetProcessor):
    def __init__(self, dataset_name="nuscenes_ego", raw_data_path=None, output_path=None):
        # Default paths if not provided
        base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../extracted_data"))
        
        # Default output: extracted_data/nuscenes_ego
        if output_path is None:
            output_path = os.path.join(base_dir, dataset_name)
            
        # Default raw: extracted_data/raw_data/nuscenes_ego (though mostly unused for this specific flow)
        if raw_data_path is None:
            raw_data_path = os.path.join(base_dir, "raw_data", dataset_name)
            
        super().__init__(dataset_name, raw_data_path, output_path)
        
    def process(self, interval_sec=1.5, limit=None):
        print(f"Processing {self.dataset_name}...")
        
        # Input directory: where the 9s samples are
        # The user put them in extracted_data/nuscenes_ego/video_samples
        # We need to construct this path relative to base_dir or hardcode as per user structure
        # base_dir is .../extracted_data
        # output_path is .../extracted_data/nuscenes_ego
        
        # Assuming video_samples is at the same level as samples (sibling in output_path)
        input_dir = os.path.join(self.output_path, "video_samples")
        
        if not os.path.exists(input_dir):
            print(f"Error: Input directory {input_dir} does not exist.")
            return

        videos = glob.glob(os.path.join(input_dir, "*.mp4"))
        print(f"Found {len(videos)} video samples in {input_dir}. Processing to {self.samples_dir}...")
        
        processed_count = 0
        
        # We cut [3.5, 5.0] from the 9s clips
        start_time = 3.5
        duration = 1.5
        
        pbar = tqdm(videos, desc="Processing Samples")
        for video_path in pbar:
            filename = os.path.basename(video_path)
            pbar.set_description(f"Sample: {filename}")
            
            # filename is like scene-0001_13516.mp4
            # We want to keep the same name for the sample? 
            # Usually samples are named by timestamp/id. 
            # Check user request: "cut every single video... and put it in .../samples"
            # Preserving filename seems appropriate or using UUID convention if required.
            # Convert .mp4 filename to name without extension for ID
            sample_id = os.path.splitext(filename)[0]
            
            out_vid = os.path.join(self.samples_dir, filename)
            out_img = os.path.join(self.samples_dir, f"{sample_id}.jpg")
            
            if not os.path.exists(out_vid) or not os.path.exists(out_img):
                # Using the helper from base class
                success = self._extract_interval(video_path, out_vid, out_img, start_time, duration)
                if success:
                    processed_count += 1
            else:
                 processed_count += 1
                 
        print(f"Finished processing. Total samples: {processed_count}")

# Registry
PROCESSORS = {
    "nvidia_demo": NvidiaDemoProcessor,
    "nuscenes_ego": NuScenesEgoProcessor
}
