import os
import glob
import subprocess
from tqdm import tqdm

# Constants
INPUT_DIR = "/home/jadelynn/cf-reasoning/nvidia_dataset_demo/extracted_data/nuscenes_ego/video_samples"
OUTPUT_DIR = "/home/jadelynn/cf-reasoning/nvidia_dataset_demo/extracted_data/nuscenes_ego/samples"

# Cut parameters (relative to the 9s clip where t=5.0 is the event)
# We want [t-1.5, t] -> [3.5, 5.0]
START_TIME = 3.5
DURATION = 1.5

def ensure_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

def main():
    ensure_dir(OUTPUT_DIR)
    
    # Check input dir
    if not os.path.exists(INPUT_DIR):
        print(f"Error: Input directory {INPUT_DIR} does not exist.")
        return

    video_files = glob.glob(os.path.join(INPUT_DIR, "*.mp4"))
    print(f"Found {len(video_files)} video samples.")
    
    processed_count = 0
    error_count = 0
    
    for vid_path in tqdm(video_files, desc="Cutting Comparison Samples"):
        filename = os.path.basename(vid_path)
        out_path = os.path.join(OUTPUT_DIR, filename)
        
        if os.path.exists(out_path):
            processed_count += 1
            # print(f"Skipping {filename}, already exists.")
            continue
            
        cmd = [
            "ffmpeg", "-y",
            "-ss", str(START_TIME),
            "-t", str(DURATION),
            "-i", vid_path,
            "-c:v", "libx264",
            "-pix_fmt", "yuv420p",
            "-loglevel", "error",
            out_path
        ]
        
        try:
            subprocess.run(cmd, check=True)
            
            # Extract frame for valid processing by run_strategy.py
            jpg_out = out_path.replace(".mp4", ".jpg")
            cmd_jpg = [
                "ffmpeg", "-y",
                "-i", out_path,
                "-frames:v", "1",
                "-q:v", "2",
                "-loglevel", "error",
                jpg_out
            ]
            subprocess.run(cmd_jpg, check=True)
            
            processed_count += 1
        except subprocess.CalledProcessError:
            print(f"Error processing {filename}")
            error_count += 1

    print(f"Done. Processed: {processed_count}, Errors: {error_count}")

if __name__ == "__main__":
    main()
