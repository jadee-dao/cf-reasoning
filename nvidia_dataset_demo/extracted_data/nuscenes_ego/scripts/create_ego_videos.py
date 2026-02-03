import json
import os
import sys
import subprocess
from tqdm import tqdm

# Constants
MAPPING_FILE = "/home/jadelynn/cf-reasoning/nvidia_dataset_demo/extracted_data/nuscenes_ego/nuscenes_ego_front_mapping.json"
IMAGE_ROOT = "/home/jadelynn/cf-reasoning/nvidia_dataset_demo/extracted_data/raw_data/nuscenes_ego"
OUTPUT_DIR = "/home/jadelynn/cf-reasoning/nvidia_dataset_demo/extracted_data/raw_data/nuscenes_ego/videos"
FPS = 12

import argparse

def ensure_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=100, help="Limit number of scenes")
    args = parser.parse_args()
    MAX_SCENES = args.limit

    if not os.path.exists(MAPPING_FILE):
        print(f"Error: Mapping file not found at {MAPPING_FILE}")
        sys.exit(1)
    
    print(f"Loading mapping from {MAPPING_FILE}...")
    with open(MAPPING_FILE, 'r') as f:
        scene_mapping = json.load(f)

    print(f"Limiting video generation to first {MAX_SCENES} scenes.")
    scene_mapping = dict(list(scene_mapping.items())[:MAX_SCENES])

    ensure_dir(OUTPUT_DIR)
    
    print(f"Found {len(scene_mapping)} scenes. Starting video generation with ffmpeg...")
    
    # Iterate over scenes
    count_generated = 0
    count_skipped = 0
    
    for scene_token, image_files in tqdm(scene_mapping.items(), desc="Processing Scenes"):
        output_path = os.path.join(OUTPUT_DIR, f"{scene_token}.mp4")
        
        if os.path.exists(output_path):
            # Check if file has size > 0
            if os.path.getsize(output_path) > 0:
                count_skipped += 1
                continue
            else:
                 os.remove(output_path) # Remove empty files

        if not image_files:
            continue
            
        # Create a temporary file list for ffmpeg
        # ffmpeg concat demuxer format: "file '/path/to/image'"
        # duration 1/FPS
        
        list_file_path = os.path.join(OUTPUT_DIR, f"{scene_token}_list.txt")
        valid_images = 0
        with open(list_file_path, 'w') as f:
            for img_rel_path in image_files:
                full_path = os.path.join(IMAGE_ROOT, img_rel_path)
                if os.path.exists(full_path):
                    # Escape single quotes in path if necessary
                    safe_path = full_path.replace("'", "'\\''")
                    f.write(f"file '{safe_path}'\n")
                    f.write(f"duration {1.0/FPS}\n")
                    valid_images += 1
        
        if not valid_images:
            print(f"Skipping {scene_token}, no valid images found locally.")
            # DEBUG: Print the first path checked
            if image_files:
                 debug_path = os.path.join(IMAGE_ROOT, image_files[0])
                 print(f"DEBUG: Checked for {debug_path}")
            os.remove(list_file_path)
            continue
            
        # Due to a quirk in ffmpeg concat demuxer, the last image needs to be specified again or it might be skipped/short.
        # But generally for simple video it's fine. 
        # Actually proper way: last entry just "file path" without duration effectively gives it default? 
        # The concat format specifies "duration" applies to the *preceding* file.
        # So we need to ensure the last frame also has duration?
        # Actually, standard practice for slide shows:
        # file 'path'
        # duration 5
        # ...
        # file 'path' (last one needs to be repeated? No, simply the last record determines when stream ends?)
        # Let's stick to the simplest input pipe or glob if possible? 
        # Glob is hard because filenames aren't perfectly sequential numbers (they are timestamps).
        # So concat demuxer is correct.
        
        # ffmpeg command
        # -f concat -safe 0 -i list.txt -c:v libx264 -pix_fmt yuv420p -r 12 output.mp4
        cmd = [
            "ffmpeg",
            "-y", # Overwrite
            "-f", "concat",
            "-safe", "0",
            "-i", list_file_path,
            "-c:v", "libx264",
            "-pix_fmt", "yuv420p", # Essential for web compatibility
            "-r", str(FPS),
            output_path
        ]
        
        try:
            subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
            count_generated += 1
        except subprocess.CalledProcessError as e:
            print(f"Error generating video for {scene_token}")
        
        # Cleanup list file
        if os.path.exists(list_file_path):
            os.remove(list_file_path)

        if count_generated >= MAX_SCENES:
             print(f"Reached limit of {MAX_SCENES} videos. Stopping.")
             break

    print(f"Video generation complete.")
    print(f"Generated: {count_generated}")
    print(f"Skipped: {count_skipped}")

if __name__ == "__main__":
    main()
