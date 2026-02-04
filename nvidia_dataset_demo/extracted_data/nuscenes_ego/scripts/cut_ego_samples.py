import numpy as np
import os
import subprocess
import glob
from tqdm import tqdm

# Constants
TRAIN_FILE = "/home/jadelynn/cf-reasoning/nvidia_dataset_demo/extracted_data/nuscenes_ego/calibration/nuscenes_train.npy"
VAL_FILE = "/home/jadelynn/cf-reasoning/nvidia_dataset_demo/extracted_data/nuscenes_ego/calibration/nuscenes_val.npy"
VIDEO_DIR = "/home/shared_data/external_drive/nuScenes/videos"
OUTPUT_DIR = "/home/shared_data/external_drive/nuScenes/video_samples"

# Parameters
HISTORY_SEC = 5.0
FUTURE_SEC = 4.0
SAMPLE_RATE = 2.0 # 2Hz data annotations
# SCENE_LIMIT will be parsed via argparse

def ensure_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

def load_data(filepath):
    print(f"Loading {filepath}...")
    try:
        data = np.load(filepath, allow_pickle=True)
        if data.shape == ():
            data = data.item()
        return data
    except Exception as e:
        print(f"Error loading {filepath}: {e}")
        return None

def process_cut(idx, ts, data_idx, video_path, output_dir):
    # idx is scene name (e.g., scene-0001)
    # ts is the timestep index (at 2Hz) representing "start of future"
    # So t=0 for the cut corresponds to ts (relative to scene start)
    # Start of clip = (ts * 0.5) - 5.0
    # Duration = 5.0 + 4.0 = 9.0
    
    start_time = (float(ts) / SAMPLE_RATE) - HISTORY_SEC
    duration = HISTORY_SEC + FUTURE_SEC
    
    # Handle negative start time (pad with black? or just clamp?)
    # ffmpeg handles negative ss by seeking before start? No.
    # We should clamp to 0 if start < 0, but then we lose history.
    # However, dataset description implies valid segments.
    if start_time < 0:
        # print(f"Warning: Start time {start_time} < 0 for {idx}, clamping to 0.")
        start_time = 0.0
        
    scene_name = idx
    # Parse scene token from video filename?
    # The .npy gives 'scene-XXXX'. The videos are named by TOKEN.
    # We need a mapping from 'scene-XXXX' to 'TOKEN'.
    # Ah, I don't have that mapping loaded here!
    # nuscenes_ego_front_mapping.json keys are SCENE TOKENS.
    # but .npy uses 'scene-0301'.
    # I need 'scene.json' or 'v1.0-trainval/scene.json' to map Names to Tokens.
    
    # Wait, let's verify if 'idx' in .npy is token or name.
    # Inspection showed "Sample[0]: scene-0301". That looks like a name.
    # My videos are named by TOKEN (e.g., "73030fb...").
    
    # I MUST load scene.json to map name -> token.
    return start_time, duration

def load_scene_mapping(meta_dir):
    scene_json = os.path.join(meta_dir, "scene.json")
    if not os.path.exists(scene_json):
        print(f"Error: scene.json not found at {scene_json}")
        return {}
    
    import json
    with open(scene_json, 'r') as f:
        scenes = json.load(f)
        
    # Map name -> token
    name_to_token = {s['name']: s['token'] for s in scenes}
    return name_to_token

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=100, help="Limit number of scenes to process")
    args = parser.parse_args()
    SCENE_LIMIT = args.limit

    ensure_dir(OUTPUT_DIR)
    
    # Load scene mapping
    meta_dir = "/home/jadelynn/cf-reasoning/nvidia_dataset_demo/extracted_data/nuscenes_ego/nuscenes_meta/v1.0-trainval"
    name_to_token = load_scene_mapping(meta_dir)
    if not name_to_token:
        return

    # Load NPY data
    # We'll merge train and val for processing
    all_data = []
    
    for fpath in [TRAIN_FILE, VAL_FILE]:
        d = load_data(fpath)
        if d:
            # d is dict of lists: idx, ts, data_idx, dist
            # Convert to list of dicts or just iterate index
            count = len(d['idx'])
            for i in range(count):
                all_data.append({
                    'idx': d['idx'][i],
                    'ts': d['ts'][i],
                    'data_idx': d['data_idx'][i],
                    'score': d['dist'][i] # assuming dist is score
                })
                
    print(f"Total samples to process: {len(all_data)}")
    
    # Filter for scenes we have videos for
    # My videos are tokens.
    available_videos = glob.glob(os.path.join(VIDEO_DIR, "*.mp4"))
    available_tokens = set([os.path.splitext(os.path.basename(v))[0] for v in available_videos])
    
    print(f"Available videos: {len(available_tokens)}")
    
    processed_count = 0
    skipped_count = 0
    existing_count = 0
    
    for sample in tqdm(all_data, desc="Cutting Samples"):
        scene_name = sample['idx']
        if scene_name not in name_to_token:
            # valid scene name check
            continue
            
        scene_token = name_to_token[scene_name]
        
        if scene_token not in available_tokens:
            skipped_count += 1
            continue
            
        video_path = os.path.join(VIDEO_DIR, f"{scene_token}.mp4")
        
        # Calculate cut times
        ts = sample['ts']
        start_time = max(0.0, (float(ts) / SAMPLE_RATE) - HISTORY_SEC)
        duration = HISTORY_SEC + FUTURE_SEC
        
        # Output filename: sceneName_sampleIdx.mp4 or data_idx?
        # User prompt showed "data_idx" list.
        # Let's use data_idx to be unique and traceable.
        out_name = f"{scene_name}_{sample['data_idx']}.mp4"
        out_path = os.path.join(OUTPUT_DIR, out_name)
        
        if os.path.exists(out_path) and os.path.getsize(out_path) > 0:
            existing_count += 1
            continue
            
        # ffmpeg cut
        # -ss before -i is faster (keyframe seek) but might preserve timestamps?
        # Re-encoding is safer for precision cutting.
        # -c:v libx264 -c:a copy
        cmd = [
            "ffmpeg", "-y",
            "-ss", str(start_time),
            "-t", str(duration),
            "-i", video_path,
            "-c:v", "libx264",
            "-pix_fmt", "yuv420p", # Maintain web compat
            "-loglevel", "error",
            out_path
        ]
        
        try:
            subprocess.run(cmd, check=True)
            processed_count += 1
        except subprocess.CalledProcessError:
            print(f"Error cutting {out_name}")

    print(f"Cutting complete.")
    total_expected = processed_count + existing_count
    print(f"Total Expected Samples (for available videos): {total_expected}")
    print(f"Newly Processed: {processed_count}")
    print(f"Skipped (Existing): {existing_count}")
    print(f"Skipped (No Source Video): {skipped_count}")

if __name__ == "__main__":
    main()
