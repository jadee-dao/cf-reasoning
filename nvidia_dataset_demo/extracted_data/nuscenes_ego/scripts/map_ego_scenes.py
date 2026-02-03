import json
import os
from collections import defaultdict

# Constants
META_DIR = "/home/jadelynn/cf-reasoning/nvidia_dataset_demo/extracted_data/nuscenes_ego/nuscenes_meta/v1.0-trainval"
OUTPUT_FILE = "/home/jadelynn/cf-reasoning/nvidia_dataset_demo/extracted_data/nuscenes_ego/nuscenes_ego_front_mapping.json"
TARGET_CHANNEL = "CAM_FRONT"

def load_json(filename):
    path = os.path.join(META_DIR, filename)
    print(f"Loading {path}...")
    with open(path, 'r') as f:
        return json.load(f)

def main():
    # 1. Load necessary metadata
    try:
        sample_data = load_json("sample_data.json")
        scene_data = load_json("scene.json")
        sample_json = load_json("sample.json")
    except FileNotFoundError as e:
        print(f"Error loading metadata: {e}")
        return

    print("Metadata loaded. Processing...")

    # 2. Index samples by scene
    # scene_token -> [sample_tokens]
    # We rely on sample.json to link samples to scenes.
    # sample["scene_token"]
    
    scene_to_samples = defaultdict(list)
    sample_token_to_scene = {}
    
    for s in sample_json:
        scene_token = s["scene_token"]
        sample_token = s["token"]
        scene_to_samples[scene_token].append(sample_token)
        sample_token_to_scene[sample_token] = scene_token

    print(f"Indexed {len(sample_json)} samples across {len(scene_to_samples)} scenes.")

    # 3. Filter sample_data for CAM_FRONT and key_frames (samples)
    # Actually, the user might want ALL sweeps or just samples?
    # Usually 'samples' are the keyframes (annotated). 'sweeps' are intermediate.
    # The snippet in extract_images.py suggested "samples/RADAR_FRONT..."
    # Let's target strictly the data associated with 'sample' (keyframes) or ALL data?
    # If the goal is "EGO_FRONT data", likely we want the video feed, so ALL frames (samples + sweeps).
    # But for now, let's start with matching identifying the file paths.
    
    # We need to map scene -> [filenames]
    # In sample_data.json:
    # "sample_token": "...", matches sample['token']
    # "channel": (we inferred this from filename or calibration? No, let's check sensor.json? 
    #   Wait, sample_data.json doesn't have "channel" field directly in the entries I saw earlier?
    #   Let me check the `view_file` output of sample_data.json again...
    #   Ah, I saw lines 1-800 of sample_data.json. 
    #   Field "filename": "samples/RADAR_FRONT/..." 
    #   Field "calibrated_sensor_token": ...
    #   We can filter by filename containing "CAM_FRONT" or check calibrated_sensor.
    
    # Let's look at sensor.json from earlier `view_file`.
    # "channel": "CAM_FRONT" -> token "725903f5b62f56118f4094b46a4470d8"
    
    # So we should find all calibrated_sensor_tokens that correspond to CAM_FRONT?
    # calibrated_sensor.json maps calibrated_sensor_token -> sensor_token.
    # sensor.json maps sensor_token -> channel name.
    
    # Let's load sensor.json and calibrated_sensor.json to be precise.
    sensor_data = load_json("sensor.json")
    calibrated_sensor_data = load_json("calibrated_sensor.json")
    
    # Map sensor token to channel
    sensor_token_to_channel = {s["token"]: s["channel"] for s in sensor_data}
    
    # Identify calibrated sensor tokens for CAM_FRONT
    cam_front_calib_tokens = set()
    for cs in calibrated_sensor_data:
        sensor_token = cs["sensor_token"]
        if sensor_token in sensor_token_to_channel:
             if sensor_token_to_channel[sensor_token] == TARGET_CHANNEL:
                 cam_front_calib_tokens.add(cs["token"])
    
    print(f"Found {len(cam_front_calib_tokens)} calibrated sensor tokens for {TARGET_CHANNEL}")

    # Now filter sample_data
    scene_to_files = defaultdict(list)
    
    count = 0
    for sd in sample_data:
        if sd["calibrated_sensor_token"] in cam_front_calib_tokens:
            # linking to scene...
            # sd has "sample_token".
            # If sd is a keyframe, it points to a sample.
            # If sd is a sweep, does it point to a sample? 
            # In the file view: "sample_token": "..." is present even for sweeps (is_key_frame: false).
            # So we can link to scene via sample.
            
            s_token = sd["sample_token"]
            if s_token in sample_token_to_scene:
                scene_token = sample_token_to_scene[s_token]
                # Store (timestamp, filename) tuple
                scene_to_files[scene_token].append((sd["timestamp"], sd["filename"]))
                count += 1
            else:
                # This might happen if sample_data points to a sample not in the split (unlikely for "trainval")
                pass

    # Sort by timestamp and extract filenames
    final_output = {}
    for scene_token, files in scene_to_files.items():
        # files is a list of (timestamp, filename)
        # Sort by timestamp (first element of tuple)
        files.sort(key=lambda x: x[0])
        # Keep only filenames
        final_output[scene_token] = [f[1] for f in files]

    print(f"Mapped {count} {TARGET_CHANNEL} files across {len(final_output)} scenes.")
    
    # Write output
    with open(OUTPUT_FILE, 'w') as f:
        json.dump(final_output, f, indent=2)
        
    print(f"Mapping saved to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
