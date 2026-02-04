import json
import os
import subprocess
import glob

import argparse

# Constants
MAPPING_FILE = "/home/jadelynn/cf-reasoning/nvidia_dataset_demo/extracted_data/nuscenes_ego/nuscenes_ego_front_mapping.json"
BLOBS_DIR = "/home/shared_data/external_drive/nuScenes"
EXTRACT_DIR = "/home/shared_data/external_drive/nuScenes"
FILE_LIST_PATH = "/home/jadelynn/cf-reasoning/nvidia_dataset_demo/extracted_data/ego_files_to_extract.txt"

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=100, help="Limit number of scenes")
    args = parser.parse_args()
    
    SCENE_LIMIT = args.limit
    # 1. Load mapping
    print(f"Loading {MAPPING_FILE}...")
    with open(MAPPING_FILE, 'r') as f:
        scene_mapping = json.load(f)

    if SCENE_LIMIT: # Limit to first 100 scenes
        print(f"Limiting extraction to first {SCENE_LIMIT} scenes.")
        # Slice the dictionary
        scene_mapping = dict(list(scene_mapping.items())[:SCENE_LIMIT])

    # 2. Collect all unique images
    all_images = set()
    for scene, images in scene_mapping.items():
        for img in images:
            all_images.add(img)
            
    print(f"Total unique images to extract: {len(all_images)}")
    
    # 4. Check for existing files first
    missing_files = []
    print("Checking for existing files...")
    for img in all_images:
        if not os.path.exists(os.path.join(EXTRACT_DIR, img)):
            missing_files.append(img)
            
    print(f"Missing {len(missing_files)} / {len(all_images)} files.")
    
    if len(missing_files) == 0:
        print("All files present, skipping extraction.")
        return

    # Write ONLY missing files to list
    with open(FILE_LIST_PATH, 'w') as f:
        for img in missing_files:
            f.write(img + "\n")
    print(f"File list updated with {len(missing_files)} missing files.")

    # 4. Find all blob archives
    archives = glob.glob(os.path.join(BLOBS_DIR, "*_blobs.tgz"))
    archives.sort()
    
    print(f"Found {len(archives)} blob archives.")
    
    # 5. Extract
    for i, archive in enumerate(archives):
        print(f"[{i+1}/{len(archives)}] Scanning {os.path.basename(archive)}...")
        
        # We use tar with --files-from. 
        # --skip-old-files avoids re-extracting if we run this multiple times.
        # We suppress stderr/stdout partially but capture errors if strictly needed.
        # GNU tar returns exit code 0 if success, >0 if some files not found (common here since we ask for ALL files from EACH blob).
        
        cmd = [
            "tar", 
            "-xf", archive, 
            "-C", EXTRACT_DIR, 
            "--files-from", FILE_LIST_PATH,
            "--skip-old-files" 
        ]
        
        try:
            # We ignore stderr because tar complains about missing files (since we ask for ALL files from EACH archive)
            subprocess.run(cmd, stderr=subprocess.DEVNULL, check=False)
        except Exception as e:
            print(f"Error processing {archive}: {e}")

    # 6. Verify count
    print("Verifying extraction...")
    found_count = 0
    # Checking 200k files might be slow, let's just check a sample or do a quick glob?
    # Actually, we can just walk the directory or check a subset.
    # Checking existence of each file is O(N) but disk access.
    
    # Quick check: check first 10 files from mapping
    missing_count = 0
    checked_count = 0
    
    for img in list(all_images)[:10]:
        path = os.path.join(EXTRACT_DIR, img)
        if os.path.exists(path):
            found_count += 1
        else:
            missing_count += 1
        checked_count += 1
            
    print(f"Checked {checked_count} sample files: {found_count} found, {missing_count} missing.")

if __name__ == "__main__":
    main()
