# %% [markdown]
# # NVIDIA PhysicalAI Dataset Explanation
# 
# This notebook explains the structure and contents of the **PhysicalAI-Autonomous-Vehicles** dataset.
# Use this to understand what input data is available for each sample.

# %%
import os
import glob
import pandas as pd

# Determine script directory to locate data
script_dir = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(script_dir, '../extracted_data')

def get_samples(data_dir):
    """Scans the directory and returns a list of sample dictionaries."""
    mp4_files = glob.glob(os.path.join(data_dir, "*.mp4"))
    samples = []
    for f in mp4_files:
        base_name = os.path.basename(f).replace('.mp4', '')
        samples.append({
            'id': base_name,
            'video_path': f,
            'timestamps_path': os.path.join(data_dir, f"{base_name}.timestamps.parquet"),
            'blurred_boxes_path': os.path.join(data_dir, f"{base_name}.blurred_boxes.parquet")
        })
    return samples

# %% [markdown]
# ## 1. Input Data Structure
# For **each example** in the dataset (identified by a unique UUID), you have exactly three files:
# 
# 1.  **Video File (`.mp4`)**: The raw camera footage.
# 2.  **Timestamps File (`.timestamps.parquet`)**: Exact timing data for each frame.
# 3.  **Blurred Boxes File (`.blurred_boxes.parquet`)**: Ground truth data for anonymized regions.

# %%
samples = get_samples(DATA_DIR)
print(f"Total Samples Available: {len(samples)}")

# Pick a sample to inspect
if samples:
    # Using sample[2] as before, or any random one
    sample = samples[2] 
    print(f"\n--- Inspecting Sample: {sample['id']} ---")
    print(f"1. Video:      {os.path.basename(sample['video_path'])}")
    print(f"2. Timestamps: {os.path.basename(sample['timestamps_path'])}")
    print(f"3. Boxes:      {os.path.basename(sample['blurred_boxes_path'])}")

# %% [markdown]
# ## 2. Timestamps Data (`.timestamps.parquet`)
# This file contains the precise capture time for every frame in the video.
# 
# - **Rows**: One row per video frame.
# - **Columns**:
#     - `timestamp`: The capture time in **microseconds**.
# 
# **Utility**:
# - Calculate exact frame rates.
# - Synchronize this camera with other potential sensors (lidar, IMU) if they were present.
# - Detect frame drops or jitter.

# %%
if samples:
    df_ts = pd.read_parquet(sample['timestamps_path'])
    print("Timestamps DataFrame Head:")
    print(df_ts.head())
    print(f"\nShape: {df_ts.shape} (Rows = Total Frames)")
    
    # Example calculation
    duration_us = df_ts['timestamp'].max() - df_ts['timestamp'].min()
    print(f"Duration: {duration_us / 1_000_000:.2f} seconds")

# %% [markdown]
# ## 3. Blurred Boxes Data (`.blurred_boxes.parquet`)
# This file contains the bounding boxes for regions that have been anonymized (e.g., faces, license plates).
# 
# - **Rows**: One row per **bounding box** (not per frame). If a frame has 3 boxes, it has 3 rows.
# - **Columns**:
#     - `frame_index`: The frame number (0-indexed) where this box appears.
#     - `x1`: Left coordinate (pixels).
#     - `y1`: Top coordinate (pixels).
#     - `x2`: Right coordinate (pixels).
#     - `y2`: Bottom coordinate (pixels).
# 
# **Utility**:
# - Ground truth for privacy preservation tasks.
# - Can be used to evaluate detection models (check if they detect these protected objects).
# - **Note**: Not all frames will have boxes! Only frames with sensitive content.

# %%
if samples:
    df_bb = pd.read_parquet(sample['blurred_boxes_path'])
    print("Blurred Boxes DataFrame Head:")
    print(df_bb.head())
    print(f"\nShape: {df_bb.shape} (Total Boxes in this clip)")
    print(f"Unique Frames with Boxes: {df_bb['frame_index'].nunique()}")
    
    # Check basic stats
    if not df_bb.empty:
        avg_width = (df_bb['x2'] - df_bb['x1']).mean()
        avg_height = (df_bb['y2'] - df_bb['y1']).mean()
        print(f"Average Box Size: {avg_width:.1f} x {avg_height:.1f} pixels")

# %% [markdown]
# ## Summary: What You Have
# 
# For every clip, you have:
# - **Visuals**: 1920x1080 RGB Video @ 30 FPS.
# - **Temporal**: Microsecond-precision timestamps for every frame.
# - **Annotations**: Bounding box coordinates (`x1, y1, x2, y2`) for every anonymized object, mapped to specific `frame_index`.
# 
# You can use this to train/test:
# - Object detection (specifically for faces/plates).
# - Anonymization systems.
# - Video analysis pipelines requiring precise timing.
