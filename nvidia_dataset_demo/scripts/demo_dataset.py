# %% [markdown]
# # NVIDIA PhysicalAI Dataset Demo
# 
# This notebook demonstrates how to load and visualize the **NVIDIA PhysicalAI-Autonomous-Vehicles** dataset.
# The dataset contains:
# - **MP4 Videos**: Front-facing wide-angle camera footage (120° FOV).
# - **Timestamps**: Parquet files containing precise timestamps for each frame.
# - **Blurred Boxes**: Parquet files containing bounding boxes for anonymized regions (faces, license plates).

# %%
import os
import glob
import cv2
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import random

# Configuration
# Determine the directory of this script
script_dir = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(script_dir, '../extracted_data')

# Set generic plot style
plt.style.use('dark_background')

# %% [markdown]
# ## Helper Functions
# Functions to discover files and load data.

# %%
def get_dataset_samples(data_dir):
    """Finds all MP4 files and their corresponding parquet files."""
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

def load_sample_data(sample):
    """Loads dataframe content for a sample."""
    # Load Timestamps
    if os.path.exists(sample['timestamps_path']):
        df_ts = pd.read_parquet(sample['timestamps_path'])
    else:
        df_ts = pd.DataFrame()
        
    # Load Blurred Boxes
    if os.path.exists(sample['blurred_boxes_path']):
        df_bb = pd.read_parquet(sample['blurred_boxes_path'])
    else:
        df_bb = pd.DataFrame()
        
    return df_ts, df_bb

# %% [markdown]
# ## Load a Sample
# Let's pick a random video from the dataset and inspect its metadata.

# %%
samples = get_dataset_samples(DATA_DIR)
# sort samples by id
samples.sort(key=lambda x: x['id'])
print(f"Found {len(samples)} samples in {DATA_DIR}")

if samples:
    # Pick the first one for consistency, or random.choice(samples)
    current_sample = samples[9]
    print(f"Selected Sample: {current_sample['id']}")
    
    df_ts, df_bb = load_sample_data(current_sample)
    
    print(f"Timestamps: {len(df_ts)} frames")
    print(f"Blurred Boxes: {len(df_bb)} entries")
    if not df_bb.empty:
        print("Blurred Boxes Columns:", list(df_bb.columns))

# %% [markdown]
# ## Visualization
# We will read a frame from the video and overlay the ground-truth blurred boxes.
# These boxes indicate areas that were anonymized in the video.

# %%
def visualize_frame(sample, frame_index, ax=None):
    """Reads a specific frame and overlays blurred boxes."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 8))
        
    # Open Video
    cap = cv2.VideoCapture(sample['video_path'])
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
    ret, frame = cap.read()
    cap.release()
    
    if not ret:
        print(f"Could not read frame {frame_index}")
        return

    # Convert BGR to RGB
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Display Image
    ax.imshow(frame_rgb)
    ax.set_title(f"Frame {frame_index}")
    ax.axis('off')
    
    # Overlay Boxes
    # Filter boxes for this frame
    # Note: df_bb has 'frame_index' column
    if 'frame_index' in df_bb.columns:
        frame_boxes = df_bb[df_bb['frame_index'] == frame_index]
        
        for _, box in frame_boxes.iterrows():
            # Box columns: x1, y1, x2, y2
            x = box['x1']
            y = box['y1']
            w = box['x2'] - box['x1']
            h = box['y2'] - box['y1']
            
            # Create a Rectangle patch
            rect = patches.Rectangle((x, y), w, h, linewidth=2, edgecolor='r', facecolor='none', alpha=0.7)
            ax.add_patch(rect)
            ax.text(x, y-5, 'Anonymized', color='r', fontsize=8, fontweight='bold')

# Pick a frame that actually has boxes
if not df_bb.empty:
    frames_with_boxes = df_bb['frame_index'].unique()
    target_frame = frames_with_boxes[0] if len(frames_with_boxes) > 0 else 0
else:
    target_frame = 0

print(f"Visualizing Frame: {target_frame}")

fig, ax = plt.subplots(figsize=(15, 10))
visualize_frame(current_sample, target_frame, ax=ax)
plt.tight_layout()
output_path = os.path.join(script_dir, 'demo_visualization.png')
plt.savefig(output_path) # Save for verification
print(f"Saved visualization to: {output_path}")
plt.show()


# %% [markdown]
# ## Generate Labeled Video
# Create a video clip with the blurred boxes overlaid to visually verify the data.

# %%
def create_labeled_video(sample, df_bboxes, output_filename, max_frames=300):
    """Generates a video with bounding boxes overlaid."""
    cap = cv2.VideoCapture(sample['video_path'])
    if not cap.isOpened():
        print("Error: Could not open video.")
        return

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    # Define codec and create VideoWriter
    # Use mp4v for the intermediate file
    fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
    out = cv2.VideoWriter(output_filename, fourcc, fps, (width, height))
    
    frame_idx = 0
    box_count = 0
    print(f"Processing up to {max_frames} frames...")
    
    while cap.isOpened() and frame_idx < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
            
        # Draw Frame Info
        cv2.putText(frame, f"Frame: {frame_idx}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # Overlay boxes for this frame
        if 'frame_index' in df_bboxes.columns:
            frame_boxes = df_bboxes[df_bboxes['frame_index'] == frame_idx]
            if not frame_boxes.empty:
                print(f"  Drawing {len(frame_boxes)} boxes at frame {frame_idx}")
                for _, box in frame_boxes.iterrows():
                    x1, y1, x2, y2 = int(box['x1']), int(box['y1']), int(box['x2']), int(box['y2'])
                    # Draw Neon Green rectangle, Thicker
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)
                    cv2.putText(frame, 'ANON', (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                    box_count += len(frame_boxes)
        
        out.write(frame)
        frame_idx += 1
        
    cap.release()
    out.release()
    print(f"Processed {frame_idx} frames. Total boxes drawn: {box_count}")
    
    # Re-encode with ffmpeg to ensure H.264 compatibility (widely supported)
    # The 'mp4v' codec from OpenCV is often MPEG-4 Part 2, which fails in some browsers/players.
    # We rename the temp file and use ffmpeg to create the final file.
    temp_filename = output_filename + ".temp.mp4"
    if os.path.exists(output_filename):
        os.rename(output_filename, temp_filename)
        
    print("Re-encoding to H.264 with ffmpeg...")
    # ffmpeg command: -i input -c:v libx264 -pix_fmt yuv420p -c:a copy output
    # -y to overwrite, -v error to reduce output
    cmd = f"ffmpeg -y -v error -i {temp_filename} -c:v libx264 -pix_fmt yuv420p {output_filename}"
    ret = os.system(cmd)
    
    if ret == 0:
        print(f"Video saved and re-encoded to: {output_filename}")
        if os.path.exists(temp_filename):
            os.remove(temp_filename)
    else:
        print("Error during ffmpeg re-encoding. Keeping original file.")
        os.rename(temp_filename, output_filename)

output_video_path = os.path.join(script_dir, 'demo_output_video.mp4')
create_labeled_video(current_sample, df_bb, output_video_path, max_frames=300)

# %%

if not df_ts.empty and 'timestamp' in df_ts.columns:
    # timestamps are in microseconds
    timestamps = df_ts['timestamp'].sort_values()
    diffs = timestamps.diff().dropna()
    
    # Calculate statistics
    mean_delta_us = diffs.mean()
    fps_estimate = 1_000_000 / mean_delta_us
    
    print(f"\nTimestamp Statistics:")
    print(f"  Mean Delta: {mean_delta_us:.2f} µs")
    print(f"  Estimated FPS: {fps_estimate:.2f}")
    print(f"  Min Delta: {diffs.min()} µs")
    print(f"  Max Delta: {diffs.max()} µs")
    
    # Plot in milliseconds for easier reading
    plt.figure(figsize=(10, 4))
    plt.plot(diffs.values / 1000.0, alpha=0.8)
    plt.axhline(y=mean_delta_us/1000.0, color='r', linestyle='--', alpha=0.5, label=f'Mean: {mean_delta_us/1000:.2f}ms')
    plt.title(f"Frame Interval (Diffs)\nEst. FPS: {fps_estimate:.2f}")
    plt.xlabel("Frame Sequence")
    plt.ylabel("Delta (milliseconds)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    ts_plot_path = os.path.join(script_dir, 'timestamp_plot.png')
    plt.savefig(ts_plot_path)
    print(f"Saved timestamp plot to: {ts_plot_path}")
    plt.show()
