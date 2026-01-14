# %%
import os
import glob
import cv2
import pandas as pd
from ultralytics import YOLO

# Determine script directory
script_dir = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(script_dir, '../extracted_data')

def get_samples(data_dir):
    mp4_files = glob.glob(os.path.join(data_dir, "*.mp4"))
    samples = []
    for f in mp4_files:
        base_name = os.path.basename(f).replace('.mp4', '')
        samples.append({
            'id': base_name,
            'video_path': f,
            'blurred_boxes_path': os.path.join(data_dir, f"{base_name}.blurred_boxes.parquet")
        })
    samples.sort(key=lambda x: x['id'])
    return samples

# %%
# Load Segmentation Model
# 'yolo11n-seg.pt' is the Nano Segmentation model
print("Loading YOLO11 Segmentation model...")
model = YOLO('yolo11n-seg.pt') 

# Get a sample
samples = get_samples(DATA_DIR)
if not samples:
    print("No samples found.")
    exit()

# Pick the same sample (index 9)
sample = samples[9] if len(samples) > 9 else samples[0]
print(f"Processing Sample: {sample['id']}")

# Load GT Data (for reference if needed, though masks explain themselves)
if os.path.exists(sample['blurred_boxes_path']):
    df_bb = pd.read_parquet(sample['blurred_boxes_path'])
else:
    df_bb = pd.DataFrame()

# %%
import numpy as np
from collections import defaultdict

# History cache: track_id -> list of centroids (x,y)
track_history = defaultdict(lambda: [])

# Configuration
HISTORY_LENGTH = 15 # Frames to look back for heading
MIN_DISPLACEMENT = 5 # Minimum pixels moved to show heading

def create_seg_video(sample, df_bboxes, output_filename, max_frames=300):
    cap = cv2.VideoCapture(sample['video_path'])
    if not cap.isOpened():
        print("Error: Could not open video.")
        return

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_filename, fourcc, fps, (width, height))
    
    frame_idx = 0
    print(f"Running TRACKING & SEGMENTATION on {max_frames} frames...")
    
    while cap.isOpened() and frame_idx < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Run YOLO Tracking (persist=True is critical for tracking)
        results = model.track(frame, persist=True, verbose=False)
        
        if results[0].boxes and results[0].masks and results[0].boxes.id is not None:
            # Get track IDs and masks
            track_ids = results[0].boxes.id.int().cpu().tolist()
            masks = results[0].masks.xy # List of polygon points
            
            # Annotate manually to add arrows
            annotated_frame = results[0].plot()
            
            for track_id, mask in zip(track_ids, masks):
                # Calculate Centroid
                if len(mask) == 0: continue
                centroid = np.mean(mask, axis=0).astype(int)
                cx, cy = centroid[0], centroid[1]
                
                # Update history
                track_history[track_id].append((cx, cy))
                if len(track_history[track_id]) > HISTORY_LENGTH:
                    track_history[track_id].pop(0)
                
                # Calculate Heading - Smoothed with Linear Regression
                hist = track_history[track_id]
                if len(hist) >= 5: # Need minimum points for a stable fit
                    # Extract X and Y coords
                    xs = [p[0] for p in hist]
                    ys = [p[1] for p in hist]
                    ts = np.arange(len(hist))
                    
                    # Calculate total displacement to gate drawing
                    dist_total = np.sqrt((xs[-1]-xs[0])**2 + (ys[-1]-ys[0])**2)
                    
                    if dist_total > MIN_DISPLACEMENT:
                        # Fit line for X and Y vs Time (t)
                        # x(t) = vx * t + x0
                        # polyfit returns [slope, intercept]
                        fit_x = np.polyfit(ts, xs, 1)
                        fit_y = np.polyfit(ts, ys, 1)
                        
                        vx = fit_x[0]
                        vy = fit_y[0]
                        
                        # Magnitude of velocity vector
                        speed = np.sqrt(vx*vx + vy*vy)
                        
                        if speed > 0.5: # Minimum speed threshold
                            scale = 50 
                            # Draw Arrow
                            nx = int(vx / speed * scale)
                            ny = int(vy / speed * scale)
                            
                            # Cyan Arrow
                            cv2.arrowedLine(annotated_frame, (cx, cy), (cx + nx, cy + ny), 
                                          (255, 255, 0), 3, tipLength=0.3)

        else:
            annotated_frame = frame
        
        cv2.putText(annotated_frame, f"Frame: {frame_idx} (TRACK+HEAD)", (50, 50), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        out.write(annotated_frame)
        frame_idx += 1
        
        if frame_idx % 30 == 0:
            print(f"Processed {frame_idx} frames...")

    cap.release()
    out.release()
    
    # Re-encode to H.264
    temp_filename = output_filename + ".temp.mp4"
    if os.path.exists(output_filename):
        os.rename(output_filename, temp_filename)
        
    print("Re-encoding to H.264...")
    cmd = f"ffmpeg -y -v error -i {temp_filename} -c:v libx264 -pix_fmt yuv420p {output_filename}"
    ret = os.system(cmd)
    
    if ret == 0:
        print(f"Video saved to: {output_filename}")
        if os.path.exists(temp_filename):
            os.remove(temp_filename)
    else:
        print("FFmpeg failed.")
        os.rename(temp_filename, output_filename)

output_video_path = os.path.join(script_dir, 'yolo_heading_video.mp4')
create_seg_video(sample, df_bb, output_video_path, max_frames=300)
