import os
import glob
import pandas as pd
import cv2

# Determine the directory of this script
script_dir = os.path.dirname(os.path.abspath(__file__))
# Data is in the parent directory under 'extracted_data'
DATA_DIR = os.path.join(script_dir, '../extracted_data')

def inspect_data():
    # Find first MP4
    mp4_files = glob.glob(os.path.join(DATA_DIR, "*.mp4"))
    if not mp4_files:
        print("No MP4 files found.")
        return

    sample_mp4 = mp4_files[0]
    base_name = os.path.basename(sample_mp4).replace('.mp4', '')
    print(f"Inspecting sample: {base_name}")

    # Inspect MP4
    cap = cv2.VideoCapture(sample_mp4)
    if cap.isOpened():
        width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        duration = frame_count / fps if fps > 0 else 0
        print(f"\nVideo Info:")
        print(f"  Resolution: {int(width)}x{int(height)}")
        print(f"  FPS: {fps}")
        print(f"  Frame Count: {frame_count}")
        print(f"  Duration: {duration:.2f}s")
        cap.release()
    else:
        print("Failed to open video.")

    # Inspect Timestamps
    ts_file = os.path.join(DATA_DIR, f"{base_name}.timestamps.parquet")
    if os.path.exists(ts_file):
        df_ts = pd.read_parquet(ts_file)
        print(f"\nTimestamps Parquet Info:")
        print(f"  Columns: {list(df_ts.columns)}")
        print(f"  Shape: {df_ts.shape}")
        print(f"  First 3 rows:\n{df_ts.head(3)}")
    else:
        print(f"No timestamps file found for {base_name}")

    # Inspect Blurred Boxes
    bb_file = os.path.join(DATA_DIR, f"{base_name}.blurred_boxes.parquet")
    if os.path.exists(bb_file):
        df_bb = pd.read_parquet(bb_file)
        print(f"\nBlurred Boxes Parquet Info:")
        print(f"  Columns: {list(df_bb.columns)}")
        print(f"  Shape: {df_bb.shape}")
        if not df_bb.empty:
            print(f"  First 3 rows:\n{df_bb.head(3)}")
    else:
        print(f"No blurred boxes file found for {base_name}")

if __name__ == "__main__":
    inspect_data()
