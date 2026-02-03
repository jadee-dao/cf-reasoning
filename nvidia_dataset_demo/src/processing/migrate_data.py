import os
import shutil
import glob

BASE_DIR = "/home/jadelynn/cf-reasoning/nvidia_dataset_demo/extracted_data"
DEST_RAW = os.path.join(BASE_DIR, "raw_data/nvidia_demo")
DEST_CALIB = os.path.join(BASE_DIR, "nvidia_demo/calibration")

# Move Raw Files
extensions = ["*.mp4", "*.parquet"]
for ext in extensions:
    files = glob.glob(os.path.join(BASE_DIR, ext))
    for f in files:
        try:
            shutil.move(f, DEST_RAW)
            print(f"Moved {os.path.basename(f)} to raw_data")
        except Exception as e:
            print(f"Error moving {f}: {e}")

# Move Calibration Files
calib_src = os.path.join(BASE_DIR, "calibration_set")
if os.path.exists(calib_src):
    files = glob.glob(os.path.join(calib_src, "*"))
    for f in files:
        try:
            shutil.move(f, DEST_CALIB)
            print(f"Moved {os.path.basename(f)} to calibration")
        except Exception as e:
            print(f"Error moving {f}: {e}")
    # Remove empty dir
    try:
        os.rmdir(calib_src)
    except:
        pass
