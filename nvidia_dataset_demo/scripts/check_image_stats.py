
import cv2
import numpy as np
import sys
import os

img_path = "extracted_data/nuscenes_ego/samples/scene-0070_11363.jpg"

def main():
    if not os.path.exists(img_path):
        import glob
        # Find it
        matches = glob.glob(f"**/{os.path.basename(img_path)}", recursive=True)
        if matches:
            path = matches[0]
        else:
            print("Image not found.")
            return
    else:
        path = img_path
        
    img = cv2.imread(path)
    if img is None:
        print("Failed to load image.")
        return
        
    # Convert to HSV to check Value (Brightness)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    brightness = hsv[:,:,2].mean()
    
    print(f"Image: {path}")
    print(f"Average Brightness: {brightness:.2f} (0-255)")
    
    if brightness < 40:
        print("Status: Very Dark (Night?)")
    elif brightness < 80:
        print("Status: Dark")
    else:
        print("Status: Normal/Bright")

if __name__ == "__main__":
    main()
