
import sys
import os
import torch
# Add the directory containing the nvidia_dataset_demo package to python path
sys.path.append(os.path.abspath("/home/jadelynn/cf-reasoning"))

try:
    from nvidia_dataset_demo.scripts.embeddings.strategies import FastVLMDescriptionStrategy
except ImportError:
    print("Could not import FastVLMDescriptionStrategy. Adjusting path...")
    sys.path.append(os.path.abspath("/home/jadelynn/cf-reasoning/nvidia_dataset_demo/scripts"))
    from embeddings.strategies import FastVLMDescriptionStrategy

def test_fastvlm():
    print("Initializing FastVLMDescriptionStrategy...")
    strategy = FastVLMDescriptionStrategy()
    
    # Test 1: Image Input
    image_path = "/home/jadelynn/cf-reasoning/dummy_image.jpg"
    print(f"\nTesting with Image: {image_path}")
    if os.path.exists(image_path):
        try:
            # logic to mock or check if manual loop is used? 
            # The current code prints "Using manual greedy generation loop..."
            embedding = strategy.generate_embedding(image_path)
            print("Image embedding generated successfully.")
        except Exception as e:
            print(f"Image generation failed: {e}")
    else:
        print(f"Image file not found: {image_path}")

    # Test 2: Video Input
    video_path = "/home/jadelynn/cf-reasoning/nvidia_dataset_demo/extracted_data/01d3588e-bca7-4a18-8e74-c6cfe9e996db.camera_front_wide_120fov.mp4"
    print(f"\nTesting with Video: {video_path}")
    if os.path.exists(video_path):
        try:
            # This is expected to fail currently as Image.open won't open mp4
            embedding = strategy.generate_embedding(video_path)
            print("Video embedding generated successfully (Unexpected).")
        except Exception as e:
            print(f"Video generation failed as expected (Current behavior): {e}")
    else:
        print(f"Video file not found: {video_path}")

if __name__ == "__main__":
    test_fastvlm()
