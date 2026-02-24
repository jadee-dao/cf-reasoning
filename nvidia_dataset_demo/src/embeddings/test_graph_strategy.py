import os
import sys
import torch
import numpy as np
from PIL import Image
import cv2

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from src.embeddings.strategies import ObjectGraphStrategy

def test_graph_strategy(image_path):
    print(f"Testing ObjectGraphStrategy with image: {image_path}")
    
    strategy = ObjectGraphStrategy()
    
    # Define debug output paths
    base_dir = "test_outputs"
    os.makedirs(base_dir, exist_ok=True)
    debug_img_path = os.path.join(base_dir, "graph_vis.jpg")
    
    # Generate embedding
    try:
        embedding = strategy.generate_embedding(
            image_path, 
            debug_output_path=debug_img_path
        )
        
        print(f"Embedding generated successfully. Shape: {embedding.shape}")
        print(f"Debug image saved to: {debug_img_path}")
        
        json_path = debug_img_path.replace(".jpg", "_graph.json")
        if os.path.exists(json_path):
            print(f"Graph JSON saved to: {json_path}")
            import json
            with open(json_path, 'r') as f:
                data = json.load(f)
                print(f"Nodes found: {len(data['nodes'])}")
                print(f"Edges found: {len(data['edges'])}")
        else:
            print("Error: Graph JSON was not saved.")
            
    except Exception as e:
        print(f"Error during strategy execution: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    if len(sys.argv) > 1:
        img_path = sys.argv[1]
    else:
        # Default fallback sample
        img_path = "/home/jadelynn/cf-reasoning/nvidia_dataset_demo/extracted_data/nuscenes_ego/samples/scene-0066_11280.jpg"
    
    if os.path.exists(img_path):
        test_graph_strategy(img_path)
    else:
        print(f"Image not found: {img_path}")
