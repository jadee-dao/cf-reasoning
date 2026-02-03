
import os
import argparse
import cv2
import torch
import numpy as np
from src.models.classifier import BaselineVideoClassifier

def load_video(video_path):
    cap = cv2.VideoCapture(video_path)
    frames = []
    if not cap.isOpened():
        raise IOError(f"Could not open video: {video_path}")
        
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame)
    cap.release()
    
    if not frames:
        raise ValueError("Video contains no frames.")
        
    # Preprocess
    video = np.array(frames)
    video = torch.from_numpy(video).float() / 255.0
    video = video.permute(3, 0, 1, 2) # (C, T, H, W)
    # Fixed size transform
    video = torch.nn.functional.interpolate(video.unsqueeze(0), size=(16, 224, 224), mode='trilinear', align_corners=False).squeeze(0)
    return video.unsqueeze(0) # Batch dim

def main():
    parser = argparse.ArgumentParser(description="Run inference on a video.")
    parser.add_argument('--video_path', type=str, required=True, help='Path to video file')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--target_type', type=str, default='p90', choices=['p90', 'p99', 'score'], help='Task type')
    parser.add_argument('--backbone', type=str, default='r3d_18', choices=['r3d_18', 'simple_cnn'], help='Backbone architecture')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', help='Device')
    
    args = parser.parse_args()
    
    # Load Model
    model = BaselineVideoClassifier(backbone_name=args.backbone, num_classes=1)
    # Usually checkpoints contain state_dict
    try:
        state_dict = torch.load(args.checkpoint, map_location=args.device)
        model.load_state_dict(state_dict)
    except Exception as e:
        print(f"Error loading checkpoint: {e}")
        return

    model.to(args.device)
    model.eval()
    
    # Load Data
    try:
        video_tensor = load_video(args.video_path)
        video_tensor = video_tensor.to(args.device)
    except Exception as e:
        print(f"Error loading video: {e}")
        return
        
    # Inference
    with torch.no_grad():
        output = model(video_tensor)
        score = output.item()
        
    print(f"Prediction for {args.video_path}:")
    if args.target_type == 'score':
        print(f"  Score: {score:.4f}")
    else:
        # Logistic probability
        prob = torch.sigmoid(torch.tensor(score)).item()
        label = prob > 0.5
        print(f"  Probability ({args.target_type}): {prob:.4f}")
        print(f"  Classification: {'Outlier' if label else 'Normal'}")

if __name__ == '__main__':
    main()
