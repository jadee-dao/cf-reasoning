
import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from src.processing.loaders import NuScenesDataset
from src.processing.inputs import VideoLoader, ImageLoader
from src.models.factory import create_model
from src.training.train import train_model

def main():
    parser = argparse.ArgumentParser(description="Train Baseline Video Classifier")
    parser.add_argument('--data_dir', type=str, required=True, help='Path to video samples')
    parser.add_argument('--json_path', type=str, required=True, help='Path to labels JSON')
    parser.add_argument('--npy_path', type=str, required=True, help='Path to scores NPY')
    parser.add_argument('--target_type', type=str, default='p90', choices=['p90', 'p99', 'score', 'log_score', 'bin_class'], help='Target to predict')
    parser.add_argument('--modality', type=str, default='video', choices=['video', 'image'], help='Input modality')
    parser.add_argument('--backbone', type=str, default='default', help='Backbone architecture')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size')
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--output_dir', type=str, default='checkpoints', help='Directory to save checkpoints')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', help='Device to run')
    
    args = parser.parse_args()
    
    # Select Input Loader Strategy based on modality
    if args.modality == 'video':
        input_loader = VideoLoader()
    elif args.modality == 'image':
        input_loader = ImageLoader()
    else:
        raise ValueError(f"Unknown modality: {args.modality}")
    
    # Dataset and Loaders
    train_dataset = NuScenesDataset(
        data_dir=args.data_dir,
        json_path=args.json_path,
        npy_path=args.npy_path,
        target_type=args.target_type,
        mode='train',
        input_loader=input_loader
    )
    
    val_dataset = NuScenesDataset(
        data_dir=args.data_dir,
        json_path=args.json_path,
        npy_path=args.npy_path,
        target_type=args.target_type,
        mode='val',
        input_loader=input_loader
    )
    
    # Weighted Sampler for training
    # For bin_class and log_score, we still want to prioritize high scores
    targets = []
    for x in train_dataset.sample_list:
        sid, didx = x['scene_id'], x['data_idx']
        score = train_dataset.scores_map.get((sid, didx), 0.0)
        targets.append(score)
    
    targets = torch.tensor(targets)
    weights = targets + 10.0 # Proportional to raw score
    num_samples = len(targets)
    
    from torch.utils.data import WeightedRandomSampler
    sampler = WeightedRandomSampler(weights, num_samples)
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, sampler=sampler, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    
    # Model Configuration
    if args.target_type == 'bin_class':
        num_classes = 10 # Deciles (0-9)
        criterion = nn.CrossEntropyLoss()
        print(f"Using CrossEntropyLoss for {num_classes}-class classification (Deciles).")
    else:
        num_classes = 1
        criterion = nn.MSELoss()
        print(f"Using MSELoss for regression (Target: {args.target_type}).")
    
    # Create model via Factory
    model = create_model(modality=input_loader.modality, backbone=args.backbone, num_classes=num_classes)
    model.to(args.device)
        
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4) # Added weight decay
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.1)  # Reduce LR every 2 epochs
    
    # Train
    train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler, # Pass scheduler
        num_epochs=args.epochs,
        device=args.device,
        save_dir=args.output_dir
    )

if __name__ == '__main__':
    main()
