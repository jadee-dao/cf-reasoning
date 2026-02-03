
import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from src.processing.loaders import NuScenesDataset
from src.models.classifier import BaselineVideoClassifier
from src.training.train import train_model

def main():
    parser = argparse.ArgumentParser(description="Train Baseline Video Classifier")
    parser.add_argument('--data_dir', type=str, required=True, help='Path to video samples')
    parser.add_argument('--json_path', type=str, required=True, help='Path to labels JSON')
    parser.add_argument('--npy_path', type=str, required=True, help='Path to scores NPY')
    parser.add_argument('--target_type', type=str, default='p90', choices=['p90', 'p99', 'score'], help='Target to predict')
    parser.add_argument('--backbone', type=str, default='r3d_18', choices=['r3d_18', 'simple_cnn'], help='Backbone architecture')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size')
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--output_dir', type=str, default='checkpoints', help='Directory to save checkpoints')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', help='Device to run one')
    
    args = parser.parse_args()
    
    # Dataset
    train_dataset = NuScenesDataset(
        data_dir=args.data_dir,
        json_path=args.json_path,
        npy_path=args.npy_path,
        target_type=args.target_type,
        mode='train'
    )
    
    val_dataset = NuScenesDataset(
        data_dir=args.data_dir,
        json_path=args.json_path,
        npy_path=args.npy_path,
        target_type=args.target_type,
        mode='val'
    )
    
    # Weighted Sampler for training
    # We want to sample high-score events more often to combat mean collapse
    targets = []
    for x in train_dataset.sample_list:
        sid, didx = x['scene_id'], x['data_idx']
        score = train_dataset.scores_map.get((sid, didx), 0.0)
        targets.append(score)
    
    targets = torch.tensor(targets)
    # Weights directly proportional to score (plus small epsilon)
    # Norm factor is already applied in dataset, but here we work with raw scores or just relative
    # Let's say weight = score + 10.0 (to ensure even 0-score items get picked sometimes)
    weights = targets + 10.0
    num_samples = len(targets)
    
    from torch.utils.data import WeightedRandomSampler
    sampler = WeightedRandomSampler(weights, num_samples)
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, sampler=sampler, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    
    # Model
    num_classes = 1 # Regression or Binary Classification
    model = BaselineVideoClassifier(backbone_name=args.backbone, num_classes=num_classes)
    model.to(args.device)
    
    # Loss and Optimizer
    if args.target_type in ['p90', 'p99']:
        print("Using BCEWithLogitsLoss for classification.")
        criterion = nn.BCEWithLogitsLoss()
    else:
        print("Using MSELoss for regression.")
        criterion = nn.MSELoss()
        
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    
    # Train
    train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        num_epochs=args.epochs,
        device=args.device,
        save_dir=args.output_dir
    )

if __name__ == '__main__':
    main()
