
import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score, mean_squared_error
import numpy as np

from torch.utils.tensorboard import SummaryWriter

def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=10, device='cuda', save_dir='checkpoints'):
    """
    Train loop.
    """
    os.makedirs(save_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=os.path.join(save_dir, 'logs'))
    best_val_loss = float('inf')
    
    global_step = 0
    
    for epoch in range(num_epochs):
        print(f"Epoch {epoch+1}/{num_epochs}")
        
        # Train
        model.train()
        running_loss = 0.0
        preds_all = []
        targets_all = []
        
        pbar = tqdm(train_loader, desc="Training")
        for videos, targets in pbar:
            videos = videos.to(device)
            targets = targets.to(device) # (B,) or (B, 1)?
            
            optimizer.zero_grad()
            
            outputs = model(videos) # (B, num_classes) usually (B, 1) for binary/regression here
            outputs = outputs.squeeze(1) # (B,)
            
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            curr_loss = loss.item()
            running_loss += curr_loss * videos.size(0)
            
            # Log step loss
            writer.add_scalar('Train/BatchLoss', curr_loss, global_step)
            global_step += 1
            
            # Store for metrics
            preds_all.extend(outputs.detach().cpu().numpy())
            targets_all.extend(targets.detach().cpu().numpy())
            
            pbar.set_postfix({'loss': curr_loss})
            
        epoch_loss = running_loss / len(train_loader.dataset)
        print(f"Train Loss: {epoch_loss:.4f}")
        writer.add_scalar('Train/EpochLoss', epoch_loss, epoch)
        
        # Validation
        val_loss, val_metrics = validate(model, val_loader, criterion, device, writer=writer, epoch=epoch)
        print(f"Val Loss: {val_loss:.4f} | Metrics: {val_metrics}")
        
        writer.add_scalar('Val/Loss', val_loss, epoch)
        for k, v in val_metrics.items():
             writer.add_scalar(f'Val/{k}', v, epoch)
        
        # Save best
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), os.path.join(save_dir, 'best_model.pth'))
            print("Saved best model.")
            
    print("Training complete.")
    writer.close()

import matplotlib.pyplot as plt

def validate(model, val_loader, criterion, device='cuda', writer=None, epoch=0):
    model.eval()
    running_loss = 0.0
    preds_all = []
    targets_all = []
    
    with torch.no_grad():
        for videos, targets in val_loader:
            videos = videos.to(device)
            targets = targets.to(device)
            
            outputs = model(videos)
            outputs = outputs.squeeze(1)
            
            loss = criterion(outputs, targets)
            running_loss += loss.item() * videos.size(0)
            
            preds_all.extend(outputs.cpu().numpy())
            targets_all.extend(targets.cpu().numpy())
            
    epoch_loss = running_loss / len(val_loader.dataset)
    
    # Metrics & Visualization
    metrics = {}
    
    # Regression specific visualization
    if isinstance(criterion, (nn.MSELoss, nn.L1Loss)):
        metrics['mse'] = mean_squared_error(targets_all, preds_all)
        
        if writer:
            # Histograms
            writer.add_histogram('Val/Predictions', np.array(preds_all), epoch)
            writer.add_histogram('Val/Targets', np.array(targets_all), epoch)
            writer.add_histogram('Val/Errors', np.array(preds_all) - np.array(targets_all), epoch)
            
            # Scatter Plot: Predicted vs Actual
            fig = plt.figure(figsize=(6, 6))
            plt.scatter(targets_all, preds_all, alpha=0.5)
            # Plot diagonal identity line
            lims = [
                np.min([plt.xlim(), plt.ylim()]),  # min of both axes
                np.max([plt.xlim(), plt.ylim()]),  # max of both axes
            ]
            plt.plot(lims, lims, 'r-', alpha=0.75, zorder=0)
            plt.title(f'Predicted vs Actual (Epoch {epoch})')
            plt.xlabel('Actual Score')
            plt.ylabel('Predicted Score')
            plt.grid(True)
            
            writer.add_figure('Val/PredVsActual', fig, epoch)
            plt.close(fig)
            
    elif isinstance(criterion, nn.BCEWithLogitsLoss):
        # Apply sigmoid
        probs = torch.sigmoid(torch.tensor(preds_all)).numpy()
        preds_binary = (probs > 0.5).astype(int)
        targets_binary = (np.array(targets_all) > 0.5).astype(int)
        acc = accuracy_score(targets_binary, preds_binary)
        f1 = f1_score(targets_binary, preds_binary)
        metrics['accuracy'] = acc
        metrics['f1'] = f1
        
        if writer:
             writer.add_histogram('Val/Probabilities', probs, epoch)
    
    return epoch_loss, metrics
