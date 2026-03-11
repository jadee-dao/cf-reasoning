import os
import json
import glob
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GCNConv, global_mean_pool, global_max_pool
import sys
from tqdm import tqdm

# Ensure we can import modules from src
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.join(BASE_DIR, "../src")
if SRC_DIR not in sys.path:
    sys.path.append(SRC_DIR)

from models.gnn import SimpleConvolutionalGNN

def load_ade_lookup(calibration_dir, shards):
    """Loads ADE ground truth from shard .jsonl files."""
    ade_lookup = {}
    for shard in shards:
        shard_path = os.path.join(calibration_dir, f"{shard}.jsonl")
        if not os.path.exists(shard_path):
            print(f"Warning: Shard file {shard_path} not found.")
            continue
            
        with open(shard_path, 'r') as f:
            for line in f:
                try:
                    data = json.loads(line)
                    scene_id = data["scene_id"]
                    timestamp_us = data["timestamp_us"]
                    ade = data.get("ADE", 0.0)
                    key = f"{scene_id}_{timestamp_us}"
                    ade_lookup[key] = ade
                except:
                    continue
    return ade_lookup

def process_graph_json(json_path, target_ade):
    """Converts a graph JSON to a torch_geometric Data object."""
    with open(json_path, 'r') as f:
        data = json.load(f)
        
    nodes = data["nodes"]
    edges = data["edges"]
    
    node_features = []
    # Map node IDs to indices
    id_to_idx = {node["id"]: i for i, node in enumerate(nodes)}
    
    for node in nodes:
        # Features: [visual (384) | pos (2) | depth (1) | area (1) | type (1) | class (1)]
        vis = np.array(node["features"])
        pos = np.array(node["pos"])
        depth = np.array([node["depth"]])
        area = np.array([node.get("area", 0.0)])
        node_type = np.array([1.0 if node["type"] == "object" else 0.0])
        node_class = np.array([float(node.get("class", -1)) / 80.0]) # simple normalization
        
        feat = np.concatenate([vis, pos, depth, area, node_type, node_class])
        node_features.append(feat)
        
    x = torch.tensor(np.array(node_features), dtype=torch.float)
    
    # Edges
    edge_index = []
    for edge in edges:
        u = id_to_idx[edge["source"]]
        v = id_to_idx[edge["target"]]
        edge_index.append([u, v])
        # Add reverse edge for undirected conv if needed, 
        # but GCNConv handles it better if we explicitly add or if graph is directed.
        # Given it's ego-centric, keeping it source -> target (ego) is fine.
        
    if not edge_index:
        # Handling graphs with only ego node
        edge_index = torch.empty((2, 0), dtype=torch.long)
    else:
        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
        
    y = torch.tensor([[target_ade]], dtype=torch.float)
    
    return Data(x=x, edge_index=edge_index, y=y)

def main():
    parser = argparse.ArgumentParser(description="Train GNN for ADE Prediction")
    parser.add_argument("--shards", type=str, nargs="+", default=["shard_00096", "shard_00097", "shard_00098"])
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--device", type=str, default=None, help="Device to use (e.g., cuda:1)")
    args = parser.parse_args()
    
    # 1. Device Selection
    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Training on {device}")
    
    # 1. Paths
    data_root = os.path.join(BASE_DIR, "../extracted_data/nvidia_demo")
    calibration_dir = os.path.join(data_root, "calibration")
    graph_root = os.path.join(data_root, "object_graphs")
    
    # 2. Load ADE labels
    print("Loading ADE ground truth...")
    ade_lookup = load_ade_lookup(calibration_dir, args.shards)
    print(f"Loaded ADE scores for {len(ade_lookup)} samples.")
    
    # 3. Load Graphs
    dataset = []
    print("Processing graphs...")
    for shard in args.shards:
        shard_graph_dir = os.path.join(graph_root, shard)
        if not os.path.exists(shard_graph_dir):
            continue
            
        graph_files = glob.glob(os.path.join(shard_graph_dir, "*_graph.json"))
        for gf in tqdm(graph_files, desc=f"Loading {shard}"):
            sample_id = os.path.basename(gf).replace("_graph.json", "")
            if sample_id in ade_lookup:
                try:
                    data = process_graph_json(gf, ade_lookup[sample_id])
                    dataset.append(data)
                except Exception as e:
                    # print(f"Error processing {gf}: {e}")
                    pass
                    
    print(f"Total valid graph samples: {len(dataset)}")
    if not dataset:
        print("No samples found. Exiting.")
        return
        
    # 4. Split and DataLoader
    num_samples = len(dataset)
    split_idx = int(num_samples * 0.8)
    train_dataset = dataset[:split_idx]
    val_dataset = dataset[split_idx:]
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size)
    
    # 5. Initialize Model
    
    # Node features: 390 (as calculated)
    model = SimpleConvolutionalGNN(node_in_channels=dataset[0].num_node_features, hidden_channels=128).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.MSELoss()
    
    # 6. Training Loop
    history = {'train_loss': [], 'val_loss': [], 'val_r2': []}
    
    for epoch in range(args.epochs):
        model.train()
        total_loss = 0
        for data in train_loader:
            data = data.to(device)
            optimizer.zero_grad()
            out = model(data.x, data.edge_index, data.batch)
            loss = criterion(out, data.y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * data.num_graphs
            
        train_loss = total_loss / len(train_dataset)
        
        # Validation
        model.eval()
        total_val_loss = 0
        all_preds = []
        all_targets = []
        with torch.no_grad():
            for data in val_loader:
                data = data.to(device)
                out = model(data.x, data.edge_index, data.batch)
                loss = criterion(out, data.y)
                total_val_loss += loss.item() * data.num_graphs
                all_preds.append(out.cpu().numpy())
                all_targets.append(data.y.cpu().numpy())
        
        val_loss = total_val_loss / len(val_dataset)
        
        # Calculate R2 Score
        all_preds = np.concatenate(all_preds, axis=0)
        all_targets = np.concatenate(all_targets, axis=0)
        r2 = r2_score(all_targets, all_preds)
        
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['val_r2'].append(r2)
        
        print(f"Epoch {epoch+1:03d}: Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}, Val R2: {r2:.4f}")
        
    # Save Model
    model_dir = os.path.join(BASE_DIR, "../models")
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, "gnn_ade_model.pth")
    torch.save(model.state_dict(), model_path)
    print(f"Model saved to {model_path}")

    # 7. Plotting
    plt.figure(figsize=(12, 5))
    
    # Loss Plot
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Val Loss')
    plt.title('GNN Training Loss (MSE)')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    # R2 Plot
    plt.subplot(1, 2, 2)
    plt.plot(history['val_r2'], label='Val R2', color='green')
    plt.title('Validation R-squared')
    plt.xlabel('Epoch')
    plt.ylabel('R2 Score')
    plt.legend()
    plt.grid(True)
    
    vis_dir = os.path.join(BASE_DIR, "../analysis_results/plots")
    os.makedirs(vis_dir, exist_ok=True)
    plot_path = os.path.join(vis_dir, "gnn_training_curves.png")
    plt.savefig(plot_path)
    print(f"Training curves saved to {plot_path}")

    # Scatter Plot (Pred vs Actual)
    plt.figure(figsize=(8, 8))
    plt.scatter(all_targets, all_preds, alpha=0.3)
    plt.plot([all_targets.min(), all_targets.max()], [all_targets.min(), all_targets.max()], 'r--')
    plt.xlabel('Ground Truth ADE')
    plt.ylabel('Predicted ADE')
    plt.title('GNN ADE Prediction: Actual vs Predicted')
    plt.grid(True)
    scatter_path = os.path.join(vis_dir, "gnn_ade_scatter.png")
    plt.savefig(scatter_path)
    print(f"Scatter plot saved to {scatter_path}")

if __name__ == "__main__":
    main()
