import os
import json
import glob
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, recall_score, precision_score, f1_score
from torch_geometric.loader import DataLoader
from torch_geometric.data import Data
import sys
from tqdm import tqdm

# Ensure we can import modules from src
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.join(BASE_DIR, "../src")
if SRC_DIR not in sys.path:
    sys.path.append(SRC_DIR)

from models.gnn import SimpleConvolutionalGNN, AttentionGNN, TransformerGNN, GINOutlierGNN

def load_ade_lookup(calibration_dir, shards, threshold=0.784):
    """Loads ADE ground truth and converts to binary labels."""
    label_lookup = {}
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
                    # Binary Label: 1 if ADE > threshold (outlier) else 0
                    label_lookup[key] = 1.0 if ade > threshold else 0.0
                except:
                    continue
    return label_lookup

def process_graph_json(json_path, label, fully_connected=False):
    """Converts a graph JSON to a torch_geometric Data object for classification."""
    with open(json_path, 'r') as f:
        data = json.load(f)
        
    nodes = data["nodes"]
    edges = data["edges"]
    
    node_features = []
    id_to_idx = {node["id"]: i for i, node in enumerate(nodes)}
    
    for node in nodes:
        # Features: [visual (384) | pos (2) | depth (1) | area (1) | type (1) | class (1)]
        vis = np.array(node["features"])
        pos = np.array(node["pos"])
        depth = np.array([node["depth"]])
        area = np.array([node.get("area", 0.0)])
        node_type = np.array([1.0 if node["type"] == "object" else 0.0])
        node_class = np.array([float(node.get("class", -1)) / 80.0])
        
        feat = np.concatenate([vis, pos, depth, area, node_type, node_class])
        node_features.append(feat)
        
    x = torch.tensor(np.array(node_features), dtype=torch.float)
    
    edge_index = []
    if fully_connected:
        # Every node connected to every other node
        for i in range(len(nodes)):
            for j in range(len(nodes)):
                if i != j:
                    edge_index.append([i, j])
    else:
        for edge in edges:
            u = id_to_idx[edge["source"]]
            v = id_to_idx[edge["target"]]
            edge_index.append([u, v])
            # Ensure bidirectional
            edge_index.append([v, u])
        
    if not edge_index:
        edge_index = torch.empty((2, 0), dtype=torch.long)
    else:
        # Remove duplicates and convert to tensor
        unique_edges = list(set([tuple(sorted(e)) for e in edge_index]))
        # We want directed edges for GAT/Transformer if bidirectional, 
        # but usually Geometric handles edge_index as directed.
        # Let's just keep them as provided (bidirectionalized above).
        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
        
    y = torch.tensor([[label]], dtype=torch.float)
    
    return Data(x=x, edge_index=edge_index, y=y)

def main():
    parser = argparse.ArgumentParser(description="Train GNN for ADE Outlier Classification")
    parser.add_argument("--shards", type=str, nargs="+", default=["shard_00096", "shard_00097", "shard_00098"])
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--threshold", type=float, default=0.784, help="ADE threshold for outliers (90th pctl)")
    parser.add_argument("--device", type=str, default=None, help="Device to use (e.g., cuda:1)")
    parser.add_argument("--model_type", type=str, choices=["gcn", "gat", "transformer", "gin"], default="gcn")
    parser.add_argument("--fully_connected", action="store_true", help="Use fully connected graph")
    args = parser.parse_args()
    
    # 1. Device Selection
    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Training on {device} (Threshold: {args.threshold})")
    
    # 2. Paths
    data_root = os.path.join(BASE_DIR, "../extracted_data/nvidia_demo")
    calibration_dir = os.path.join(data_root, "calibration")
    graph_root = os.path.join(data_root, "object_graphs")
    
    # 3. Load Outlier labels
    print("Loading ADE ground truth and applying thresholds...")
    label_lookup = load_ade_lookup(calibration_dir, args.shards, args.threshold)
    print(f"Loaded labels for {len(label_lookup)} samples. Positives: {sum(label_lookup.values())}")
    
    # 4. Load Graphs
    dataset = []
    print("Processing graphs...")
    for shard in args.shards:
        shard_graph_dir = os.path.join(graph_root, shard)
        if not os.path.exists(shard_graph_dir):
            continue
            
        graph_files = glob.glob(os.path.join(shard_graph_dir, "*_graph.json"))
        for gf in tqdm(graph_files, desc=f"Loading {shard}"):
            sample_id = os.path.basename(gf).replace("_graph.json", "")
            if sample_id in label_lookup:
                try:
                    data = process_graph_json(gf, label_lookup[sample_id], fully_connected=args.fully_connected)
                    dataset.append(data)
                except:
                    pass
                    
    print(f"Total valid graph samples: {len(dataset)}")
    if not dataset:
        print("No samples found. Exiting.")
        return
        
    # 5. Split and DataLoader
    np.random.seed(42)
    indices = np.random.permutation(len(dataset))
    num_samples = len(dataset)
    split_idx = int(num_samples * 0.8)
    
    train_dataset = [dataset[i] for i in indices[:split_idx]]
    val_dataset = [dataset[i] for i in indices[split_idx:]]
    
    # Handle Class Imbalance (Simple weighting)
    num_pos = sum([d.y.item() for d in train_dataset])
    pos_weight = torch.tensor([(len(train_dataset) - num_pos) / num_pos], device=device)
    print(f"Positive weight for loss: {pos_weight.item():.2f}")
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size)
    
    # 6. Initialize Model
    in_channels = dataset[0].num_node_features
    if args.model_type == "gat":
        print(f"Initializing AttentionGNN (GATv2)...")
        model = AttentionGNN(node_in_channels=in_channels, hidden_channels=128).to(device)
    elif args.model_type == "transformer":
        print(f"Initializing TransformerGNN...")
        model = TransformerGNN(node_in_channels=in_channels, hidden_channels=128).to(device)
    elif args.model_type == "gin":
        print(f"Initializing GINOutlierGNN...")
        model = GINOutlierGNN(node_in_channels=in_channels, hidden_channels=128).to(device)
    else:
        print(f"Initializing SimpleConvolutionalGNN (GCN)...")
        model = SimpleConvolutionalGNN(node_in_channels=in_channels, hidden_channels=128).to(device)
        
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    # Binary Cross Entropy with Logits (includes sigmoid)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    
    history = {'train_loss': [], 'val_loss': [], 'val_auc': []}
    
    # 7. Training Loop
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
        all_probs = []
        all_labels = []
        with torch.no_grad():
            for data in val_loader:
                data = data.to(device)
                out = model(data.x, data.edge_index, data.batch)
                loss = criterion(out, data.y)
                total_val_loss += loss.item() * data.num_graphs
                # Probabilities via sigmoid
                probs = torch.sigmoid(out).cpu().numpy()
                all_probs.append(probs)
                all_labels.append(data.y.cpu().numpy())
        
        val_loss = total_val_loss / len(val_dataset)
        all_probs = np.concatenate(all_probs)
        all_labels = np.concatenate(all_labels)
        
        # Calculate AUC
        fpr, tpr, _ = roc_curve(all_labels, all_probs)
        roc_auc = auc(fpr, tpr)
        
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['val_auc'].append(roc_auc)
        
        print(f"Epoch {epoch+1:03d}: Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val AUC: {roc_auc:.4f}")
        
    # Final Metrics
    preds = (all_probs > 0.5).astype(float)
    precision = precision_score(all_labels, preds)
    recall = recall_score(all_labels, preds)
    f1 = f1_score(all_labels, preds)
    print(f"\nFinal Metrics on Val Set:\nPrecision: {precision:.4f}\nRecall: {recall:.4f}\nF1: {f1:.4f}\nAUC: {roc_auc:.4f}")

    # 8. Plotting
    vis_dir = os.path.join(BASE_DIR, "../analysis_results/plots")
    os.makedirs(vis_dir, exist_ok=True)
    
    plt.figure(figsize=(15, 5))
    
    # 1. Loss and AUC curves
    plt.subplot(1, 3, 1)
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Val Loss')
    plt.title('Loss')
    plt.xlabel('Epoch')
    plt.legend()
    
    plt.subplot(1, 3, 2)
    plt.plot(history['val_auc'], label='Val AUC', color='green')
    plt.title('AUC-ROC over Epochs')
    plt.xlabel('Epoch')
    plt.legend()
    
    # 2. ROC Curve
    plt.subplot(1, 3, 3)
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    
    plt.tight_layout()
    plot_path = os.path.join(vis_dir, f"gnn_classifier_roc_{args.model_type}.png")
    plt.savefig(plot_path)
    print(f"\nplots saved to {plot_path}")
    
    # Save Model
    model_path = os.path.join(BASE_DIR, f"../models/gnn_outlier_model_{args.model_type}.pth")
    torch.save(model.state_dict(), model_path)
    print(f"Model saved to {model_path}")

if __name__ == "__main__":
    main()
