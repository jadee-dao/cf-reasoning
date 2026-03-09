import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import json
import os
import glob
import pickle
from tqdm import tqdm
import argparse

# --- Model Definitions ---

# --- Model Definitions ---

class GCNLayer(nn.Module):
    def __init__(self, in_features, out_features):
        super(GCNLayer, self).__init__()
        self.linear = nn.Linear(in_features, out_features)

    def forward(self, x, adj):
        deg = torch.sum(adj, dim=1)
        deg_inv_sqrt = torch.pow(deg + 1e-6, -0.5)
        adj_norm = deg_inv_sqrt.view(-1, 1) * adj * deg_inv_sqrt.view(1, -1)
        out = torch.mm(adj_norm, x)
        out = self.linear(out)
        return out

class GraphAutoencoder(nn.Module):
    def __init__(self, node_feat_dim, hidden_dim, latent_dim):
        super(GraphAutoencoder, self).__init__()
        self.gcn1 = GCNLayer(node_feat_dim, hidden_dim)
        self.gcn2 = GCNLayer(hidden_dim, latent_dim)
        self.decoder_feat = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, node_feat_dim)
        )
        
    def encode(self, x, adj):
        h = F.relu(self.gcn1(x, adj))
        z = self.gcn2(h, adj)
        return z

    def forward(self, x, adj):
        z = self.encode(x, adj)
        adj_recon = torch.sigmoid(torch.mm(z, z.t()))
        x_recon = self.decoder_feat(z)
        return adj_recon, x_recon, z

class GNNClassifier(nn.Module):
    def __init__(self, node_feat_dim, hidden_dim, out_dim=1):
        super(GNNClassifier, self).__init__()
        self.gcn1 = GCNLayer(node_feat_dim, hidden_dim)
        self.gcn2 = GCNLayer(hidden_dim, hidden_dim)
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, out_dim)
        )
        
    def forward(self, x, adj):
        h = F.relu(self.gcn1(x, adj))
        h = F.relu(self.gcn2(h, adj))
        # Global Pooling (Mean)
        x_pooled = torch.mean(h, dim=0)
        out = torch.sigmoid(self.head(x_pooled))
        return out

# --- Data Loading ---

def load_graph_json(path):
    with open(path, 'r') as f:
        data = json.load(f)
    nodes, edges = data['nodes'], data['edges']
    node_feat = []
    for n in nodes:
        feat = n['features'] + [n['depth'], n.get('class', -1), n.get('area', 0)]
        node_feat.append(feat)
    node_feat = torch.tensor(node_feat, dtype=torch.float32)
    N = len(nodes)
    adj = torch.eye(N)
    node_id_to_idx = {n['id']: i for i, n in enumerate(nodes)}
    for e in edges:
        u, v = node_id_to_idx[e['source']], node_id_to_idx[e['target']]
        adj[u, v] = adj[v, u] = 1.0
    return node_feat, adj

def load_labels():
    # 1. ASIL Labels
    asil_path = "analysis_results/ground_truth/asil_scores_nvidia_demo.json"
    with open(asil_path, 'r') as f:
        asil_data = json.load(f)
    
    # 2. ADE Labels
    ade_path = "extracted_data/nvidia_demo/calibration/worst-ade-log-10-90pctl-filtered.json"
    ade_outliers = set()
    if os.path.exists(ade_path):
        with open(ade_path, 'r') as f:
            ade_data = json.load(f)
        for scene_val in ade_data.get("results", {}).values():
            scene_id = scene_val["scene_id"]
            for item in scene_val.get("top3_worst", []):
                ade_outliers.add(f"{scene_id}_{item['t_rel_us']}")
                
    return asil_data, ade_outliers

# --- Main Script ---

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["gae", "classifier"], default="gae")
    parser.add_argument("--target", choices=["ade", "asil_a", "asil_b"], default="ade")
    parser.add_argument("--dataset", default="nvidia_demo")
    parser.add_argument("--epochs", type=int, default=30)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Mode: {args.mode}, Target: {args.target}, Device: {device}")

    # 1. Collect Graphs & Labels
    graph_paths = glob.glob(f"analysis_results/visualize_samples/object_graph/{args.dataset}/*_graph.json")
    asil_data, ade_outliers = load_labels()
    
    # Map labels to paths
    data_list = []
    for p in graph_paths:
        sample_id = os.path.basename(p).replace("_graph.json", "")
        # Build binary label for classification
        label = 0
        if args.target == "ade":
            label = 1 if sample_id in ade_outliers else 0
        elif args.target == "asil_a":
            label = 1 if asil_data.get(sample_id, {}).get("asil_level") == "A" else 0
        elif args.target == "asil_b":
            label = 1 if asil_data.get(sample_id, {}).get("asil_level") == "B" else 0
        
        # Hazard check for GAE (only train on truly normal scenes)
        is_hazard = (sample_id in ade_outliers) or (asil_data.get(sample_id, {}).get("asil_score", 0) > 0)
        
        data_list.append({"path": p, "id": sample_id, "label": label, "is_hazard": is_hazard})

    # 2. Train/Test Split
    np.random.seed(42)
    np.random.shuffle(data_list)
    split_idx = int(0.7 * len(data_list))
    train_pool = data_list[:split_idx]
    test_set = data_list[split_idx:]
    
    if args.mode == "gae":
        train_set = [d for d in train_pool if not d["is_hazard"]]
        model = GraphAutoencoder(387, 128, 64).to(device)
    else:
        train_set = train_pool # Classifiers need mixed data
        model = GNNClassifier(387, 128).to(device)
        
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    # 3. Training
    print(f"Training on {len(train_set)} samples...")
    for epoch in range(args.epochs):
        model.train()
        total_loss = 0
        for d in tqdm(train_set, desc=f"Epoch {epoch+1}", disable=True):
            try:
                x, adj = load_graph_json(d["path"])
                x, adj = x.to(device), adj.to(device)
                optimizer.zero_grad()
                
                if args.mode == "gae":
                    adj_r, x_r, _ = model(x, adj)
                    loss = F.mse_loss(adj_r, adj) + F.mse_loss(x_r, x)
                else:
                    pred = model(x, adj)
                    target = torch.tensor([d["label"]], dtype=torch.float32).to(device)
                    loss = F.binary_cross_entropy(pred.view(-1), target.view(-1))
                    
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            except Exception as e:
                print(f"Error in training sample {d['id']}: {e}")
                continue
        if (epoch+1) % 5 == 0:
            print(f"Epoch {epoch+1}, Avg Loss: {total_loss/len(train_set):.6f}")

    # 4. Evaluation & Export
    model.eval()
    results = {}
    print(f"Evaluating {len(test_set)} test samples...")
    for d in tqdm(test_set):
        try:
            x, adj = load_graph_json(d["path"])
            x, adj = x.to(device), adj.to(device)
            with torch.no_grad():
                if args.mode == "gae":
                    adj_r, x_r, _ = model(x, adj)
                    score = (F.mse_loss(adj_r, adj) + F.mse_loss(x_r, x)).item()
                else:
                    score = model(x, adj).item()
            results[d["id"]] = {"score": score, "label": d["label"]}
        except: continue

    out_file = f"analysis_results/outliers/gnn_nvidia_{args.mode}_{args.target}.json"
    os.makedirs(os.path.dirname(out_file), exist_ok=True)
    with open(out_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved results to {out_file}")

if __name__ == "__main__":
    main()
