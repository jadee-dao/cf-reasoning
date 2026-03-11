import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATv2Conv, TransformerConv, GINConv, global_mean_pool, global_max_pool

class SimpleConvolutionalGNN(nn.Module):
    """
    A simple Convolutional GNN for graph-level regression (ADE prediction).
    """
    def __init__(self, node_in_channels, hidden_channels, out_channels=1):
        super(SimpleConvolutionalGNN, self).__init__()
        
        # Graph convolution layers
        self.conv1 = GCNConv(node_in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.conv3 = GCNConv(hidden_channels, hidden_channels)
        
        # Regression head
        self.fc1 = nn.Linear(hidden_channels * 2, hidden_channels) # concat mean and max pool
        self.fc2 = nn.Linear(hidden_channels, out_channels)
        
        self.dropout = nn.Dropout(p=0.2)

    def forward(self, x, edge_index, batch):
        # 1. Node embeddings
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.dropout(x)
        
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = self.dropout(x)
        
        x = self.conv3(x, edge_index)
        x = F.relu(x)
        
        # 2. Graph-level pooling
        # Using both mean and max pooling to capture different graph statistics
        x_mean = global_mean_pool(x, batch)
        x_max = global_max_pool(x, batch)
        x_pooled = torch.cat([x_mean, x_max], dim=1)
        
        # 3. Final regression
        x = F.relu(self.fc1(x_pooled))
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x

class AttentionGNN(nn.Module):
    """
    Graph Attention Network (GATv2) for ADE Outlier detection.
    Uses multi-head attention to focus on critical interaction nodes.
    """
    def __init__(self, node_in_channels, hidden_channels, num_heads=4, out_channels=1):
        super(AttentionGNN, self).__init__()
        
        # heads * out_channels = total hidden dim
        self.conv1 = GATv2Conv(node_in_channels, hidden_channels // num_heads, heads=num_heads)
        self.conv2 = GATv2Conv(hidden_channels, hidden_channels // num_heads, heads=num_heads)
        self.conv3 = GATv2Conv(hidden_channels, hidden_channels // num_heads, heads=num_heads)
        
        # Regression/Classification head
        self.fc1 = nn.Linear(hidden_channels * 2, hidden_channels)
        self.fc2 = nn.Linear(hidden_channels, out_channels)
        
        self.dropout = nn.Dropout(p=0.2)

    def forward(self, x, edge_index, batch):
        # 1. Attention Layers
        x = F.elu(self.conv1(x, edge_index))
        x = self.dropout(x)
        
        x = F.elu(self.conv2(x, edge_index))
        x = self.dropout(x)
        
        x = F.elu(self.conv3(x, edge_index))
        
        # 2. Hybrid Pooling
        x_mean = global_mean_pool(x, batch)
        x_max = global_max_pool(x, batch)
        x_pooled = torch.cat([x_mean, x_max], dim=1)
        
        # 3. Final Prediction
        x = F.relu(self.fc1(x_pooled))
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x

class TransformerGNN(nn.Module):
    """
    Graph Transformer Network for ADE Outlier detection.
    Uses TransformerConv for all-to-all attention if edges are provided.
    """
    def __init__(self, node_in_channels, hidden_channels, num_heads=4, out_channels=1):
        super(TransformerGNN, self).__init__()
        
        self.conv1 = TransformerConv(node_in_channels, hidden_channels // num_heads, heads=num_heads, dropout=0.1)
        self.conv2 = TransformerConv(hidden_channels, hidden_channels // num_heads, heads=num_heads, dropout=0.1)
        
        self.fc1 = nn.Linear(hidden_channels * 2, hidden_channels)
        self.fc2 = nn.Linear(hidden_channels, out_channels)
        self.dropout = nn.Dropout(0.2)

    def forward(self, x, edge_index, batch):
        x = F.elu(self.conv1(x, edge_index))
        x = self.dropout(x)
        x = F.elu(self.conv2(x, edge_index))
        
        x_mean = global_mean_pool(x, batch)
        x_max = global_max_pool(x, batch)
        x_pooled = torch.cat([x_mean, x_max], dim=1)
        
        x = F.relu(self.fc1(x_pooled))
        x = self.fc2(x)
        return x

class GINOutlierGNN(nn.Module):
    """
    Graph Isomorphism Network (GIN) for graph-level classification.
    Known for high discriminative power.
    """
    def __init__(self, node_in_channels, hidden_channels, out_channels=1):
        super(GINOutlierGNN, self).__init__()
        
        nn1 = nn.Sequential(nn.Linear(node_in_channels, hidden_channels), nn.ReLU(), nn.Linear(hidden_channels, hidden_channels))
        self.conv1 = GINConv(nn1)
        
        nn2 = nn.Sequential(nn.Linear(hidden_channels, hidden_channels), nn.ReLU(), nn.Linear(hidden_channels, hidden_channels))
        self.conv2 = GINConv(nn2)
        
        self.fc1 = nn.Linear(hidden_channels * 2, hidden_channels)
        self.fc2 = nn.Linear(hidden_channels, out_channels)

    def forward(self, x, edge_index, batch):
        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))
        
        x_mean = global_mean_pool(x, batch)
        x_max = global_max_pool(x, batch)
        x_pooled = torch.cat([x_mean, x_max], dim=1)
        
        x = F.relu(self.fc1(x_pooled))
        x = self.fc2(x)
        return x
