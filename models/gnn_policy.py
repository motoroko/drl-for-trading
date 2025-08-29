import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool

class GNNPolicy(nn.Module):
    def __init__(self, node_feature_dim, action_dim, hidden_dim=128):
        super().__init__()
        self.conv1 = GCNConv(node_feature_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)

        self.action_head = nn.Linear(hidden_dim, action_dim)
        self.log_std = nn.Parameter(torch.zeros(action_dim))

        self.value_head = nn.Linear(hidden_dim, 1)

    def forward(self, data):
        # data is torch_geometric.data.Data or batch from DataLoader
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))

        # graph pooling (mean)
        x = global_mean_pool(x, batch)  # shape: [batch_size, hidden_dim]

        action_mean = self.action_head(x)
        value = self.value_head(x)

        return action_mean, self.log_std.exp(), value.squeeze(-1)
