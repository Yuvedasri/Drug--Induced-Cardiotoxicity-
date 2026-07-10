"""
Lightweight Graph Attention Network for hERG cardiotoxicity classification.
Baseline (non-causal) architecture -- this is the foundation we'll extend
with the causal component next.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv, global_mean_pool, global_max_pool


class LightweightGAT(nn.Module):
    def __init__(self, node_dim, edge_dim, hidden_dim=64, heads=4, num_layers=3, dropout=0.2):
        super().__init__()
        self.dropout = dropout

        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()

        in_dim = node_dim
        for i in range(num_layers):
            out_dim = hidden_dim
            concat = True
            self.convs.append(
                GATv2Conv(in_dim, out_dim, heads=heads, edge_dim=edge_dim,
                          concat=concat, dropout=dropout)
            )
            self.norms.append(nn.LayerNorm(out_dim * heads))
            in_dim = out_dim * heads

        pooled_dim = in_dim * 2  # mean + max pooling concatenated

        self.classifier = nn.Sequential(
            nn.Linear(pooled_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

        self._last_attention = None  # populated on forward when return_attention=True

    def forward(self, x, edge_index, edge_attr, batch, return_attention=False):
        attn_weights = []
        for conv, norm in zip(self.convs, self.norms):
            if return_attention:
                x, (ei, alpha) = conv(x, edge_index, edge_attr, return_attention_weights=True)
                attn_weights.append((ei, alpha))
            else:
                x = conv(x, edge_index, edge_attr)
            x = norm(x)
            x = F.elu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

        mean_pool = global_mean_pool(x, batch)
        max_pool = global_max_pool(x, batch)
        pooled = torch.cat([mean_pool, max_pool], dim=1)

        out = self.classifier(pooled).squeeze(-1)

        if return_attention:
            return out, attn_weights
        return out