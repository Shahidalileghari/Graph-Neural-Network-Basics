"""Optional PyTorch models. Install the 'torch' extra to use this module."""
try:
    import torch
    from torch import nn
    import torch.nn.functional as F
except ImportError as exc:  # pragma: no cover - optional dependency
    raise ImportError("Install with `pip install -e '.[torch]'` to use torch models.") from exc


class GCN(torch.nn.Module):
    """Two-layer GCN using dense adjacency normalization for small teaching graphs."""
    def __init__(self, in_channels: int, hidden_channels: int, out_channels: int,
                 dropout: float = 0.5):
        super().__init__()
        self.lin1 = nn.Linear(in_channels, hidden_channels, bias=False)
        self.lin2 = nn.Linear(hidden_channels, out_channels, bias=False)
        self.dropout = dropout

    @staticmethod
    def normalize_adjacency(adj):
        n = adj.size(0)
        adj = adj + torch.eye(n, device=adj.device, dtype=adj.dtype)
        degree = adj.sum(dim=1).clamp_min(1.0)
        inv = degree.pow(-0.5)
        return inv.unsqueeze(1) * adj * inv.unsqueeze(0)

    def forward(self, x, adjacency):
        norm = self.normalize_adjacency(adjacency)
        x = norm @ self.lin1(x)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        return norm @ self.lin2(x)


class MLPBaseline(torch.nn.Module):
    """Feature-only baseline; useful for measuring the value of graph structure."""
    def __init__(self, in_channels: int, hidden_channels: int, out_channels: int,
                 dropout: float = 0.5):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_channels, hidden_channels),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels, out_channels),
        )

    def forward(self, x):
        return self.net(x)
