import gc
from typing import Iterable

import torch
from torch import nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, SAGEConv


class NodeClassifierA(nn.Module):
    """Memory-light multi-class node classifier with optional graph smoothing."""

    def __init__(
        self,
        in_channels: int,
        num_classes: int = 7,
        hidden_channels: int = 256,
        num_layers: int = 3,
        dropout: float = 0.35,
        batch_size: int = 65536,
        propagation_steps: int = 0,
        propagation_alpha: float = 0.0,
    ):
        super().__init__()
        if num_layers < 2:
            raise ValueError("num_layers must be at least 2")

        self.in_channels = int(in_channels)
        self.num_classes = int(num_classes)
        self.hidden_channels = int(hidden_channels)
        self.num_layers = int(num_layers)
        self.dropout = float(dropout)
        self.batch_size = int(batch_size)
        self.propagation_steps = int(propagation_steps)
        self.propagation_alpha = float(propagation_alpha)

        layers: list[nn.Module] = []
        dim = in_channels
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(dim, hidden_channels))
            layers.append(nn.BatchNorm1d(hidden_channels))
            layers.append(nn.ReLU(inplace=True))
            layers.append(nn.Dropout(dropout))
            dim = hidden_channels
        layers.append(nn.Linear(dim, num_classes))
        self.net = nn.Sequential(*layers)

        self.register_buffer("x_mean", torch.zeros(in_channels, dtype=torch.float32))
        self.register_buffer("x_std", torch.ones(in_channels, dtype=torch.float32))
        self.register_buffer("train_nodes", torch.empty(0, dtype=torch.long))
        self.register_buffer("train_labels", torch.empty(0, dtype=torch.long))

    def set_normalizer(self, mean: torch.Tensor, std: torch.Tensor) -> None:
        self.x_mean.copy_(mean.float())
        self.x_std.copy_(std.float().clamp_min(1e-6))

    def set_label_anchors(self, train_nodes: torch.Tensor, train_labels: torch.Tensor) -> None:
        self.train_nodes = train_nodes.detach().cpu().long()
        self.train_labels = train_labels.detach().cpu().long()

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor | None = None) -> torch.Tensor:
        del edge_index
        x = x.float()
        x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        x = (x - self.x_mean.to(x.device)) / self.x_std.to(x.device)
        x = torch.clamp(x, min=-10.0, max=10.0)
        logits = self.net(x)
        return torch.nan_to_num(logits, nan=0.0, posinf=20.0, neginf=-20.0)

    @torch.no_grad()
    def predict_logits(
        self,
        x: torch.Tensor,
        device: torch.device | str | None = None,
        batch_size: int | None = None,
    ) -> torch.Tensor:
        was_training = self.training
        self.eval()
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        device = torch.device(device)
        batch_size = int(batch_size or self.batch_size)
        self.to(device)

        out = torch.empty((x.size(0), self.num_classes), dtype=torch.float32)
        for start in range(0, x.size(0), batch_size):
            end = min(start + batch_size, x.size(0))
            xb = x[start:end].to(device, non_blocking=True)
            out[start:end] = self(xb).cpu()
            del xb
        if device.type == "cuda":
            torch.cuda.empty_cache()
        if was_training:
            self.train()
        return out

    @torch.no_grad()
    def propagate(self, probs: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        if self.propagation_steps <= 0 or self.propagation_alpha <= 0.0:
            return probs.float()

        edge_index = edge_index.cpu().long()
        row, col = edge_index[0], edge_index[1]
        n, c = probs.shape
        base = probs.float().cpu()
        current = base.clone()
        train_nodes = self.train_nodes.cpu()
        train_labels = self.train_labels.cpu()
        anchor = torch.zeros((train_nodes.numel(), c), dtype=torch.float32)
        if train_nodes.numel() > 0:
            anchor[torch.arange(train_nodes.numel()), train_labels] = 1.0

        for _ in range(int(self.propagation_steps)):
            acc = torch.zeros((n, c), dtype=torch.float32)
            deg = torch.zeros(n, dtype=torch.float32)
            acc.index_add_(0, col, current[row])
            deg.index_add_(0, col, torch.ones(col.numel(), dtype=torch.float32))
            neigh = acc / deg.clamp_min(1.0).unsqueeze(1)
            current = (1.0 - self.propagation_alpha) * base + self.propagation_alpha * neigh
            if train_nodes.numel() > 0:
                current[train_nodes] = anchor
            del acc, deg, neigh
            gc.collect()
        return current.clamp_min_(0.0)

    @torch.no_grad()
    def predict_all(self, data, device: torch.device | str | None = None) -> torch.Tensor:
        logits = self.predict_logits(data.x, device=device)
        probs = torch.softmax(logits, dim=1)
        probs = self.propagate(probs, data.edge_index)
        return probs.argmax(dim=1).long()


def iter_batches(index: torch.Tensor, batch_size: int, shuffle: bool = True) -> Iterable[torch.Tensor]:
    order = torch.randperm(index.numel()) if shuffle else torch.arange(index.numel())
    for start in range(0, index.numel(), batch_size):
        yield index[order[start:start + batch_size]]


class GCN_A(nn.Module):
    """Two-layer full-batch GCN for the Cora-like Dataset A graph."""

    def __init__(self, in_channels: int, hidden_channels: int, out_channels: int, dropout: float = 0.5):
        super().__init__()
        self.dropout = float(dropout)
        self.conv1 = GCNConv(in_channels, hidden_channels, cached=False)
        self.norm1 = nn.LayerNorm(hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels, cached=False)
        self.skip = nn.Linear(in_channels, out_channels, bias=False)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        x = torch.nan_to_num(x.float(), nan=0.0, posinf=0.0, neginf=0.0)
        x = F.normalize(x, p=2, dim=-1)
        h = F.dropout(x, p=self.dropout, training=self.training)
        h = self.conv1(h, edge_index)
        h = self.norm1(h)
        h = F.relu(h, inplace=True)
        h = F.dropout(h, p=self.dropout, training=self.training)
        h = self.conv2(h, edge_index)
        return torch.nan_to_num(h + self.skip(x), nan=0.0, posinf=20.0, neginf=-20.0)


class GraphSAGE_A(nn.Module):
    """Full-batch GraphSAGE alternative for Dataset A."""

    def __init__(self, in_channels: int, hidden_channels: int, out_channels: int, dropout: float = 0.5):
        super().__init__()
        self.dropout = float(dropout)
        self.conv1 = SAGEConv(in_channels, hidden_channels, aggr="mean")
        self.norm1 = nn.LayerNorm(hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, out_channels, aggr="mean")
        self.skip = nn.Linear(in_channels, out_channels, bias=False)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        x = torch.nan_to_num(x.float(), nan=0.0, posinf=0.0, neginf=0.0)
        x = F.normalize(x, p=2, dim=-1)
        h = F.dropout(x, p=self.dropout, training=self.training)
        h = self.conv1(h, edge_index)
        h = self.norm1(h)
        h = F.relu(h, inplace=True)
        h = F.dropout(h, p=self.dropout, training=self.training)
        h = self.conv2(h, edge_index)
        return torch.nan_to_num(h + self.skip(x), nan=0.0, posinf=20.0, neginf=-20.0)