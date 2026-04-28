import gc
from typing import Iterable

import torch
from torch import nn
import torch.nn.functional as F


class FeatureMLPWithPropagation(nn.Module):
    """
    Dataset-B binary node classifier.

    The trainable part is a mini-batch MLP over node features, which keeps GPU
    memory bounded on Kaggle T4.  At inference we optionally smooth the MLP
    probabilities over edge_index on CPU, using only train labels as anchors.
    """

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int = 512,
        num_layers: int = 3,
        dropout: float = 0.25,
        batch_size: int = 65536,
        propagation_steps: int = 2,
        propagation_alpha: float = 0.45,
        propagation_chunk_size: int = 5_000_000,
    ):
        super().__init__()
        if num_layers < 2:
            raise ValueError("num_layers must be at least 2")

        self.in_channels = int(in_channels)
        self.hidden_channels = int(hidden_channels)
        self.num_layers = int(num_layers)
        self.dropout = float(dropout)
        self.batch_size = int(batch_size)
        self.propagation_steps = int(propagation_steps)
        self.propagation_alpha = float(propagation_alpha)
        self.propagation_chunk_size = int(propagation_chunk_size)

        layers: list[nn.Module] = []
        dim = in_channels
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(dim, hidden_channels))
            layers.append(nn.BatchNorm1d(hidden_channels))
            layers.append(nn.ReLU(inplace=True))
            layers.append(nn.Dropout(dropout))
            dim = hidden_channels
        layers.append(nn.Linear(dim, 1))
        self.net = nn.Sequential(*layers)

        self.register_buffer("x_mean", torch.zeros(in_channels, dtype=torch.float32))
        self.register_buffer("x_std", torch.ones(in_channels, dtype=torch.float32))
        self.register_buffer("train_nodes", torch.empty(0, dtype=torch.long))
        self.register_buffer("train_labels", torch.empty(0, dtype=torch.float32))

    def set_normalizer(self, mean: torch.Tensor, std: torch.Tensor) -> None:
        self.x_mean.copy_(mean.float())
        self.x_std.copy_(std.float().clamp_min(1e-6))

    def set_label_anchors(self, train_nodes: torch.Tensor, train_labels: torch.Tensor) -> None:
        self.train_nodes = train_nodes.detach().cpu().long()
        self.train_labels = train_labels.detach().cpu().float()

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor | None = None) -> torch.Tensor:
        del edge_index
        x = x.float()
        x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        x = (x - self.x_mean.to(x.device)) / self.x_std.to(x.device)
        x = torch.clamp(x, min=-10.0, max=10.0)
        logits = self.net(x)
        return torch.nan_to_num(logits, nan=0.0, posinf=20.0, neginf=-20.0)

    @torch.no_grad()
    def predict_mlp(
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

        n = int(x.size(0))
        out = torch.empty(n, dtype=torch.float32)
        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            xb = x[start:end].to(device, non_blocking=True)
            score = torch.sigmoid(self(xb)).view(-1)
            out[start:end] = torch.nan_to_num(score, nan=0.5, posinf=1.0, neginf=0.0).cpu()
            del xb
        if device.type == "cuda":
            torch.cuda.empty_cache()
        if was_training:
            self.train()
        return out

    @torch.no_grad()
    def propagate(self, scores: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        if self.propagation_steps <= 0 or self.propagation_alpha <= 0.0:
            return scores.float()

        edge_index = edge_index.cpu().long()
        row, col = edge_index[0], edge_index[1]
        n = int(scores.numel())
        base = scores.float().cpu()
        current = base.clone()
        train_nodes = self.train_nodes.cpu()
        train_labels = self.train_labels.cpu()
        chunk = max(1, int(self.propagation_chunk_size))

        for _ in range(int(self.propagation_steps)):
            acc = torch.zeros(n, dtype=torch.float32)
            deg = torch.zeros(n, dtype=torch.float32)
            for start in range(0, row.numel(), chunk):
                end = min(start + chunk, row.numel())
                src = row[start:end]
                dst = col[start:end]
                acc.index_add_(0, dst, current[src])
                deg.index_add_(0, dst, torch.ones(dst.numel(), dtype=torch.float32))

            neigh = acc / deg.clamp_min(1.0)
            current = (1.0 - self.propagation_alpha) * base + self.propagation_alpha * neigh
            if train_nodes.numel() > 0:
                current[train_nodes] = train_labels
            del acc, deg, neigh
            gc.collect()
        return current.clamp_(0.0, 1.0)

    @torch.no_grad()
    def predict_all(
        self,
        data,
        device: torch.device | str | None = None,
        batch_size: int | None = None,
        num_workers: int = 0,
    ) -> torch.Tensor:
        del num_workers
        scores = self.predict_mlp(data.x, device=device, batch_size=batch_size)
        return self.propagate(scores, data.edge_index)


def iter_batches(index: torch.Tensor, batch_size: int, shuffle: bool = True) -> Iterable[torch.Tensor]:
    order = torch.randperm(index.numel()) if shuffle else torch.arange(index.numel())
    for start in range(0, index.numel(), batch_size):
        yield index[order[start:start + batch_size]]


def make_pos_weight(y: torch.Tensor) -> torch.Tensor:
    y = y.float()
    pos = y.sum().clamp_min(1.0)
    neg = (y.numel() - y.sum()).clamp_min(1.0)
    return torch.tensor([float(neg / pos)], dtype=torch.float32)