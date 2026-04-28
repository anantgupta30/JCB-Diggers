from typing import Iterable

import torch
from torch import nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv
from torch_geometric.utils import dropout_edge
from torch_geometric.nn import GATv2Conv


class LinkPredictorC(nn.Module):
    """Feature-based link predictor for Dataset C."""

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int = 256,
        embed_channels: int = 128,
        dropout: float = 0.20,
        score_batch_size: int = 262144,
    ):
        super().__init__()
        self.in_channels = int(in_channels)
        self.hidden_channels = int(hidden_channels)
        self.embed_channels = int(embed_channels)
        self.dropout = float(dropout)
        self.score_batch_size = int(score_batch_size)

        self.register_buffer("x_mean", torch.zeros(in_channels, dtype=torch.float32))
        self.register_buffer("x_std", torch.ones(in_channels, dtype=torch.float32))
        self.register_buffer("log_degree", torch.empty(0, dtype=torch.float32))

        self.encoder = nn.Sequential(
            nn.Linear(in_channels, hidden_channels),
            nn.BatchNorm1d(hidden_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels, embed_channels),
            nn.ReLU(inplace=True),
        )
        pair_dim = embed_channels * 4 + 2
        self.scorer = nn.Sequential(
            nn.Linear(pair_dim, hidden_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels, hidden_channels // 2),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_channels // 2, 1),
        )

    def set_normalizer(self, mean: torch.Tensor, std: torch.Tensor) -> None:
        self.x_mean.copy_(mean.float())
        self.x_std.copy_(std.float().clamp_min(1e-6))

    def set_degree(self, edge_index: torch.Tensor, num_nodes: int) -> None:
        deg = torch.zeros(int(num_nodes), dtype=torch.float32)
        if edge_index.numel() > 0:
            deg.index_add_(0, edge_index[0].cpu().long(), torch.ones(edge_index.size(1), dtype=torch.float32))
        self.log_degree = torch.log1p(deg)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        x = x.float()
        x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        x = (x - self.x_mean.to(x.device)) / self.x_std.to(x.device)
        x = torch.clamp(x, min=-10.0, max=10.0)
        return self.encoder(x)

    def score_from_embeddings(self, z: torch.Tensor, edge_pairs: torch.Tensor) -> torch.Tensor:
        src = z[edge_pairs[:, 0].long()]
        dst = z[edge_pairs[:, 1].long()]
        if self.log_degree.numel() >= z.size(0):
            degree = self.log_degree.to(z.device)
            deg_pair = torch.stack(
                [degree[edge_pairs[:, 0].long()], degree[edge_pairs[:, 1].long()]],
                dim=1,
            )
        else:
            deg_pair = torch.zeros((edge_pairs.size(0), 2), dtype=z.dtype, device=z.device)
        pair = torch.cat([src, dst, src * dst, torch.abs(src - dst), deg_pair], dim=1)
        logits = self.scorer(pair).view(-1)
        return torch.nan_to_num(logits, nan=0.0, posinf=20.0, neginf=-20.0)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, edge_pairs: torch.Tensor) -> torch.Tensor:
        del edge_index
        if edge_pairs.numel() == 0:
            return torch.empty(0, dtype=torch.float32, device=x.device)
        z = self.encode(x)
        return self.score_from_embeddings(z, edge_pairs)

    @torch.no_grad()
    def score_edges(
        self,
        x: torch.Tensor,
        edge_pairs: torch.Tensor,
        device: torch.device | str | None = None,
        batch_size: int | None = None,
    ) -> torch.Tensor:
        was_training = self.training
        self.eval()
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        device = torch.device(device)
        batch_size = int(batch_size or self.score_batch_size)
        self.to(device)
        x = x.to(device, non_blocking=True)
        z = self.encode(x)

        out = torch.empty(edge_pairs.size(0), dtype=torch.float32)
        edge_pairs = edge_pairs.cpu().long()
        for start in range(0, edge_pairs.size(0), batch_size):
            end = min(start + batch_size, edge_pairs.size(0))
            pairs = edge_pairs[start:end].to(device, non_blocking=True)
            out[start:end] = self.score_from_embeddings(z, pairs).cpu()
            del pairs
        if device.type == "cuda":
            torch.cuda.empty_cache()
        if was_training:
            self.train()
        return out


def iter_edge_batches(pos_edges: torch.Tensor, neg_edges: torch.Tensor, batch_size: int, shuffle: bool = True) -> Iterable[tuple[torch.Tensor, torch.Tensor]]:
    n = min(pos_edges.size(0), neg_edges.size(0))
    order = torch.randperm(n) if shuffle else torch.arange(n)
    half = max(1, batch_size // 2)
    for start in range(0, n, half):
        idx = order[start:start + half]
        edges = torch.cat([pos_edges[idx], neg_edges[idx]], dim=0)
        labels = torch.cat([
            torch.ones(idx.numel(), dtype=torch.float32),
            torch.zeros(idx.numel(), dtype=torch.float32),
        ])
        perm = torch.randperm(edges.size(0)) if shuffle else torch.arange(edges.size(0))
        yield edges[perm], labels[perm]


class DualSignalLinkPredictorC(nn.Module):
    """
    Dual feature/topology link scorer using GATv2 for structural robustness.
    Maintains Cosine Similarity to prevent gradient collapse.
    """
    expects_edge_index = True

    def __init__(
        self,
        raw_in_channels: int,
        in_channels: int = 256,    # Widened from 128
        hidden_channels: int = 256,
        embed_channels: int = 128,
        dropout: float = 0.30,     # Back to standard dropout
        score_batch_size: int = 262144,
    ):
        super().__init__()
        self.raw_in_channels = int(raw_in_channels)
        self.in_channels = int(in_channels)
        self.hidden_channels = int(hidden_channels)
        self.embed_channels = int(embed_channels)
        self.dropout = float(dropout)
        self.score_batch_size = int(score_batch_size)

        # 1. Wider single-layer projection
        self.input_proj = nn.Sequential(
            nn.Linear(raw_in_channels, in_channels),
            nn.LayerNorm(in_channels),
            nn.Dropout(dropout)
        )

        # 2. GATv2 Path (Multi-head attention)
        # 4 heads of size 64 concatenate back to 256
        self.gat1 = GATv2Conv(in_channels, hidden_channels // 4, heads=4, concat=True, dropout=dropout)
        self.norm1 = nn.LayerNorm(hidden_channels)
        self.gat2 = GATv2Conv(hidden_channels, embed_channels, heads=1, concat=False, dropout=dropout)
        
        # 3. Feature Path
        self.mlp = nn.Sequential(
            nn.Linear(in_channels, hidden_channels),
            nn.LayerNorm(hidden_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels, embed_channels),
        )
        
        self.logit_alpha = nn.Parameter(torch.zeros(1))
        self.temperature = nn.Parameter(torch.tensor(5.0))

    def encode(self, x: torch.Tensor, edge_index: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = torch.nan_to_num(x.float(), nan=0.0, posinf=0.0, neginf=0.0)
        
        # Project raw features
        x_proj = self.input_proj(x)
        x_proj = F.relu(x_proj, inplace=True)

        # GAT Path
        h = F.dropout(x_proj, p=self.dropout, training=self.training)
        h = self.gat1(h, edge_index)
        h = self.norm1(h)
        h = F.relu(h, inplace=True)
        h = F.dropout(h, p=self.dropout, training=self.training)
        z_gnn = self.gat2(h, edge_index)
        
        # Feat Path
        z_feat = self.mlp(x_proj)
        
        return z_gnn, z_feat

    def decode(self, z_gnn: torch.Tensor, z_feat: torch.Tensor, edge_pairs: torch.Tensor) -> torch.Tensor:
        src = edge_pairs[:, 0].long()
        dst = edge_pairs[:, 1].long()
        
        # Cosine Similarity bounded scoring
        gnn_score = F.cosine_similarity(z_gnn[src], z_gnn[dst], dim=-1)
        feat_score = F.cosine_similarity(z_feat[src], z_feat[dst], dim=-1)
        
        alpha = torch.sigmoid(self.logit_alpha)
        scores = alpha * gnn_score + (1.0 - alpha) * feat_score
        
        return scores * self.temperature

    def bpr_loss(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        pos_edges: torch.Tensor,
        neg_edges: torch.Tensor,
    ) -> torch.Tensor:
        z_gnn, z_feat = self.encode(x, edge_index)
        pos_scores = self.decode(z_gnn, z_feat, pos_edges)
        neg_scores = self.decode(z_gnn, z_feat, neg_edges)
        
        margin = 0.5 # Slightly tighter margin
        return -F.logsigmoid(pos_scores - neg_scores - margin).mean()

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, edge_pairs: torch.Tensor) -> torch.Tensor:
        z_gnn, z_feat = self.encode(x, edge_index)
        return self.decode(z_gnn, z_feat, edge_pairs)

    @torch.no_grad()
    def score_edges(self, x: torch.Tensor, edge_index: torch.Tensor, edge_pairs: torch.Tensor, device=None, batch_size=None) -> torch.Tensor:
        was_training = self.training
        self.eval()
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        device = torch.device(device)
        batch_size = int(batch_size or self.score_batch_size)
        self.to(device)

        x = x.to(device, non_blocking=True)
        edge_index = edge_index.to(device, non_blocking=True)
        z_gnn, z_feat = self.encode(x, edge_index)
        edge_pairs = edge_pairs.cpu().long()
        out = torch.empty(edge_pairs.size(0), dtype=torch.float32)
        for start in range(0, edge_pairs.size(0), batch_size):
            end = min(start + batch_size, edge_pairs.size(0))
            pairs = edge_pairs[start:end].to(device, non_blocking=True)
            out[start:end] = self.decode(z_gnn, z_feat, pairs).cpu()
            del pairs
        if device.type == "cuda":
            torch.cuda.empty_cache()
        if was_training:
            self.train()
        return out


def sample_negative_edges(pos_edges: torch.Tensor, num_nodes: int, existing: set[tuple[int, int]]) -> torch.Tensor:
    neg = torch.empty_like(pos_edges)
    filled = 0
    while filled < pos_edges.size(0):
        remaining = pos_edges.size(0) - filled
        src = pos_edges[filled:filled + remaining, 0]
        dst = torch.randint(0, num_nodes, (remaining,), dtype=torch.long)
        for i, (u, v) in enumerate(zip(src.tolist(), dst.tolist())):
            if u != v and (u, v) not in existing and (v, u) not in existing:
                neg[filled] = torch.tensor([u, v], dtype=torch.long)
                filled += 1
                if filled >= pos_edges.size(0):
                    break
    return neg