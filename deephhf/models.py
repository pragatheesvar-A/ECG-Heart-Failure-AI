from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F


class ResidualBlock1D(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int = 2, dropout: float = 0.0):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=7, stride=stride, padding=3, bias=False)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=5, padding=2, bias=False)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.skip = nn.Identity() if in_channels == out_channels and stride == 1 else nn.Sequential(
            nn.Conv1d(in_channels, out_channels, 1, stride=stride, bias=False), nn.BatchNorm1d(out_channels)
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.skip(x)
        x = F.gelu(self.bn1(self.conv1(x)))
        x = self.dropout(x)
        x = self.bn2(self.conv2(x))
        return F.gelu(x + residual)


class ECGEncoder(nn.Module):
    """Four-block residual 1D CNN producing a compact window embedding and logit."""
    def __init__(self, channels: list[int] = [32, 64, 96, 128], embedding_dim: int = 128, dropout: float = 0.15):
        super().__init__()
        self.stem = nn.Sequential(nn.Conv1d(1, channels[0], 15, stride=2, padding=7, bias=False), nn.BatchNorm1d(channels[0]), nn.GELU())
        blocks = []
        in_channels = channels[0]
        for out_channels in channels:
            blocks.append(ResidualBlock1D(in_channels, out_channels, stride=2, dropout=dropout))
            in_channels = out_channels
        self.blocks = nn.Sequential(*blocks)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.embedding = nn.Sequential(nn.Linear(channels[-1], embedding_dim), nn.LayerNorm(embedding_dim), nn.GELU(), nn.Dropout(dropout))
        self.classifier = nn.Linear(embedding_dim, 1)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        x = self.blocks(self.stem(x))
        return self.embedding(self.pool(x).squeeze(-1))

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        embedding = self.encode(x)
        return self.classifier(embedding).squeeze(-1), embedding


class AttentionBlock(nn.Module):
    """Transformer block that retains attention weights for gradient-weighted explanations."""
    def __init__(self, dim: int, heads: int, ff_dim: int, dropout: float):
        super().__init__()
        if dim % heads:
            raise ValueError("embedding dimension must be divisible by number of attention heads")
        self.heads = heads
        self.head_dim = dim // heads
        self.qkv = nn.Linear(dim, 3 * dim)
        self.attn_dropout = nn.Dropout(dropout)
        self.out_projection = nn.Linear(dim, dim)
        self.norm1 = nn.LayerNorm(dim)
        self.ff = nn.Sequential(nn.Linear(dim, ff_dim), nn.GELU(), nn.Dropout(dropout), nn.Linear(ff_dim, dim))
        self.norm2 = nn.LayerNorm(dim)
        self.drop = nn.Dropout(dropout)
        self.last_attention: torch.Tensor | None = None

    def forward(self, x: torch.Tensor, padding_mask: torch.Tensor | None = None) -> torch.Tensor:
        batch, length, dim = x.shape
        qkv = self.qkv(x).reshape(batch, length, 3, self.heads, self.head_dim).permute(2, 0, 3, 1, 4)
        query, key, value = qkv[0], qkv[1], qkv[2]
        scores = (query @ key.transpose(-2, -1)) / (self.head_dim ** 0.5)
        if padding_mask is not None:
            scores = scores.masked_fill(padding_mask[:, None, None, :], float("-inf"))
        weights = torch.softmax(scores, dim=-1)
        if weights.requires_grad:
            weights.retain_grad()
        self.last_attention = weights
        attended = self.attn_dropout(weights) @ value
        attended = self.out_projection(attended.transpose(1, 2).reshape(batch, length, dim))
        x = self.norm1(x + self.drop(attended))
        return self.norm2(x + self.drop(self.ff(x)))


class RecordingTransformer(nn.Module):
    def __init__(self, embedding_dim: int = 128, heads: int = 4, layers: int = 3, ff_dim: int = 256, dropout: float = 0.15, max_windows: int = 800):
        super().__init__()
        self.position = nn.Parameter(torch.randn(1, max_windows + 1, embedding_dim) * 0.02)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embedding_dim))
        self.layers = nn.ModuleList([AttentionBlock(embedding_dim, heads, ff_dim, dropout) for _ in range(layers)])
        self.norm = nn.LayerNorm(embedding_dim)
        self.classifier = nn.Linear(embedding_dim, 1)

    def forward(self, embeddings: torch.Tensor, padding_mask: torch.Tensor | None = None) -> torch.Tensor:
        batch, length, _ = embeddings.shape
        if length + 1 > self.position.shape[1]:
            raise ValueError(f"Sequence length {length} exceeds max_windows")
        x = torch.cat((self.cls_token.expand(batch, -1, -1), embeddings), dim=1) + self.position[:, : length + 1]
        mask = None if padding_mask is None else torch.cat((torch.zeros((batch, 1), dtype=torch.bool, device=padding_mask.device), padding_mask), dim=1)
        for layer in self.layers:
            x = layer(x, mask)
        return self.classifier(self.norm(x[:, 0])).squeeze(-1)

    def gradient_attention(self) -> torch.Tensor:
        """Gradient-weighted CLS attention for each input window, normalized to [0,1]."""
        block = self.layers[-1]
        weights = block.last_attention
        if weights is None or weights.grad is None:
            raise RuntimeError("Call backward() on a score before requesting gradient attention")
        scores = (weights.grad[:, :, 0, 1:] * weights[:, :, 0, 1:]).mean(dim=1).clamp_min(0)
        return scores / scores.sum(dim=1, keepdim=True).clamp_min(1e-8)
