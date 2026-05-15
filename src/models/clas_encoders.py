"""Encoder heads for CLAS windows: linear, GRU on raw time steps, and 1D CNN+GRU."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class CLASLinearEncoder(nn.Module):
    """Flatten (B, C, L) and apply a single linear map to embedding_dim."""

    def __init__(self, in_channels: int, seq_len: int, embedding_dim: int) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.seq_len = seq_len
        flat = in_channels * seq_len
        self.proj = nn.Linear(flat, embedding_dim)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 3:
            raise ValueError(f"Expected (B,C,L), got {tuple(x.shape)}")
        b, c, l = x.shape
        if c != self.in_channels or l != self.seq_len:
            raise ValueError(f"Expected C={self.in_channels}, L={self.seq_len}, got {c}, {l}")
        flat = x.reshape(b, c * l)
        return self.proj(flat)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.encode(x)


class CLASGruEncoder(nn.Module):
    """GRU directly on (B, L, C) from raw windows; embedding is final-layer last hidden state."""

    def __init__(
        self,
        in_channels: int,
        seq_len: int,
        hidden_size: int = 64,
        num_layers: int = 1,
        dropout: float = 0.25,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.seq_len = seq_len
        gru_dropout = dropout if num_layers > 1 else 0.0
        self.gru = nn.GRU(
            input_size=in_channels,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=gru_dropout,
        )
        self.dropout = nn.Dropout(dropout)
        self._emb_dim = hidden_size

    @property
    def embedding_dim(self) -> int:
        return self._emb_dim

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 3:
            raise ValueError(f"Expected (B,C,L), got {tuple(x.shape)}")
        b, c, l = x.shape
        if c != self.in_channels or l != self.seq_len:
            raise ValueError(f"Expected C={self.in_channels}, L={self.seq_len}, got {c}, {l}")
        x_seq = x.transpose(1, 2)  # (B, L, C)
        _out, h_n = self.gru(x_seq)
        last = h_n[-1]  # (B, hidden)
        return self.dropout(last)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.encode(x)


class CLASCnnGruEncoder(nn.Module):
    """1D CNN stack + GRU; `encode` returns last GRU hidden state."""

    def __init__(
        self,
        in_channels: int,
        seq_len: int,
        conv1_filters: int = 32,
        conv2_filters: int = 64,
        kernel_size_1: int = 7,
        kernel_size_2: int = 5,
        pool_size: int = 4,
        gru_hidden: int = 64,
        num_layers: int = 1,
        dropout: float = 0.25,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.seq_len = seq_len
        self.conv1 = nn.Conv1d(
            in_channels, conv1_filters, kernel_size=kernel_size_1, padding=kernel_size_1 // 2
        )
        self.bn1 = nn.BatchNorm1d(conv1_filters)
        self.pool1 = nn.MaxPool1d(pool_size)
        self.conv2 = nn.Conv1d(
            conv1_filters, conv2_filters, kernel_size=kernel_size_2, padding=kernel_size_2 // 2
        )
        self.bn2 = nn.BatchNorm1d(conv2_filters)
        self.pool2 = nn.MaxPool1d(pool_size)
        gru_dropout = dropout if num_layers > 1 else 0.0
        self.gru = nn.GRU(
            input_size=conv2_filters,
            hidden_size=gru_hidden,
            num_layers=num_layers,
            batch_first=True,
            dropout=gru_dropout,
        )
        self.dropout = nn.Dropout(dropout)
        self._emb_dim = gru_hidden

    @property
    def embedding_dim(self) -> int:
        return self._emb_dim

    def _cnn_features(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, L)
        h = F.relu(self.bn1(self.conv1(x)))
        h = self.pool1(h)
        h = F.relu(self.bn2(self.conv2(h)))
        h = self.pool2(h)
        return h.permute(0, 2, 1)  # (B, T', C2)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        feat = self._cnn_features(x)
        gru_out, _ = self.gru(feat)
        last = gru_out[:, -1, :]
        return self.dropout(last)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.encode(x)


class EncoderWithHead(nn.Module):
    """Wrapper: encoder + linear classifier head (for supervised pre-training)."""

    def __init__(self, encoder: nn.Module, embedding_dim: int, num_classes: int = 2) -> None:
        super().__init__()
        self.encoder = encoder
        self.head = nn.Linear(embedding_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.encoder.encode(x) if hasattr(self.encoder, "encode") else self.encoder(x)
        return self.head(z)
