"""Small sequence classifiers for SWELL minute-level tensors (B, T, F)."""

from __future__ import annotations

from typing import Literal

import torch
import torch.nn as nn

# cuDNN's fused RNN kernels reject seq_len > 65535 (see pytorch/pytorch#133751). Long windows
# (e.g. 210 s at moderate sample rates) then hit CUDNN_STATUS_NOT_SUPPORTED; run those steps
# with cuDNN disabled so PyTorch falls back to a generic implementation.
_CUDNN_RNN_MAX_TIMESTEPS = 65_535


def _gru_forward_batch_first(gru: nn.GRU, x: torch.Tensor) -> torch.Tensor:
    """Return GRU output (B, T, H); uses non-cuDNN path when T exceeds cuDNN limits."""
    x = x.contiguous()
    if x.is_cuda and x.size(1) > _CUDNN_RNN_MAX_TIMESTEPS:
        with torch.backends.cudnn.flags(enabled=False):
            out, _ = gru(x)
    else:
        out, _ = gru(x)
    return out


class _SeqAttentionPool(nn.Module):
    """Weighted mean over time (B, T, H) -> (B, H). Better than last-step only for long windows."""

    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.score = nn.Linear(hidden_dim, 1)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        w = torch.softmax(self.score(h).squeeze(-1), dim=1)
        return torch.bmm(w.unsqueeze(1), h).squeeze(1)


def _pool_sequence(
    h: torch.Tensor, pool: Literal["last", "attn"], attn: _SeqAttentionPool | None
) -> torch.Tensor:
    if pool == "attn":
        if attn is None:
            raise ValueError("attn pool requested but attention module is None")
        return attn(h)
    return h[:, -1, :]


def _classification_head(
    hidden_dim: int,
    num_classes: int,
    dropout: float,
    head: Literal["linear", "mlp"],
) -> nn.Module:
    if head == "mlp":
        return nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )
    return nn.Sequential(nn.Dropout(dropout), nn.Linear(hidden_dim, num_classes))


class SwellFlattenLinearClassifier(nn.Module):
    """
    Single linear layer on the flattened sequence (bag-of-steps).

    Interprets the whole window of length T and feature dim F as one vector
    of size T*F — a linear encoder / linear decision boundary baseline.
    """

    def __init__(self, seq_len: int, input_dim: int, num_classes: int = 2) -> None:
        super().__init__()
        self.seq_len = seq_len
        self.input_dim = input_dim
        self.fc = nn.Linear(seq_len * input_dim, num_classes)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Flattened input before the linear head (shape ``(B, seq_len * input_dim)``)."""
        return x.reshape(x.size(0), -1)

    @property
    def encoding_dim(self) -> int:
        return int(self.seq_len * self.input_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, F)
        return self.fc(self.encode(x))


class SwellGruClassifier(nn.Module):
    """
    Compact unidirectional GRU on (B, T, F) → pool → linear head (default).

    Defaults are intentionally small (``hidden_dim=16``, ``head='linear'``, ``pool='last'``)
    to reduce overfitting vs the flattened linear baseline on long waveform windows.
    Use ``head='mlp'``, ``pool='attn'``, or larger ``hidden_dim`` when you need more capacity.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 16,
        num_layers: int = 1,
        dropout: float = 0.2,
        num_classes: int = 2,
        *,
        pool: Literal["last", "attn"] = "last",
        input_norm: bool = True,
        head: Literal["linear", "mlp"] = "linear",
    ) -> None:
        super().__init__()
        self.pool_mode = pool
        self._hidden_dim = hidden_dim
        self.input_norm = nn.LayerNorm(input_dim) if input_norm else nn.Identity()
        gru_dropout = dropout if num_layers > 1 else 0.0
        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=gru_dropout,
        )
        self.attn = _SeqAttentionPool(hidden_dim) if pool == "attn" else None
        self.dropout = nn.Dropout(dropout)
        self.head = _classification_head(hidden_dim, num_classes, dropout, head)

    @property
    def encoding_dim(self) -> int:
        return int(self._hidden_dim)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_norm(x)
        out = _gru_forward_batch_first(self.gru, x)
        return _pool_sequence(out, self.pool_mode, self.attn)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.dropout(self.encode(x)))


class SwellCnnGruClassifier(nn.Module):
    """
    Lightweight 1D conv (default one layer) → small GRU → pool → linear head.

    Expects x of shape (B, T, F). Conv sees (B, F, T). Prefer a single entry in
    ``conv_channels`` (e.g. ``[16]``) for raw 1-channel waveforms.
    """

    def __init__(
        self,
        input_dim: int,
        conv_channels: list[int],
        kernel_size: int = 3,
        gru_hidden: int = 16,
        gru_layers: int = 1,
        dropout: float = 0.2,
        num_classes: int = 2,
        *,
        pool: Literal["last", "attn"] = "last",
        input_norm: bool = True,
        head: Literal["linear", "mlp"] = "linear",
    ) -> None:
        super().__init__()
        if kernel_size % 2 == 0:
            raise ValueError("kernel_size must be odd so padding=k//2 preserves length.")
        padding = kernel_size // 2
        blocks: list[nn.Module] = []
        c_in = input_dim
        for c_out in conv_channels:
            blocks.append(nn.Conv1d(c_in, c_out, kernel_size=kernel_size, padding=padding))
            blocks.append(nn.ReLU(inplace=True))
            c_in = c_out
        self.conv = nn.Sequential(*blocks)
        self._hidden_dim = gru_hidden
        self.input_norm = nn.LayerNorm(input_dim) if input_norm else nn.Identity()
        gru_dropout = dropout if gru_layers > 1 else 0.0
        self.gru = nn.GRU(
            input_size=c_in,
            hidden_size=gru_hidden,
            num_layers=gru_layers,
            batch_first=True,
            dropout=gru_dropout,
        )
        self.attn = _SeqAttentionPool(gru_hidden) if pool == "attn" else None
        self.pool_mode = pool
        self.dropout = nn.Dropout(dropout)
        self.head = _classification_head(gru_hidden, num_classes, dropout, head)

    @property
    def encoding_dim(self) -> int:
        return int(self._hidden_dim)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Pooled GRU state (B, H) before the classification head."""
        x = self.input_norm(x)
        z = x.transpose(1, 2).contiguous()
        z = self.conv(z)
        z = z.transpose(1, 2).contiguous()
        out = _gru_forward_batch_first(self.gru, z)
        return _pool_sequence(out, self.pool_mode, self.attn)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.dropout(self.encode(x)))
