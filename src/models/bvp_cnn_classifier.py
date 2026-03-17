"""Lightweight 1D CNN-GRU classifier for raw BVP waveforms.

Designed for stress detection from wrist PPG/BVP signals at 64 Hz.

Architecture (inspired by Mim et al. dual-branch CNN-GRU):
    Conv1D(1→64, k=7) → BN → ReLU → MaxPool(4)
    Conv1D(64→128, k=5) → BN → ReLU → MaxPool(4)
    GRU(128→gru_hidden)
    Dropout → FC(gru_hidden → num_classes)

Input: (B, L) raw BVP samples where L = window_samples (e.g. 1280 for 20s @ 64 Hz)
Output: (B, num_classes) logits

The forward_with_state() method mirrors GRUClassifier.forward_with_state()
so that the dynamic gate pipeline can treat BVP identically to ECG/EDA.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class BVPCNNClassifier(nn.Module):
    """1D CNN-GRU classifier for raw BVP windows."""

    def __init__(
        self,
        input_len: int = 1280,
        conv1_filters: int = 64,
        conv2_filters: int = 128,
        kernel_size_1: int = 7,
        kernel_size_2: int = 5,
        pool_size: int = 4,
        gru_hidden: int = 64,
        num_layers: int = 1,
        num_classes: int = 2,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        self.input_len = input_len

        # CNN feature extractor
        self.conv1 = nn.Conv1d(1, conv1_filters, kernel_size=kernel_size_1, padding=kernel_size_1 // 2)
        self.bn1 = nn.BatchNorm1d(conv1_filters)
        self.pool1 = nn.MaxPool1d(pool_size)

        self.conv2 = nn.Conv1d(conv1_filters, conv2_filters, kernel_size=kernel_size_2, padding=kernel_size_2 // 2)
        self.bn2 = nn.BatchNorm1d(conv2_filters)
        self.pool2 = nn.MaxPool1d(pool_size)

        # GRU for temporal modelling
        gru_dropout = dropout if num_layers > 1 else 0.0
        self.gru = nn.GRU(
            input_size=conv2_filters,
            hidden_size=gru_hidden,
            num_layers=num_layers,
            batch_first=True,
            dropout=gru_dropout,
            bidirectional=False,
        )

        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(gru_hidden, num_classes)

    def _cnn_encode(self, x: torch.Tensor) -> torch.Tensor:
        """Run CNN feature extraction.

        Parameters
        ----------
        x : torch.Tensor
            Shape (B, 1, L).

        Returns
        -------
        torch.Tensor
            Shape (B, T', conv2_filters) — ready for GRU (batch_first).
        """
        h = F.relu(self.bn1(self.conv1(x)))  # (B, C1, L)
        h = self.pool1(h)                     # (B, C1, L//pool)
        h = F.relu(self.bn2(self.conv2(h)))   # (B, C2, L//pool)
        h = self.pool2(h)                     # (B, C2, L//pool^2)
        h = h.permute(0, 2, 1)               # (B, T', C2)  — batch_first for GRU
        return h

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Classify raw BVP windows.

        Parameters
        ----------
        x : torch.Tensor
            Shape (B, L) or (B, 1, L) raw BVP samples.

        Returns
        -------
        torch.Tensor
            Shape (B, num_classes) logits.
        """
        if x.dim() == 2:
            x = x.unsqueeze(1)  # (B, 1, L)

        features = self._cnn_encode(x)        # (B, T', C2)
        gru_out, _ = self.gru(features)        # (B, T', H)
        last = gru_out[:, -1, :]               # (B, H)
        last = self.dropout(last)
        return self.head(last)                  # (B, C)

    def forward_with_state(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return per-step outputs matching GRUClassifier.forward_with_state().

        Parameters
        ----------
        x : torch.Tensor
            Shape (B, L) or (B, 1, L).

        Returns
        -------
        logits_seq : torch.Tensor
            Shape (B, T', C) — per-step logits from the GRU sequence.
        stress_prob_seq : torch.Tensor
            Shape (B, T') — probability for class index 1.
        hidden_seq : torch.Tensor
            Shape (B, T', H) — GRU hidden states.
        """
        if x.dim() == 2:
            x = x.unsqueeze(1)  # (B, 1, L)

        features = self._cnn_encode(x)        # (B, T', C2)
        hidden_seq, _ = self.gru(features)     # (B, T', H)
        logits_seq = self.head(self.dropout(hidden_seq))  # (B, T', C)
        stress_prob_seq = F.softmax(logits_seq, dim=-1)[..., 1]  # (B, T')
        return logits_seq, stress_prob_seq, hidden_seq
