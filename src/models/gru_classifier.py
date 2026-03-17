from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class GRUClassifier(nn.Module):
    """Compact unidirectional GRU classifier for sequence labeling/classification."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 32,
        num_layers: int = 1,
        dropout: float = 0.1,
        num_classes: int = 2,
    ) -> None:
        super().__init__()
        gru_dropout = dropout if num_layers > 1 else 0.0
        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=gru_dropout,
            bidirectional=False,
        )
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(hidden_dim, num_classes)

    def forward(self, x: torch.Tensor, return_sequence: bool = False) -> torch.Tensor:
        """
        Parameters
        ----------
        x : torch.Tensor
            Shape (B, T, F).
        return_sequence : bool
            If True, returns per-time-step logits (B, T, C).
            Else returns final-step logits (B, C).
        """
        out, _ = self.gru(x)
        if return_sequence:
            out = self.dropout(out)
            return self.head(out)

        last = out[:, -1, :]
        last = self.dropout(last)
        return self.head(last)

    def forward_with_state(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Return sequence logits, per-step stress probabilities, and hidden states.

        Parameters
        ----------
        x : torch.Tensor
            Shape (B, T, F).

        Returns
        -------
        logits_seq : torch.Tensor
            Shape (B, T, C).
        stress_prob_seq : torch.Tensor
            Shape (B, T), probability for class index 1.
        hidden_seq : torch.Tensor
            Shape (B, T, H), GRU hidden states.
        """
        hidden_seq, _ = self.gru(x)
        logits_seq = self.head(self.dropout(hidden_seq))
        stress_prob_seq = F.softmax(logits_seq, dim=-1)[..., 1]
        return logits_seq, stress_prob_seq, hidden_seq
