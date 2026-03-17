"""Three-modality actor-critic agents for multi-agent RL.

Each agent has its own actor (policy) and critic (value) networks.
All agents share the same reward signal but have independent observations.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical


class AgentActorCritic(nn.Module):
    """Single agent's actor-critic network.

    Uses causal 1D convolutions over the observation window to capture temporal
    dynamics (rising/falling trends) for early detection. Conv1d is fully
    parallelizable on MPS, unlike GRU which runs serially.
    """

    def __init__(self, obs_dim: int, seq_len: int, hidden: int = 64):
        super().__init__()
        self.input_norm = nn.LayerNorm(obs_dim)
        # Causal 1D conv stack: captures local temporal patterns
        # kernel_size=5 covers ~5 timestep trends
        self.conv1 = nn.Conv1d(obs_dim, 32, kernel_size=5, padding=4)  # causal: pad left
        self.conv2 = nn.Conv1d(32, 16, kernel_size=3, padding=2)       # causal: pad left
        self.shared = nn.Sequential(
            nn.Linear(16, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
        )
        self.actor = nn.Linear(hidden, 2)   # alert / wait
        self.critic = nn.Linear(hidden, 1)  # value estimate

    def forward(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass.

        Parameters
        ----------
        obs : (B, seq_len, obs_dim) or (seq_len, obs_dim)

        Returns
        -------
        action_logits : (B, 2)
        value : (B, 1)
        """
        if obs.dim() == 2:
            obs = obs.unsqueeze(0)
        # obs: (B, seq_len, obs_dim)
        obs = torch.nan_to_num(obs, nan=0.0)
        obs = self.input_norm(obs)
        # Conv1d expects (B, C, L) — transpose from (B, L, C)
        x = obs.transpose(1, 2)  # (B, obs_dim, seq_len)
        x = F.relu(self.conv1(x)[:, :, :obs.shape[1]])  # causal: trim right
        x = F.relu(self.conv2(x)[:, :, :obs.shape[1]])   # causal: trim right
        # Take last timestep (most recent, like GRU last hidden)
        h = x[:, :, -1]  # (B, 16)
        h = self.shared(h)
        return self.actor(h), self.critic(h)

    def act(self, obs: torch.Tensor) -> tuple[int, float, float]:
        """Sample action, return (action, log_prob, value).

        Parameters
        ----------
        obs : (seq_len, obs_dim) — single observation, no batch dim.
        """
        obs = obs.unsqueeze(0)  # add batch dim
        logits, value = self.forward(obs)
        dist = Categorical(logits=logits)
        action = dist.sample()
        return (
            int(action.item()),
            float(dist.log_prob(action).item()),
            float(value.squeeze(-1).item()),
        )

    def evaluate(self, obs: torch.Tensor, actions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Evaluate actions for PPO update.

        Parameters
        ----------
        obs : (B, seq_len, obs_dim)
        actions : (B,) int actions

        Returns
        -------
        log_probs : (B,)
        values : (B,)
        entropy : (B,)
        """
        logits, values = self.forward(obs)
        dist = Categorical(logits=logits)
        log_probs = dist.log_prob(actions)
        entropy = dist.entropy()
        return log_probs, values.squeeze(-1), entropy


class ThreeModActorCritic(nn.Module):
    """Container for three independent actor-critic agents."""

    def __init__(
        self,
        cardiac_obs_dim: int,
        somatic_obs_dim: int,
        vascular_obs_dim: int,
        seq_len: int,
        hidden: int = 64,
    ):
        super().__init__()
        self.cardiac = AgentActorCritic(cardiac_obs_dim, seq_len, hidden)
        self.somatic = AgentActorCritic(somatic_obs_dim, seq_len, hidden)
        self.vascular = AgentActorCritic(vascular_obs_dim, seq_len, hidden)

        self._agents = {
            "cardiac": self.cardiac,
            "somatic": self.somatic,
            "vascular": self.vascular,
        }

    def get_agent(self, name: str) -> AgentActorCritic:
        return self._agents[name]
