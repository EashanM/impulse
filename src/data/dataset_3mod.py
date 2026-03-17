"""Three-modality episode dataset for RL environment.

Each episode contains pre-computed embeddings from frozen encoders plus
shared quality proxy vectors for all three modalities.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


PROXY_DIM = 15  # Fixed proxy vector dimensionality


@dataclass
class SubjectEpisode3Mod:
    """One subject's recording as a 3-modality sequential episode.

    Fields
    ------
    cardiac_embed : (T, D_c) — ECG GRU hidden states (e.g. 64-dim)
    somatic_embed : (T, D_s) — EDA GRU hidden states (e.g. 32-dim)
    vascular_embed : (T, D_v) — BVP CNN-GRU hidden states (e.g. 64-dim)
    proxies : (T, 15) — shared quality proxy vector
    labels : (T,) — binary stress labels 0/1
    """

    subject_id: int
    cardiac_embed: np.ndarray
    somatic_embed: np.ndarray
    vascular_embed: np.ndarray
    proxies: np.ndarray
    labels: np.ndarray
    seq_len: int

    def get_observation(self, t: int, agent: str) -> np.ndarray:
        """Return (seq_len, embed_dim + proxy_dim) observation for agent at time t.

        Each agent sees its own modality embedding concatenated with the shared
        quality proxy vector from all three modalities.
        """
        embed = self._get_embed(agent)
        D_e = embed.shape[1]
        D_p = self.proxies.shape[1]
        D = D_e + D_p

        if t < self.seq_len - 1:
            n_visible = t + 1
            pad = np.zeros((self.seq_len - n_visible, D), dtype=np.float32)
            embed_win = embed[:t + 1].astype(np.float32)
            proxy_win = self.proxies[:t + 1].astype(np.float32)
            window = np.concatenate([embed_win, proxy_win], axis=1)
            obs = np.concatenate([pad, window], axis=0)
        else:
            embed_win = embed[t - self.seq_len + 1:t + 1].astype(np.float32)
            proxy_win = self.proxies[t - self.seq_len + 1:t + 1].astype(np.float32)
            obs = np.concatenate([embed_win, proxy_win], axis=1)

        return obs

    def _get_embed(self, agent: str) -> np.ndarray:
        if agent == "cardiac":
            return self.cardiac_embed
        elif agent == "somatic":
            return self.somatic_embed
        elif agent == "vascular":
            return self.vascular_embed
        raise ValueError(f"Unknown agent: {agent}")

    def __len__(self) -> int:
        return len(self.labels)

    def precompute_stressor_block_info(self) -> np.ndarray:
        """For each timestep, compute (steps_into_stressor, total_stressor_steps)."""
        T = len(self.labels)
        result = np.full((T, 2), -1, dtype=np.int32)
        i = 0
        while i < T:
            if self.labels[i] != 1:
                i += 1
                continue
            start = i
            while i < T and self.labels[i] == 1:
                i += 1
            end = i
            total = end - start
            for t_idx in range(start, end):
                result[t_idx, 0] = t_idx - start
                result[t_idx, 1] = total
        return result
