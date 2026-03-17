"""
Episode-level dataset wrapper for the PettingZoo environment.

Converts a ProcessedSubject into sequential observations that the
environment can step through, with sliding-window history for the agents.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.data.preprocessor import ProcessedSubject


@dataclass
class SubjectEpisode:
    """
    One subject's recording as a sequential episode.

    Provides windowed observations: at time t, each agent sees the last
    `seq_len` feature vectors as a (seq_len, F) array.
    """

    subject_id: int | str
    cardiac: np.ndarray    # (T, F_cardiac)
    somatic: np.ndarray    # (T, F_somatic)
    labels: np.ndarray     # (T,)
    seq_len: int

    @classmethod
    def from_processed(cls, proc: ProcessedSubject, seq_len: int) -> "SubjectEpisode":
        return cls(
            subject_id=proc.subject_id,
            cardiac=proc.cardiac_features,
            somatic=proc.somatic_features,
            labels=proc.labels,
            seq_len=seq_len,
        )

    def get_observation(self, t: int, agent: str) -> np.ndarray:
        """
        Return the observation for `agent` at time step `t`.

        Includes the current step's features (feats[t]) so the agent sees
        the most recent window when deciding. Shape: (seq_len, F).
        Left-padded with zeros when t < seq_len - 1.
        """
        feats = self.cardiac if agent == "cardiac" else self.somatic
        F = feats.shape[1]

        # Include feats[t] in the observation (agent sees current window)
        if t < self.seq_len - 1:
            n_visible = t + 1
            pad = np.zeros((self.seq_len - n_visible, F), dtype=np.float32)
            window = feats[: t + 1].astype(np.float32)
            obs = np.concatenate([pad, window], axis=0)
        else:
            obs = feats[t - self.seq_len + 1 : t + 1].astype(np.float32)
        return obs

    def precompute_stressor_block_info(self) -> np.ndarray:
        """
        For each time step t, compute (steps_into_stressor, total_stressor_steps).
        When in stress (label=1): steps_into_stressor = t - stress_start,
        total_stressor_steps = stress_end - stress_start.
        When not in stress: (-1, -1).

        Used by the reward function for early-detection scaling within the stress window.
        """
        T = len(self.labels)
        # (T, 2): col 0 = steps_into_stressor, col 1 = total_stressor_steps
        result = np.full((T, 2), -1, dtype=np.int32)

        i = 0
        while i < T:
            if self.labels[i] != 1:  # not stress
                i += 1
                continue
            start = i
            while i < T and self.labels[i] == 1:
                i += 1
            end = i
            total = end - start
            for t in range(start, end):
                result[t, 0] = t - start
                result[t, 1] = total
        return result

    def __len__(self) -> int:
        return len(self.labels)
