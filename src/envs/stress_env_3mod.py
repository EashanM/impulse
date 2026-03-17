"""PettingZoo-compatible 3-agent environment for stress monitoring.

Three agents (cardiac, somatic, vascular) observe physiological embeddings
and quality proxies, independently deciding whether to alert. The global
alarm fires when the consensus protocol is satisfied.

Extends the 2-agent StressMonitorEnv with a third BVP-based agent.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np
from gymnasium import spaces
from pettingzoo import AECEnv
from pettingzoo.utils.agent_selector import AgentSelector

from src.config import EnvConfig, RewardConfig
from src.data.dataset_3mod import SubjectEpisode3Mod
from src.envs.reward import compute_reward


class StressMonitor3ModEnv(AECEnv):
    metadata = {"render_modes": [], "name": "stress_monitor_3mod_v0"}

    def __init__(
        self,
        episode: SubjectEpisode3Mod,
        reward_config: RewardConfig,
        env_config: EnvConfig,
    ):
        super().__init__()

        self.episode = episode
        self.reward_config = reward_config
        self.env_config = env_config

        self.agents = ["cardiac", "somatic", "vascular"]
        self.possible_agents = ["cardiac", "somatic", "vascular"]

        self._stressor_block_info = episode.precompute_stressor_block_info()

        # Observation spaces: embedding + proxy vector per agent
        D_c = episode.cardiac_embed.shape[1]
        D_s = episode.somatic_embed.shape[1]
        D_v = episode.vascular_embed.shape[1]
        D_p = episode.proxies.shape[1]
        seq_len = env_config.seq_len

        self.observation_spaces = {
            "cardiac": spaces.Box(-np.inf, np.inf, shape=(seq_len, D_c + D_p), dtype=np.float32),
            "somatic": spaces.Box(-np.inf, np.inf, shape=(seq_len, D_s + D_p), dtype=np.float32),
            "vascular": spaces.Box(-np.inf, np.inf, shape=(seq_len, D_v + D_p), dtype=np.float32),
        }
        self.action_spaces = {a: spaces.Discrete(2) for a in self.agents}

        self._agent_selector = AgentSelector(self.agents)
        self.t = 0
        self._actions_this_step: Dict[str, int] = {}
        self._cumulative_fp_penalty = 0.0
        self._fp_count = 0

    def observation_space(self, agent: str) -> spaces.Space:
        return self.observation_spaces[agent]

    def action_space(self, agent: str) -> spaces.Space:
        return self.action_spaces[agent]

    def observe(self, agent: str) -> np.ndarray:
        return self.episode.get_observation(self.t, agent)

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        self.t = 0
        self._actions_this_step = {}

        if self.env_config.random_start:
            first_stress = None
            for i in range(len(self.episode.labels)):
                if self.episode.labels[i] == 1:
                    first_stress = i
                    break
            if first_stress is not None and first_stress > self.env_config.seq_len:
                max_start = first_stress - self.env_config.seq_len
                self.t = np.random.randint(0, max_start + 1)

        self.agents = self.possible_agents[:]
        self.rewards = {a: 0.0 for a in self.agents}
        self.terminations = {a: False for a in self.agents}
        self.truncations = {a: False for a in self.agents}
        self.infos = {a: {} for a in self.agents}
        self._cumulative_rewards = {a: 0.0 for a in self.agents}

        self._agent_selector.reset()
        self.agent_selection = self._agent_selector.next()
        self._cumulative_fp_penalty = 0.0
        self._fp_count = 0

    def step(self, action: int):
        if self.terminations[self.agent_selection] or self.truncations[self.agent_selection]:
            self._was_dead_step(action)
            return

        current_agent = self.agent_selection
        self._actions_this_step[current_agent] = action

        if len(self._actions_this_step) == len(self.agents):
            self._resolve_step()
            self._actions_this_step = {}

        self.agent_selection = self._agent_selector.next()
        self._accumulate_rewards()

    def _resolve_step(self):
        """Evaluate joint action with consensus, compute reward, advance time."""
        a_c = self._actions_this_step.get("cardiac", 0)
        a_s = self._actions_this_step.get("somatic", 0)
        a_v = self._actions_this_step.get("vascular", 0)
        votes = a_c + a_s + a_v

        consensus = getattr(self.env_config, "consensus", "majority")
        if consensus == "majority":
            joint_alert = votes >= 2
        elif consensus == "and":
            joint_alert = votes == 3
        else:  # "or"
            joint_alert = votes >= 1

        current_label = int(self.episode.labels[self.t])
        steps_into = int(self._stressor_block_info[self.t, 0])
        total_stressor = int(self._stressor_block_info[self.t, 1])

        reward, terminate = compute_reward(
            joint_alert=joint_alert,
            current_label=current_label,
            steps_into_stressor=steps_into,
            total_stressor_steps=total_stressor,
            config=self.reward_config,
            cumulative_fp_penalty=self._cumulative_fp_penalty,
            fp_count=self._fp_count,
        )

        if joint_alert and current_label == 0:
            self._cumulative_fp_penalty += reward
            self._fp_count += 1

        for a in self.agents:
            self.rewards[a] = reward

        self.t += 1
        episode_over = terminate or self.t >= len(self.episode)

        if episode_over:
            for a in self.agents:
                self.terminations[a] = True

        self.infos = {
            a: {
                "joint_alert": joint_alert,
                "label": current_label,
                "t": self.t - 1,
                "steps_into_stressor": steps_into,
                "total_stressor_steps": total_stressor,
                "votes": {"cardiac": a_c, "somatic": a_s, "vascular": a_v},
            }
            for a in self.agents
        }

    def step_all(
        self, cardiac_action: int, somatic_action: int, vascular_action: int
    ) -> Tuple[Dict[str, np.ndarray], float, bool, dict]:
        """Step all three agents simultaneously (bypasses AEC turn order)."""
        self._actions_this_step = {
            "cardiac": cardiac_action,
            "somatic": somatic_action,
            "vascular": vascular_action,
        }
        self._resolve_step()

        done = any(self.terminations.values())
        obs = {a: self.observe(a) for a in self.agents} if not done else {}
        reward = self.rewards["cardiac"]
        info = self.infos["cardiac"]

        self._agent_selector.reset()
        self.agent_selection = self._agent_selector.next()

        return obs, reward, done, info

    def get_current_obs(self) -> Dict[str, np.ndarray]:
        """Get observations for all three agents at current time step."""
        return {a: self.observe(a) for a in self.agents}
