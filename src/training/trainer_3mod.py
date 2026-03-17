"""PPO trainer for 3-agent stress monitoring environment.

Handles rollout collection and PPO updates for three independent
actor-critic agents that share a common reward signal.

Key optimization: since observations are pre-computed from frozen encoders,
we batch all forward passes per episode instead of doing them one at a time.
"""

from __future__ import annotations

from typing import Dict, List

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical

from src.data.dataset_3mod import SubjectEpisode3Mod
from src.envs.stress_env_3mod import StressMonitor3ModEnv
from src.models.actor_critic_3mod import ThreeModActorCritic
from src.envs.reward import compute_reward


AGENT_NAMES = ["cardiac", "somatic", "vascular"]


def collect_rollout_batched(
    episode: SubjectEpisode3Mod,
    agents: ThreeModActorCritic,
    reward_config,
    env_config,
    device: torch.device,
) -> Dict[str, dict]:
    """Batched rollout: forward-pass all timesteps at once, then compute rewards.

    Since observations are pre-computed, we don't need to step through the env
    one timestep at a time for the forward pass. We batch all obs, get all
    actions/values/log_probs in one GPU call, then resolve rewards sequentially.
    """
    T = len(episode)
    stressor_info = episode.precompute_stressor_block_info()
    consensus = getattr(env_config, "consensus", "majority")

    # Pre-compute ALL observations for each agent — one big tensor
    all_obs = {}
    for a in AGENT_NAMES:
        obs_list = [episode.get_observation(t, a) for t in range(T)]
        all_obs[a] = torch.from_numpy(np.array(obs_list, dtype=np.float32)).to(device)

    # Batch forward pass — one call per agent for the entire episode
    all_logits = {}
    all_values = {}
    with torch.no_grad():
        for a in AGENT_NAMES:
            logits, values = agents.get_agent(a).forward(all_obs[a])
            all_logits[a] = logits.cpu()
            all_values[a] = values.squeeze(-1).cpu()

    # Sample actions from logits
    all_actions = {}
    all_log_probs = {}
    for a in AGENT_NAMES:
        dist = Categorical(logits=all_logits[a])
        actions = dist.sample()
        all_actions[a] = actions
        all_log_probs[a] = dist.log_prob(actions)

    # Per-agent individual rewards (fixes lazy agent / free-rider problem)
    # Each agent gets TP/FP/FN/TN based on its OWN action, not the consensus
    agent_rewards = {a: np.zeros(T, dtype=np.float32) for a in AGENT_NAMES}
    dones = np.zeros(T, dtype=bool)

    tp_reward = reward_config.tp_reward
    fp_penalty = reward_config.fp_penalty
    fn_penalty = reward_config.fn_penalty
    tn_reward = reward_config.tn_reward

    for t in range(T):
        label = int(episode.labels[t])
        for a in AGENT_NAMES:
            action = int(all_actions[a][t].item())
            if action == 1:  # agent alerts
                if label == 1:
                    agent_rewards[a][t] = tp_reward    # correct alert
                else:
                    agent_rewards[a][t] = fp_penalty   # false alarm
            else:            # agent waits
                if label == 1:
                    agent_rewards[a][t] = fn_penalty   # missed stress
                else:
                    agent_rewards[a][t] = tn_reward    # correct wait
        dones[t] = False

    # Mark last step as done
    dones[-1] = True

    trajectories = {}
    for a in AGENT_NAMES:
        trajectories[a] = {
            "obs": all_obs[a].cpu().numpy(),
            "actions": all_actions[a].numpy().tolist(),
            "log_probs": all_log_probs[a].numpy().tolist(),
            "values": all_values[a].numpy().tolist(),
            "rewards": agent_rewards[a].tolist(),
            "dones": dones.tolist(),
        }

    return trajectories


def compute_gae(
    rewards: List[float],
    values: List[float],
    dones: List[bool],
    gamma: float,
    gae_lambda: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute Generalized Advantage Estimation."""
    T = len(rewards)
    advantages = np.zeros(T, dtype=np.float32)
    gae = 0.0

    for t in reversed(range(T)):
        if dones[t]:
            next_value = 0.0
        else:
            next_value = values[t + 1] if t + 1 < T else 0.0

        delta = rewards[t] + gamma * next_value - values[t]
        gae = delta + gamma * gae_lambda * (0.0 if dones[t] else 1.0) * gae
        advantages[t] = gae

    returns = advantages + np.array(values, dtype=np.float32)
    return advantages, returns


def ppo_update(
    agent_net: nn.Module,
    obs: torch.Tensor,
    actions: torch.Tensor,
    old_log_probs: torch.Tensor,
    advantages: torch.Tensor,
    returns: torch.Tensor,
    optimizer: torch.optim.Optimizer,
    clip_epsilon: float = 0.2,
    entropy_coef: float = 0.01,
    value_coef: float = 0.5,
    max_grad_norm: float = 0.5,
    ppo_epochs: int = 4,
    mini_batch_size: int = 64,
) -> dict:
    """Run PPO update on collected trajectory data for one agent."""
    N = obs.shape[0]
    total_policy_loss = 0.0
    total_value_loss = 0.0
    total_entropy = 0.0
    n_updates = 0

    for _ in range(ppo_epochs):
        indices = torch.randperm(N)
        for start in range(0, N, mini_batch_size):
            end = min(start + mini_batch_size, N)
            idx = indices[start:end]

            mb_obs = obs[idx]
            mb_actions = actions[idx]
            mb_old_log_probs = old_log_probs[idx]
            mb_advantages = advantages[idx]
            mb_returns = returns[idx]

            # Normalize advantages
            mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

            log_probs, values, entropy = agent_net.evaluate(mb_obs, mb_actions)

            ratio = torch.exp(log_probs - mb_old_log_probs)
            surr1 = ratio * mb_advantages
            surr2 = torch.clamp(ratio, 1.0 - clip_epsilon, 1.0 + clip_epsilon) * mb_advantages
            policy_loss = -torch.min(surr1, surr2).mean()

            value_loss = nn.functional.mse_loss(values, mb_returns)
            entropy_loss = -entropy.mean()

            loss = policy_loss + value_coef * value_loss + entropy_coef * entropy_loss

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(agent_net.parameters(), max_grad_norm)
            optimizer.step()

            total_policy_loss += policy_loss.item()
            total_value_loss += value_loss.item()
            total_entropy += entropy.mean().item()
            n_updates += 1

    return {
        "policy_loss": total_policy_loss / max(n_updates, 1),
        "value_loss": total_value_loss / max(n_updates, 1),
        "entropy": total_entropy / max(n_updates, 1),
    }


def train_episode_batch(
    env: StressMonitor3ModEnv,
    agents: ThreeModActorCritic,
    optimizers: Dict[str, torch.optim.Optimizer],
    episodes: List[SubjectEpisode3Mod],
    device: torch.device,
    gamma: float = 0.99,
    gae_lambda: float = 0.95,
    clip_epsilon: float = 0.2,
    entropy_coef: float = 0.01,
    value_coef: float = 0.5,
    max_grad_norm: float = 0.5,
    ppo_epochs: int = 4,
    mini_batch_size: int = 64,
    episodes_per_update: int = 4,
) -> dict:
    """Collect multiple episodes and perform PPO update."""
    all_trajectories = {a: {"obs": [], "actions": [], "log_probs": [], "values": [], "rewards": [], "dones": []} for a in AGENT_NAMES}

    total_reward = 0.0
    total_steps = 0

    for _ in range(episodes_per_update):
        ep = episodes[np.random.randint(len(episodes))]

        traj = collect_rollout_batched(
            ep, agents, env.reward_config, env.env_config, device
        )

        for a in AGENT_NAMES:
            all_trajectories[a]["obs"].append(traj[a]["obs"])
            all_trajectories[a]["actions"].extend(traj[a]["actions"])
            all_trajectories[a]["log_probs"].extend(traj[a]["log_probs"])
            all_trajectories[a]["values"].extend(traj[a]["values"])
            all_trajectories[a]["rewards"].extend(traj[a]["rewards"])
            all_trajectories[a]["dones"].extend(traj[a]["dones"])

        total_reward += sum(traj["cardiac"]["rewards"])
        total_steps += len(traj["cardiac"]["rewards"])

    # PPO update per agent
    stats = {"mean_reward": total_reward / episodes_per_update, "mean_steps": total_steps / episodes_per_update}

    for a in AGENT_NAMES:
        T = len(all_trajectories[a]["rewards"])
        if T == 0:
            continue

        advantages, returns = compute_gae(
            all_trajectories[a]["rewards"],
            all_trajectories[a]["values"],
            all_trajectories[a]["dones"],
            gamma, gae_lambda,
        )

        obs_np = np.concatenate(all_trajectories[a]["obs"], axis=0)
        obs_t = torch.from_numpy(obs_np).to(device)
        actions_t = torch.tensor(all_trajectories[a]["actions"], dtype=torch.long, device=device)
        old_lp_t = torch.tensor(all_trajectories[a]["log_probs"], dtype=torch.float32, device=device)
        adv_t = torch.from_numpy(advantages).to(device)
        ret_t = torch.from_numpy(returns).to(device)

        agent_stats = ppo_update(
            agents.get_agent(a), obs_t, actions_t, old_lp_t, adv_t, ret_t,
            optimizers[a],
            clip_epsilon=clip_epsilon, entropy_coef=entropy_coef,
            value_coef=value_coef, max_grad_norm=max_grad_norm,
            ppo_epochs=ppo_epochs, mini_batch_size=mini_batch_size,
        )
        stats[f"{a}_policy_loss"] = agent_stats["policy_loss"]
        stats[f"{a}_entropy"] = agent_stats["entropy"]

    return stats
