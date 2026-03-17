"""
Reward function for the stress monitoring environment.

Isolated as a pure function for easy ablation and unit testing.

Focus: stress detection during the stress window. Reward scales by earliness
within the stress window (earlier detection = higher reward).
"""

from __future__ import annotations

from typing import Tuple

from src.config import RewardConfig


def compute_reward(
    joint_alert: bool,
    current_label: int,
    steps_into_stressor: int,
    total_stressor_steps: int,
    config: RewardConfig,
    cumulative_fp_penalty: float = 0.0,
    fp_count: int = 0,
) -> Tuple[float, bool]:
    """
    Compute the shared reward and termination signal.

    Parameters
    ----------
    joint_alert : bool
        True if both agents voted to alert this step (consensus met).
    current_label : int
        0 = baseline, 1 = stress (no pre-stress).
    steps_into_stressor : int
        Steps into the current stress block. -1 if not in stress.
    total_stressor_steps : int
        Total length of the stress block. -1 if not in stress.
    config : RewardConfig
        Reward magnitudes and flags.

    Returns
    -------
    (reward, should_terminate) : Tuple[float, bool]
    """

    if joint_alert:
        if current_label == 1:
            # TRUE POSITIVE — alert during stress
            reward = config.tp_reward
            if config.lead_time_scaling and total_stressor_steps > 0:
                # Scale by how early the alert is within the stress window.
                # Earlier alert -> higher fraction -> higher reward.
                # steps_into_stressor=0 -> scale=1; steps_into_stressor=total -> scale=0
                frac_remaining = 1.0 - (steps_into_stressor / total_stressor_steps)
                scale = max(config.lead_time_floor, min(1.0, frac_remaining))
                reward *= scale
            return reward, True

        # current_label == 0: FALSE POSITIVE — alert during baseline
        repeat_scaled_fp = config.fp_penalty * (config.fp_repeat_scale ** max(fp_count, 0))
        if config.fp_cap is not None:
            cap_value = -abs(config.fp_cap)
            remaining = cap_value - cumulative_fp_penalty
            reward = max(repeat_scaled_fp, remaining)
            reward = min(0.0, reward)  # never give positive reward for FP
        else:
            reward = repeat_scaled_fp
        return reward, config.terminate_on_fp

    # No alert this step
    if current_label == 1:
        # FALSE NEGATIVE — stress is happening, no alert was raised
        return config.fn_penalty, True

    # Waiting during baseline — neutral
    return config.tn_reward + config.baseline_step_penalty, False
