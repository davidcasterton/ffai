"""
Reward shaping for the PPO auction draft agent.

All rewards normalized to [-1, +1] scale so the value function
can be trained stably alongside the policy.

Mid-draft (per pick):
  base   = (fair_dollar_value - bid_amount) / 200   ∈ [-0.5, +0.5]
  bonus  = +0.1 if position was needed
  penalty = -0.2 if remaining budget < min_needed after purchase

Terminal (end of season):
  standing reward = {1st: 5.0, 2nd: 3.0, 3rd: 2.0, 4th: 1.0, else 0}
  wins bonus = wins * 0.3

Prior implementation ranged from 200 to 500 in the same reward space
and did per-pick scalar updates with no computational graph — both fixed here.
"""

from typing import Optional


def mid_draft_reward(
    fair_dollar_value: float,
    bid_amount: float,
    position_needed: bool,
    budget_safe: bool,
    budget_max: float = 200.0,
) -> float:
    """
    Reward signal after winning a bid during the draft.

    Args:
        fair_dollar_value: VORP-derived fair auction dollar value for the player
        bid_amount: amount paid at auction
        position_needed: True if the RL team still needed this position
        budget_safe: True if remaining budget >= minimum needed to complete roster
        budget_max: total budget (default $200)

    Returns:
        float reward ∈ approximately [-0.7, +0.6]
    """
    # Value efficiency: positive if we got a bargain, negative if we overpaid
    base = (fair_dollar_value - bid_amount) / budget_max  # ∈ [-0.5, +0.5] typically

    # Position need bonus
    if position_needed:
        base += 0.1

    # Budget safety penalty
    if not budget_safe:
        base -= 0.2

    return float(base)


def terminal_reward(
    standing_position: int,
    wins: int,
    total_weeks: int = 17,
) -> float:
    """
    Terminal reward at the end of the season.

    Args:
        standing_position: 0-indexed final standing (0 = 1st place)
        wins: number of weekly wins
        total_weeks: total weeks in the season (default 17)

    Returns:
        float reward (typically 0 to 10)
    """
    standing_rewards = {
        0: 5.0,   # 1st place
        1: 3.0,   # 2nd place
        2: 2.0,   # 3rd place
        3: 1.0,   # 4th place
    }
    standing_r = standing_rewards.get(standing_position, 0.0)
    win_r = wins * 0.3

    return standing_r + win_r


def normalize_terminal_reward(raw_reward: float, max_possible: float = 10.1) -> float:
    """Scale terminal reward to roughly [0, 1] for stable training."""
    return raw_reward / max_possible
