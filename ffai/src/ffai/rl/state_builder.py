"""
Centralized state construction for the PPO agent.

State vector layout (~56 dims):
  [0:4]   Budget context       (4)  rl_budget/200, budget_per_slot, draft_progress, remaining_slots/14
  [4:15]  Opponent pressure    (11) opponent_budget_fractions (budget/200 for each opponent)
  [15:27] My roster state      (12) per-position: [slots_filled_fraction, points_accumulated/500]
                                    positions: QB, RB, WR, TE, D/ST, K (6 positions × 2)
  [27:45] Market state         (18) per-position: [avg_remaining_value/50, top_available_pts/400, scarcity_ratio]
                                    6 positions × 3
  [45:53] Current player       (8)  value_model_points/400, value_model_dollar/80, PAR/200, VORP_dollar/80,
                                    position_onehot (4: RB, WR, QB, other)
  [53:56] Bid context          (3)  current_bid/200, current_bid/fair_value, min_needed_budget/200

Total: 56 dims
"""

import numpy as np
import torch
from typing import Dict, List, Optional, Any

POSITIONS = ['QB', 'RB', 'WR', 'TE', 'D/ST', 'K']
NUM_POSITIONS = len(POSITIONS)
STATE_DIM = 56

BUDGET_MAX = 200.0
POINTS_MAX = 400.0          # max projected season points any player could have
DOLLAR_MAX = 80.0           # max auction value cap
PAR_MAX = 200.0             # max PAR (points above replacement)
POINTS_ACCUM_MAX = 500.0    # max accumulated points on roster

# Number of starter slots per position (for slot_filled_fraction)
POSITION_STARTER_SLOTS = {
    'QB': 1,
    'RB': 2,
    'WR': 2,
    'TE': 1,
    'D/ST': 1,
    'K': 1,
}


def build_state(
    sim_state: Dict[str, Any],
    current_player: Optional[Dict[str, Any]] = None,
    current_bid: float = 0.0,
) -> torch.Tensor:
    """
    Build a 56-dim state tensor from the simulation state dict.

    Args:
        sim_state: dict returned by AuctionDraftSimulator.get_state()
        current_player: player dict being currently bid on (can be None during nomination)
        current_bid: current highest bid amount (0 if nominating)

    Returns:
        state tensor of shape (56,)
    """
    features = np.zeros(STATE_DIM, dtype=np.float32)
    idx = 0

    # ------------------------------------------------------------------
    # [0:4] Budget context
    # ------------------------------------------------------------------
    rl_budget = float(sim_state.get('rl_team_budget', 200))
    remaining_slots = max(1, sum(
        v for v in sim_state.get('position_needs', {}).values()
    ))
    budget_per_slot = rl_budget / remaining_slots
    draft_progress = float(sim_state.get('draft_progress', 0.0))
    total_roster_slots = 14.0  # standard ESPN roster size

    features[0] = rl_budget / BUDGET_MAX
    features[1] = budget_per_slot / BUDGET_MAX
    features[2] = draft_progress
    features[3] = remaining_slots / total_roster_slots
    idx = 4

    # ------------------------------------------------------------------
    # [4:15] Opponent pressure (11 opponents)
    # ------------------------------------------------------------------
    opp_budgets = sim_state.get('opponent_budgets', [])
    for i in range(11):
        if i < len(opp_budgets):
            features[idx + i] = float(opp_budgets[i]) / BUDGET_MAX
        else:
            features[idx + i] = 0.0
    idx += 11  # idx = 15

    # ------------------------------------------------------------------
    # [15:27] My roster state (6 positions × 2 features)
    # ------------------------------------------------------------------
    position_needs = sim_state.get('position_needs', {})
    position_values = sim_state.get('position_values', {})

    for pos in POSITIONS:
        starter_slots = POSITION_STARTER_SLOTS.get(pos, 1)
        filled = starter_slots - position_needs.get(pos, 0)
        filled = max(0, filled)
        features[idx] = filled / starter_slots  # slots_filled_fraction
        # points accumulated at this position (from sim_state or 0)
        pos_pts = position_values.get(pos, {}).get('avg_points', 0.0)
        features[idx + 1] = float(pos_pts) / POINTS_ACCUM_MAX
        idx += 2  # idx = 27

    # ------------------------------------------------------------------
    # [27:45] Market state (6 positions × 3 features)
    # ------------------------------------------------------------------
    position_scarcity = sim_state.get('position_scarcity', {})

    for pos in POSITIONS:
        pos_info = position_values.get(pos, {})
        avg_val = float(pos_info.get('avg_value', 0.0))
        avg_pts = float(pos_info.get('avg_points', 0.0))
        scarcity = float(position_scarcity.get(pos, 0.0))

        # scarcity_ratio: how many more needed than available (clipped to [0,1])
        # We normalize by 24 (max RB need across all teams)
        scarcity_ratio = min(1.0, scarcity / 24.0)

        features[idx] = avg_val / DOLLAR_MAX
        features[idx + 1] = avg_pts / POINTS_MAX
        features[idx + 2] = scarcity_ratio
        idx += 3  # idx = 45

    # ------------------------------------------------------------------
    # [45:53] Current player (8 features)
    # ------------------------------------------------------------------
    if current_player is not None:
        pts_hat = float(current_player.get('projected_points', 0.0))
        dollar_hat = float(current_player.get('auction_value', 0.0))
        par = float(current_player.get('PAR', 0.0))
        vorp = float(current_player.get('VORP_dollar', dollar_hat))
        pos = current_player.get('position', '')

        features[idx] = pts_hat / POINTS_MAX
        features[idx + 1] = dollar_hat / DOLLAR_MAX
        features[idx + 2] = par / PAR_MAX
        features[idx + 3] = vorp / DOLLAR_MAX
        # Position one-hot (4 bins: RB, WR, QB, other)
        features[idx + 4] = 1.0 if pos == 'RB' else 0.0
        features[idx + 5] = 1.0 if pos == 'WR' else 0.0
        features[idx + 6] = 1.0 if pos == 'QB' else 0.0
        features[idx + 7] = 1.0 if pos not in ('RB', 'WR', 'QB') else 0.0
    idx += 8  # idx = 53

    # ------------------------------------------------------------------
    # [53:56] Bid context
    # ------------------------------------------------------------------
    fair_value = float(current_player.get('VORP_dollar', 1.0)) if current_player else 1.0
    min_needed = float(sim_state.get('remaining_budget_per_need', 0.0)) * remaining_slots

    features[53] = current_bid / BUDGET_MAX
    features[54] = current_bid / max(1.0, fair_value)  # ratio of bid to fair value
    features[55] = min(1.0, min_needed / BUDGET_MAX)

    return torch.from_numpy(features)
