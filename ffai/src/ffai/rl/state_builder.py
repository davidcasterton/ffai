"""
Centralized state construction for the PPO agent.

State vector layout (72 dims):
  [0:4]   Budget context       (4)  rl_budget/200, budget_per_slot, draft_progress, remaining_slots/14
  [4:15]  Opponent pressure    (11) opponent_budget_fractions (budget/200 for each opponent)
  [15:27] My roster state      (12) per-position: [slots_filled_fraction, points_accumulated/500]
                                    positions: QB, RB, WR, TE, D/ST, K (6 positions × 2)
  [27:45] Market state         (18) per-position: [avg_remaining_value/50, top_available_pts/400, scarcity_ratio]
                                    6 positions × 3
  [45:53] Current player       (8)  value_model_points/400, value_model_dollar/80, PAR/200, VORP_dollar/80,
                                    position_onehot (4: RB, WR, QB, other)
  [53:56] Bid context          (3)  current_bid/200, current_bid/fair_value, min_needed_budget/200
  [56:62] Player history       (6)  pts_3yr_avg_norm, proj_ratio_3yr_avg, yoy_pct_change,
                                    weekly_pts_cv, floor_pts_norm, years_in_league_norm
  [62:66] Position strategy    (4)  winning_budget_share, roi_3yr_avg, proj_accuracy_ratio,
                                    scarcity_historical
  [66:72] Opponent tendencies  (6)  aggregated over top-3 budget opponents:
                                    avg_rb_budget_share, avg_wr_budget_share,
                                    avg_high_bid_rate, avg_dollar_one_rate,
                                    avg_bid_per_proj_pt_rb, avg_bid_per_proj_pt_wr

Total: 72 dims
"""

import numpy as np
import torch
from typing import Dict, List, Optional, Any

POSITIONS = ['QB', 'RB', 'WR', 'TE', 'D/ST', 'K']
NUM_POSITIONS = len(POSITIONS)
STATE_DIM = 72

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
    team_name: Optional[str] = None,
    current_player: Optional[Dict[str, Any]] = None,
    current_bid: float = 0.0,
    feature_store=None,
    year: int = 2024,
) -> torch.Tensor:
    """
    Build a 72-dim state tensor from the simulation state dict.

    Args:
        sim_state: dict returned by AuctionDraftSimulator.get_state() or
                   get_state_for(team_name) for perspective-aware observation.
        team_name: the team whose perspective this state represents (for
                   documentation; the state dict must already be perspective-
                   correct, built via sim.get_state_for(team_name)).
        current_player: player dict being currently bid on (can be None during nomination)
        current_bid: current highest bid amount (0 if nominating)
        feature_store: optional FeatureStore instance for dims 56-71
        year: season year for feature_store lookups

    Returns:
        state tensor of shape (72,)
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
    idx = 56  # idx = 56

    # ------------------------------------------------------------------
    # [56:62] Player history (6 features) — requires feature_store
    # ------------------------------------------------------------------
    if feature_store is not None and current_player is not None:
        pid = str(current_player.get('player_id', ''))
        pos = current_player.get('position', '')
        ph = feature_store.get_player_features(pid, year, position=pos)

        features[56] = float(ph.get('pts_3yr_avg', 0.0) or 0.0) / POINTS_MAX
        features[57] = float(np.clip(ph.get('proj_ratio_3yr_avg', 1.0) or 1.0, 0.0, 3.0)) / 3.0
        features[58] = float(np.clip(ph.get('yoy_pct_change', 0.0) or 0.0, -1.0, 1.0))
        features[59] = float(np.clip(ph.get('weekly_pts_cv', 0.0) or 0.0, 0.0, 3.0)) / 3.0
        features[60] = float(ph.get('floor_pts', 0.0) or 0.0) / 40.0
        features[61] = float(min(1.0, (ph.get('years_in_league', 0.0) or 0.0) / 10.0))
    idx = 62  # idx = 62

    # ------------------------------------------------------------------
    # [62:66] Position strategy (4 features)
    # ------------------------------------------------------------------
    if feature_store is not None and current_player is not None:
        pos = current_player.get('position', '')
        ps = feature_store.get_position_strategy(pos, year)

        features[62] = float(ps.get('winning_budget_share', 0.0) or 0.0)
        features[63] = float(np.clip(ps.get('roi_mean', 0.0) or 0.0, 0.0, 20.0)) / 20.0
        features[64] = float(np.clip(ps.get('proj_accuracy_ratio', 1.0) or 1.0, 0.0, 3.0)) / 3.0
        features[65] = float(ps.get('budget_share_pct', 0.0) or 0.0)
    idx = 66  # idx = 66

    # ------------------------------------------------------------------
    # [66:72] Opponent tendencies (6 features, top-3 budget opponents)
    # ------------------------------------------------------------------
    if feature_store is not None:
        opponent_tendencies = sim_state.get('opponent_tendencies', [])
        if opponent_tendencies:
            rb_shares, wr_shares = [], []
            high_bid_rates, dollar_one_rates = [], []
            rb_efficiencies, wr_efficiencies = [], []

            for tend in opponent_tendencies[:3]:
                rb_shares.append(float(tend.get('rb_budget_share', 0.0) or 0.0))
                wr_shares.append(float(tend.get('wr_budget_share', 0.0) or 0.0))
                high_bid_rates.append(float(tend.get('high_bid_rate', 0.0) or 0.0))
                dollar_one_rates.append(float(tend.get('dollar_one_rate', 0.0) or 0.0))
                rb_efficiencies.append(float(tend.get('bid_per_proj_pt_rb', 0.0) or 0.0))
                wr_efficiencies.append(float(tend.get('bid_per_proj_pt_wr', 0.0) or 0.0))

            features[66] = float(np.mean(rb_shares)) if rb_shares else 0.0
            features[67] = float(np.mean(wr_shares)) if wr_shares else 0.0
            features[68] = float(np.mean(high_bid_rates)) if high_bid_rates else 0.0
            features[69] = float(np.mean(dollar_one_rates)) if dollar_one_rates else 0.0
            features[70] = float(np.clip(np.mean(rb_efficiencies), 0.0, 1.0)) if rb_efficiencies else 0.0
            features[71] = float(np.clip(np.mean(wr_efficiencies), 0.0, 1.0)) if wr_efficiencies else 0.0

    return torch.from_numpy(features)
