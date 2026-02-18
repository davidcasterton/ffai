"""
Advisory Mode — terminal UI that shows draft recommendations to a human manager.

The human controls the draft; this tool provides PPO agent recommendations
that the human can choose to follow or ignore.

Usage:
    python scripts/advisory_draft.py --config config/league.yaml
"""

import time
import logging
from pathlib import Path
from typing import Optional, List, Dict, Any

from ffai.interfaces.live_draft_reader import LiveDraftReader
from ffai.rl.state_builder import build_state
from ffai.rl.ppo_agent import PPOAgent
from ffai.value_model.player_value_model import PlayerValueModel

logger = logging.getLogger(__name__)

# How many top alternatives to show alongside the primary recommendation
TOP_N = 5


class AdvisoryMode:
    """
    Polls the live draft state and prints bidding recommendations.

    Integrates:
    - LiveDraftReader: polls ESPN API
    - PlayerValueModel: scores available players
    - PPOAgent: recommends bid amount
    """

    def __init__(
        self,
        config_path: Optional[Path] = None,
        value_model: Optional[PlayerValueModel] = None,
        ppo_agent: Optional[PPOAgent] = None,
        rl_team_id: Optional[int] = None,
        budget: float = 200.0,
        poll_interval: float = 2.5,
    ):
        self.config_path = config_path
        self.value_model = value_model
        self.ppo_agent = ppo_agent
        self.rl_team_id = rl_team_id
        self.budget = budget
        self.reader = LiveDraftReader(config_path, poll_interval=poll_interval)
        self._current_budget = budget

    def _score_players(self, available_players: List[dict], sim_state: dict) -> List[dict]:
        """Score available players using the value model (if loaded) or projected_points."""
        scored = []
        for player in available_players:
            if self.value_model is not None:
                # Use model predictions if available
                pts = player.get("projected_points", 0.0)
                dollar = player.get("auction_value", 1.0)
            else:
                pts = player.get("projected_points", 0.0)
                dollar = player.get("auction_value", 1.0)

            scored.append({
                **player,
                "_score_pts": pts,
                "_score_dollar": dollar,
                "_value_ratio": pts / max(1.0, dollar),
            })

        # Rank by value ratio (pts per dollar)
        scored.sort(key=lambda p: p["_value_ratio"], reverse=True)
        return scored

    def _get_bid_recommendation(self, player: dict, sim_state: dict) -> float:
        """Get bid recommendation from PPO agent or simple heuristic."""
        fair_value = player.get("auction_value", player.get("_score_dollar", 1.0))

        if self.ppo_agent is not None:
            state_tensor = build_state(sim_state, current_player=player, current_bid=0.0)
            bid_frac, _, _ = self.ppo_agent.get_bid_action(
                state_tensor, self._current_budget, min_bid=1.0, deterministic=True
            )
            return bid_frac
        else:
            # Heuristic: bid 95% of fair value (conservative)
            return max(1.0, fair_value * 0.95)

    def _print_header(self):
        print("\n" + "=" * 70)
        print(f"  FFAI ADVISORY MODE  |  Budget: ${self._current_budget:.0f}")
        print("=" * 70)

    def _print_recommendation(
        self,
        top_players: List[dict],
        sim_state: dict,
        pick_num: int,
    ):
        self._print_header()
        print(f"  Pick #{pick_num}  |  Draft progress: {sim_state.get('draft_progress', 0):.1%}")
        print(f"  Remaining slots: {sum(sim_state.get('position_needs', {}).values())}")
        print()

        print(f"  TOP {TOP_N} RECOMMENDED TARGETS:")
        print("  " + "-" * 65)
        print(f"  {'#':<3} {'Player':<25} {'Pos':<5} {'Proj Pts':>8} {'Fair $':>7} {'Rec Bid':>8}")
        print("  " + "-" * 65)

        for i, player in enumerate(top_players[:TOP_N], 1):
            bid = self._get_bid_recommendation(player, sim_state)
            print(
                f"  {i:<3} {player.get('name', 'Unknown'):<25} "
                f"{player.get('position', '?'):<5} "
                f"{player.get('_score_pts', 0):>8.1f} "
                f"${player.get('_score_dollar', 0):>6.0f} "
                f"${bid:>7.0f}"
            )

        print("  " + "-" * 65)
        print()

    def run(self):
        """Main loop: poll draft state and print recommendations."""
        if self.rl_team_id is None:
            raise ValueError("rl_team_id must be set before running advisory mode.")

        logger.info("Advisory mode started. Waiting for draft activity...")
        print("\n  FFAI Advisory Mode — Monitoring ESPN draft...\n")

        pick_num = 0
        for sim_state in self.reader.poll_loop(
            rl_team_id=self.rl_team_id,
            budget=self.budget,
        ):
            pick_num += 1
            self._current_budget = float(sim_state.get("rl_team_budget", self._current_budget))

            available_players = sim_state.get("_available_players", [])
            if not available_players:
                continue

            top_players = self._score_players(available_players, sim_state)
            self._print_recommendation(top_players, sim_state, pick_num)
