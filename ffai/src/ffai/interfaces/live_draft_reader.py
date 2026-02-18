"""
Live Draft Reader — polls the ESPN REST API during a real draft.

Polls: GET https://lm-api-reads.fantasy.espn.com/apis/v3/games/ffl/seasons/{year}/segments/0/leagues/{league_id}?view=mDraftDetail

Auth: ESPN cookies (SWID, espn_s2) loaded from config/league.yaml.
Poll interval: 2-3 seconds (well within the 60-90s nomination timer).

Converts ESPN API responses to the same state dict format used by the
AuctionDraftSimulator so the PPO agent can make decisions on live data.
"""

import time
import requests
import logging
from pathlib import Path
from typing import Optional, Dict, Any, List

from ffai.data.espn_scraper import load_league_config

logger = logging.getLogger(__name__)

ESPN_DRAFT_API = (
    "https://lm-api-reads.fantasy.espn.com/apis/v3/games/ffl"
    "/seasons/{year}/segments/0/leagues/{league_id}?view=mDraftDetail"
)

POSITIONS = ['QB', 'RB', 'WR', 'TE', 'D/ST', 'K']


class LiveDraftReader:
    """
    Polls ESPN draft API at regular intervals and parses state.

    Usage:
        reader = LiveDraftReader(config_path)
        for state in reader.poll_loop():
            # state is a dict compatible with state_builder.build_state()
            recommendation = agent.recommend(state)
    """

    def __init__(
        self,
        config_path: Optional[Path] = None,
        poll_interval: float = 2.5,
    ):
        config = load_league_config(config_path)
        self.league_id = config["league"]["league_id"]
        self.year = config["league"]["year"]
        self.swid = config["auth"]["swid"]
        self.espn_s2 = config["auth"]["espn_s2"]
        self.poll_interval = poll_interval

        self.session = requests.Session()
        self.session.cookies.set("SWID", self.swid)
        self.session.cookies.set("espn_s2", self.espn_s2)

        self._last_pick_count = -1
        self._available_players_cache: List[dict] = []

    def _fetch_draft_state(self) -> Optional[dict]:
        """Fetch current draft state from ESPN API."""
        url = ESPN_DRAFT_API.format(year=self.year, league_id=self.league_id)
        try:
            resp = self.session.get(url, timeout=10)
            resp.raise_for_status()
            return resp.json()
        except requests.RequestException as e:
            logger.warning(f"Failed to fetch draft state: {e}")
            return None

    def convert_to_simulation_state(
        self,
        api_response: dict,
        rl_team_id: int,
        budget: float = 200.0,
    ) -> Optional[dict]:
        """
        Convert an ESPN draft API response to the state dict format used
        by the AuctionDraftSimulator and state_builder.

        Args:
            api_response: raw JSON from ESPN draft API
            rl_team_id: the ESPN team_id of the team we're controlling
            budget: starting budget

        Returns:
            state dict compatible with state_builder.build_state(), or None if parsing fails
        """
        try:
            draft_detail = api_response.get("draftDetail", {})
            picks = draft_detail.get("picks", [])

            # Build a map of team_id → budget spent
            team_budgets: Dict[int, float] = {}
            player_ids_drafted: List[int] = []

            for pick in picks:
                team_id = pick.get("teamId")
                bid = pick.get("bidAmount", 0)
                if team_id is not None:
                    team_budgets[team_id] = team_budgets.get(team_id, 0) + bid
                player_id = pick.get("playerId")
                if player_id:
                    player_ids_drafted.append(player_id)

            # Figure out RL team remaining budget
            rl_spent = team_budgets.get(rl_team_id, 0)
            rl_budget = budget - rl_spent

            # Opponent budgets (sorted)
            opponent_budgets = [
                budget - spent
                for tid, spent in team_budgets.items()
                if tid != rl_team_id
            ]

            # Get available players (not yet drafted)
            all_players = self._parse_available_players(api_response, player_ids_drafted)
            self._available_players_cache = all_players

            # Position scarcity — count available per position
            position_scarcity = {}
            for pos in POSITIONS:
                available_count = sum(1 for p in all_players if p["position"] == pos)
                # Simple heuristic: scarcity = max(0, 12 - available_count)
                position_scarcity[pos] = max(0, 12 - available_count)

            position_values = {}
            for pos in POSITIONS:
                pos_players = [p for p in all_players if p["position"] == pos]
                if pos_players:
                    position_values[pos] = {
                        "avg_value": sum(p.get("auction_value", 0) for p in pos_players) / len(pos_players),
                        "avg_points": sum(p.get("projected_points", 0) for p in pos_players) / len(pos_players),
                    }

            # RL team's current roster composition (rough — from pick history)
            rl_picks = [p for p in picks if p.get("teamId") == rl_team_id]
            position_needs = {pos: 1 for pos in POSITIONS}  # placeholder — needs roster slot data

            state = {
                "rl_team_budget": rl_budget,
                "opponent_budgets": opponent_budgets,
                "draft_turn": len(picks),
                "position_needs": position_needs,
                "position_counts": {},
                "position_scarcity": position_scarcity,
                "position_values": position_values,
                "remaining_budget_per_need": rl_budget / max(1, sum(position_needs.values())),
                "draft_progress": len(picks) / max(1, 12 * 14),  # 12 teams × 14 slots
                "total_team_points": 0.0,
                "_raw_picks": picks,
                "_available_players": all_players,
            }

            return state

        except Exception as e:
            logger.error(f"Error parsing ESPN draft API response: {e}")
            logger.exception("Stack trace:")
            return None

    def _parse_available_players(self, api_response: dict, drafted_ids: List[int]) -> List[dict]:
        """Extract available (undrafted) players from the API response."""
        players = []
        drafted_set = set(drafted_ids)

        players_data = api_response.get("players", [])
        for entry in players_data:
            player_pool = entry.get("playerPoolEntry", {})
            player = player_pool.get("player", {})
            player_id = player.get("id")

            if player_id in drafted_set:
                continue

            eligibility = player.get("eligibleSlots", [])
            position = self._slot_to_position(eligibility)
            projected = player_pool.get("keeperValue", 0)
            auction_val = player_pool.get("auctionValueAverage", 0)

            players.append({
                "player_id": player_id,
                "name": player.get("fullName", "Unknown"),
                "position": position,
                "projected_points": float(projected),
                "auction_value": float(auction_val),
            })

        return players

    def _slot_to_position(self, eligible_slots: List[int]) -> str:
        """Map ESPN slot IDs to position string."""
        # ESPN slot IDs: 0=QB, 2=RB, 4=WR, 6=TE, 16=D/ST, 17=K
        slot_map = {0: "QB", 2: "RB", 4: "WR", 6: "TE", 16: "D/ST", 17: "K"}
        for slot_id, pos in slot_map.items():
            if slot_id in eligible_slots:
                return pos
        return "UNKNOWN"

    def get_available_players(self) -> List[dict]:
        """Return the cached list of available players from the last poll."""
        return self._available_players_cache

    def poll_loop(
        self,
        rl_team_id: int,
        budget: float = 200.0,
        max_polls: int = 10000,
    ):
        """
        Generator that yields state dicts on each new pick.

        Only yields when the pick count has changed (new nomination or bid).
        """
        polls = 0
        while polls < max_polls:
            api_response = self._fetch_draft_state()
            if api_response is None:
                time.sleep(self.poll_interval)
                polls += 1
                continue

            picks = api_response.get("draftDetail", {}).get("picks", [])
            pick_count = len(picks)

            if pick_count != self._last_pick_count:
                self._last_pick_count = pick_count
                state = self.convert_to_simulation_state(api_response, rl_team_id, budget)
                if state:
                    logger.info(f"Poll #{polls}: {pick_count} picks made")
                    yield state

            time.sleep(self.poll_interval)
            polls += 1
