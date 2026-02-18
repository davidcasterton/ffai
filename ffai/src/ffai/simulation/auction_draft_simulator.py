"""
Auction Draft Simulator — fixed version.

Bug fixes from rl_model.py / original auction_draft_simulator.py:
1. Removed `import pdb; pdb.set_trace()` at the round limit safety check.
2. Fixed KeyError on `position_counts["RB/WR/TE"]` — the ESPN API key for the
   FLEX slot is "RB/WR/TE" but older leagues may use "FLEX". Now uses a helper
   `_get_flex_key()` to look up whichever key is present.
3. Integrated with new module structure (ffai.simulation, ffai.data.espn_scraper).
"""

import pandas as pd
import numpy as np
from pathlib import Path
import json
from ffai import get_logger
from ffai.data.espn_scraper import ESPNDraftScraper
import copy
import logging

logger = get_logger(__name__)


class InvalidRosterException(Exception):
    """Exception raised when a team's roster is invalid."""
    def __init__(self, team_name: str, message: str):
        self.team_name = team_name
        self.message = message
        super().__init__(f"Invalid roster for {team_name}\n{message}")


class AuctionDraftSimulator:
    def __init__(self, year, budget=200, rl_team="Team 1", rl_model=None):
        self.year = year
        self.budget = budget
        self.rl_team_name = rl_team
        self.rl_model = rl_model
        self.draft_completed: bool = False

        self.data_dir = Path(__file__).parent.parent / "data/raw"

        scraper = ESPNDraftScraper()
        self.draft_df, self.stats_df, self.weekly_df, self.predraft_df, self.settings = scraper.load_or_fetch_data(self.year)

        self.teams: dict = self.initialize_teams()
        self.nomination_order: list = list(self.teams.keys())
        self.available_players: list = self.initialize_available_players()
        self.players_drafted: list = []

    # ------------------------------------------------------------------
    # Initialization
    # ------------------------------------------------------------------

    def initialize_teams(self) -> dict:
        """Initialize teams with mandatory slot budget reserves."""
        teams: dict = {}
        for i in range(12):
            team_name = f"Team {i+1}"

            roster_slots = {
                "QB": None,
                "RB 1": None,
                "RB 2": None,
                "WR 1": None,
                "WR 2": None,
                "TE": None,
                "FLEX": None,   # RB/WR/TE
                "D/ST": None,
            }

            if self.settings["position_slot_counts"].get("K"):
                roster_slots["K"] = None

            num_bench_slots = self.settings["position_slot_counts"].get("BE", 0)
            for j in range(num_bench_slots):
                roster_slots[f"BENCH {j+1}"] = None

            teams[team_name] = {
                "current_budget": copy.deepcopy(self.budget),
                "roster": roster_slots,
                "roster_completed": False,
                "position_counts": {
                    pos: {"required": count, "filled": 0}
                    for pos, count in self.settings["position_slot_counts"].items()
                }
            }

        return teams

    def _get_flex_key(self, position_counts: dict) -> str:
        """
        Return the key used for the FLEX slot in position_counts.

        ESPN API may return "RB/WR/TE" or "FLEX" depending on league settings version.
        Falls back to "RB/WR/TE" if neither is found (older leagues without a FLEX slot).
        """
        for candidate in ("RB/WR/TE", "FLEX"):
            if candidate in position_counts:
                return candidate
        return "RB/WR/TE"  # fallback (will get KeyError if truly absent — handled by caller)

    def initialize_available_players(self) -> list:
        """Initialize available players with valid data only."""
        available_players = self.predraft_df.sort_values(
            by='projected_points', ascending=False
        ).to_dict('records')

        total_spots = sum(
            sum(pc["required"] for pc in team["position_counts"].values())
            for team in self.teams.values()
        )

        if len(available_players) < total_spots:
            raise ValueError(
                f"Not enough available players ({len(available_players)}) "
                f"to fill all roster spots ({total_spots})"
            )

        return available_players

    # ------------------------------------------------------------------
    # Main simulation loop
    # ------------------------------------------------------------------

    def simulate_draft(self):
        """Run auction draft simulation."""
        current_nominator_idx: int = 0
        round_num: int = 0
        total_reward: float = 0

        while not self.all_rosters_complete():
            round_num += 1
            if round_num > 500:
                # Safety check — log and break. No pdb breakpoint.
                logger.error("Draft exceeded maximum rounds (500). Likely a roster completion bug.")
                for team_name in self.teams.keys():
                    logger.warning(
                        f"{team_name} roster:\n"
                        f"{self.get_team_roster_overview_by_slot(team_name)}"
                    )
                break

            nominating_team_name: str = self.nomination_order[current_nominator_idx]
            if self.teams[nominating_team_name]["roster_completed"]:
                logger.debug(f"{nominating_team_name} roster is complete, skipping nomination")
                current_nominator_idx = (current_nominator_idx + 1) % len(self.nomination_order)
                continue

            try:
                if nominating_team_name == self.rl_team_name and self.rl_model:
                    state = self.get_state()
                    nominated_player = self.rl_model.nominate_player(state, self.available_players)
                    if not nominated_player:
                        current_nominator_idx = (current_nominator_idx + 1) % len(self.nomination_order)
                        continue
                else:
                    nominated_player = self.nominate_player(nominating_team_name)
                    if not nominated_player:
                        current_nominator_idx = (current_nominator_idx + 1) % len(self.nomination_order)
                        continue

                # Start bidding at $1
                current_bid: int = 1
                current_winner: str = nominating_team_name
                active_bidders: set = set(self.teams.keys())

                while len(active_bidders) > 1:
                    highest_bid: int = current_bid
                    highest_bidder = None

                    for team_name in list(active_bidders):
                        if team_name == current_winner:
                            continue

                        if not self.should_bid(team_name, nominated_player, current_bid):
                            active_bidders.remove(team_name)
                            continue

                        if team_name == self.rl_team_name and self.rl_model:
                            state = self.get_state()
                            max_bid = self.rl_model.get_bid(
                                state, nominated_player,
                                current_bid + 1,
                                self.teams[team_name]["current_budget"]
                            )
                        else:
                            randomness = np.random.uniform(-0.1, 0.1)
                            max_bid = round(nominated_player.get('auction_value', 0) * (1 + randomness))

                        if max_bid > current_bid:
                            highest_bid = current_bid + 1
                            highest_bidder = team_name

                    if highest_bidder:
                        current_bid = highest_bid
                        current_winner = highest_bidder
                    else:
                        break

                logger.debug(f"{current_winner} wins {nominated_player['name']} for ${current_bid}")
                self.add_player_to_roster(current_winner, nominated_player, current_bid)

                # Mid-draft reward for RL team (stored for PPO buffer externally)
                if current_winner == self.rl_team_name and self.rl_model:
                    reward = self._calculate_mid_draft_reward(nominated_player, current_bid)
                    total_reward += reward

            except InvalidRosterException as e:
                logger.warning(f"Invalid roster move: {e}")
                break

            current_nominator_idx = (current_nominator_idx + 1) % len(self.nomination_order)

        return self.draft_completed, self.teams, total_reward

    def draft_steps(self):
        """Generator: step-driven auction draft for PufferLib integration.

        Yields (state_dict, player, current_bid) at each RL bidding decision.
        Caller resumes via .send(max_bid) where max_bid is the maximum the RL
        team is willing to pay. After each .send(), read self._step_reward for
        the reward earned by that action (non-zero only when RL team wins a player).

        Raises StopIteration when all rosters are complete.
        """
        self._step_reward = 0.0
        current_nominator_idx: int = 0
        round_num: int = 0

        while not self.all_rosters_complete():
            round_num += 1
            if round_num > 500:
                logger.error("Draft exceeded maximum rounds (500). Likely a roster completion bug.")
                for team_name in self.teams.keys():
                    logger.warning(
                        f"{team_name} roster:\n"
                        f"{self.get_team_roster_overview_by_slot(team_name)}"
                    )
                break

            nominating_team_name: str = self.nomination_order[current_nominator_idx]
            if self.teams[nominating_team_name]["roster_completed"]:
                current_nominator_idx = (current_nominator_idx + 1) % len(self.nomination_order)
                continue

            try:
                # All teams (including RL team) use the heuristic nominator.
                # Nomination strategy is not learned in this port.
                nominated_player = self.nominate_player(nominating_team_name)
                if not nominated_player:
                    current_nominator_idx = (current_nominator_idx + 1) % len(self.nomination_order)
                    continue

                current_bid: int = 1
                current_winner: str = nominating_team_name
                active_bidders: set = set(self.teams.keys())

                while len(active_bidders) > 1:
                    highest_bid: int = current_bid
                    highest_bidder = None

                    for team_name in list(active_bidders):
                        if team_name == current_winner:
                            continue

                        if not self.should_bid(team_name, nominated_player, current_bid):
                            active_bidders.remove(team_name)
                            continue

                        if team_name == self.rl_team_name:
                            state = self.get_state()
                            # Yield RL decision point; reset reward after resuming
                            max_bid = yield (state, nominated_player, current_bid)
                            self._step_reward = 0.0
                        else:
                            randomness = np.random.uniform(-0.1, 0.1)
                            max_bid = round(nominated_player.get('auction_value', 0) * (1 + randomness))

                        if max_bid > current_bid:
                            highest_bid = current_bid + 1
                            highest_bidder = team_name

                    if highest_bidder:
                        current_bid = highest_bid
                        current_winner = highest_bidder
                    else:
                        break

                logger.debug(f"{current_winner} wins {nominated_player['name']} for ${current_bid}")
                self.add_player_to_roster(current_winner, nominated_player, current_bid)

                # Set reward if RL team won this player (read by env after next send())
                if current_winner == self.rl_team_name:
                    self._step_reward = self._calculate_mid_draft_reward(nominated_player, current_bid)

            except InvalidRosterException as e:
                logger.warning(f"Invalid roster move: {e}")
                break

            current_nominator_idx = (current_nominator_idx + 1) % len(self.nomination_order)

    def _calculate_mid_draft_reward(self, player: dict, bid_amount: int) -> float:
        """
        Calculate mid-draft reward for RL team using the reward module convention.
        Returns a small scalar for logging; actual PPO rewards are computed externally.
        """
        fair_value = float(player.get('VORP_dollar', player.get('auction_value', 1)))
        pos = player.get('position', '')
        needed_positions = self.get_positions_needed(self.rl_team_name)
        position_needed = pos in needed_positions

        remaining_budget = self.teams[self.rl_team_name]["current_budget"]
        min_needed = self.get_min_budget_needed_to_complete_roster(self.rl_team_name)
        budget_safe = remaining_budget >= min_needed

        base = (fair_value - bid_amount) / 200.0
        if position_needed:
            base += 0.1
        if not budget_safe:
            base -= 0.2
        return float(base)

    # ------------------------------------------------------------------
    # Roster management
    # ------------------------------------------------------------------

    def get_min_budget_needed_to_complete_roster(self, team_name: str) -> int:
        """Minimum $1/slot needed to finish roster."""
        min_needed = 0
        for position, pc in self.teams[team_name]["position_counts"].items():
            if pc["filled"] < pc["required"]:
                min_needed += pc["required"] - pc["filled"]
        return min_needed

    def add_player_to_roster(self, team_name: str, player: dict, bid_amount: int) -> None:
        """Add player to team's roster."""
        self.teams[team_name]["current_budget"] -= bid_amount

        player_data = {
            "player_id": player["player_id"],
            "name": player["name"],
            "position": player["position"],
            "projected_points": player["projected_points"],
            "bid_amount": bid_amount,
        }

        position = player["position"]
        roster = self.teams[team_name]["roster"]
        position_counts = self.teams[team_name]["position_counts"]
        flex_key = self._get_flex_key(position_counts)

        def find_empty_bench_slot():
            num_bench_slots = self.settings["position_slot_counts"].get("BE", 0)
            for i in range(num_bench_slots):
                bench_key = f"BENCH {i+1}"
                if bench_key in roster and roster[bench_key] is None:
                    return bench_key
            raise InvalidRosterException(
                team_name,
                f"{team_name} has no empty bench slots for {player['name']} ({position}).\n"
                f"{self.get_team_roster_overview_by_slot(team_name)}"
            )

        if position == "QB":
            if roster["QB"] is None:
                roster["QB"] = player_data
                position_counts["QB"]["filled"] += 1
            else:
                bench_slot = find_empty_bench_slot()
                roster[bench_slot] = player_data
                if "BE" in position_counts:
                    position_counts["BE"]["filled"] += 1

        elif position == "K":
            if "K" in roster and roster["K"] is None:
                roster["K"] = player_data
                position_counts["K"]["filled"] += 1
            else:
                bench_slot = find_empty_bench_slot()
                roster[bench_slot] = player_data
                if "BE" in position_counts:
                    position_counts["BE"]["filled"] += 1

        elif position == "D/ST":
            if roster["D/ST"] is None:
                roster["D/ST"] = player_data
                position_counts["D/ST"]["filled"] += 1
            else:
                bench_slot = find_empty_bench_slot()
                roster[bench_slot] = player_data
                if "BE" in position_counts:
                    position_counts["BE"]["filled"] += 1

        elif position == "RB":
            if roster["RB 1"] is None:
                roster["RB 1"] = player_data
                position_counts["RB"]["filled"] += 1
            elif roster["RB 2"] is None:
                roster["RB 2"] = player_data
                position_counts["RB"]["filled"] += 1
            elif (roster["FLEX"] is None
                  and flex_key in position_counts
                  and position_counts[flex_key]["filled"] < position_counts[flex_key]["required"]):
                roster["FLEX"] = player_data
                position_counts[flex_key]["filled"] += 1
            else:
                bench_slot = find_empty_bench_slot()
                roster[bench_slot] = player_data
                if "BE" in position_counts:
                    position_counts["BE"]["filled"] += 1

        elif position == "WR":
            if roster["WR 1"] is None:
                roster["WR 1"] = player_data
                position_counts["WR"]["filled"] += 1
            elif roster["WR 2"] is None:
                roster["WR 2"] = player_data
                position_counts["WR"]["filled"] += 1
            elif (roster["FLEX"] is None
                  and flex_key in position_counts
                  and position_counts[flex_key]["filled"] < position_counts[flex_key]["required"]):
                roster["FLEX"] = player_data
                position_counts[flex_key]["filled"] += 1
            else:
                bench_slot = find_empty_bench_slot()
                roster[bench_slot] = player_data
                if "BE" in position_counts:
                    position_counts["BE"]["filled"] += 1

        elif position == "TE":
            if roster["TE"] is None:
                roster["TE"] = player_data
                position_counts["TE"]["filled"] += 1
            elif (roster["FLEX"] is None
                  and flex_key in position_counts
                  and position_counts[flex_key]["filled"] < position_counts[flex_key]["required"]):
                roster["FLEX"] = player_data
                position_counts[flex_key]["filled"] += 1
            else:
                bench_slot = find_empty_bench_slot()
                roster[bench_slot] = player_data
                if "BE" in position_counts:
                    position_counts["BE"]["filled"] += 1

        # Remove from available pool
        self.available_players.remove(player)

        if team_name == self.rl_team_name:
            self.check_if_roster_is_valid(team_name)

    def should_bid(self, team: str, player: dict, current_bid: int) -> bool:
        """Determine if a team should bid on this player."""
        if self.teams[team]["roster_completed"]:
            return False

        needed_positions = self.get_positions_needed(team)
        if not needed_positions:
            self.teams[team]["roster_completed"] = True
            return False

        def_count = sum(
            1 for p in self.teams[team]["roster"].values()
            if p is not None and p["position"] == "D/ST"
        )
        k_count = sum(
            1 for p in self.teams[team]["roster"].values()
            if p is not None and p["position"] == "K"
        )

        if (player["position"] == "D/ST" and def_count >= 2) or \
           (player["position"] == "K" and k_count >= 2):
            return False

        if player["position"] not in needed_positions:
            return False

        max_bid = min(
            self.teams[team]["current_budget"],
            player.get("auction_value", 1)
        )
        can_afford = (
            self.teams[team]["current_budget"] - current_bid
            >= self.get_min_budget_needed_to_complete_roster(team)
        )

        return current_bid <= max_bid and can_afford

    def nominate_player(self, team_name: str):
        """Select a player to nominate for auction."""
        if self.teams[team_name]["roster_completed"]:
            return None

        positions_needed = self.get_positions_needed(team_name)
        if not positions_needed:
            self.teams[team_name]["roster_completed"] = True
            return None

        candidates = []
        for position in positions_needed:
            position_players = [p for p in self.available_players if p['position'] == position]
            position_players.sort(key=lambda x: x['projected_points'], reverse=True)
            candidates.extend(position_players[:3])

        if not candidates:
            raise InvalidRosterException(
                team_name,
                f"{team_name} has no candidates for positions {positions_needed}"
            )

        return np.random.choice(candidates)

    def all_rosters_complete(self) -> bool:
        """Check if all teams have completed their rosters."""
        for team_name in self.teams.keys():
            if not self.teams[team_name]["roster_completed"]:
                return False
        logger.info("---- All rosters are complete! ----")
        self.draft_completed = True
        return True

    # ------------------------------------------------------------------
    # State / info helpers
    # ------------------------------------------------------------------

    def get_state(self) -> dict:
        """Get current state dict for RL model / state builder."""
        num_roster_slots = 14

        rl_team = self.teams[self.rl_team_name]
        position_needs: dict = {
            pos: pc["required"] - pc["filled"]
            for pos, pc in rl_team["position_counts"].items()
        }

        position_counts = {}
        for slot, player in rl_team["roster"].items():
            if player is not None:
                pos = player["position"]
                position_counts[pos] = position_counts.get(pos, 0) + 1

        position_scarcity = {}
        for pos in ["QB", "RB", "WR", "TE", "D/ST", "K"]:
            available = len([p for p in self.available_players if p["position"] == pos])
            total_needed = sum(
                pc["required"] - pc["filled"]
                for team in self.teams.values()
                for _pos, pc in team["position_counts"].items()
                if _pos == pos
            )
            position_scarcity[pos] = max(0, total_needed - available)

        position_values = {}
        for pos in ["QB", "RB", "WR", "TE", "D/ST", "K"]:
            pos_players = [p for p in self.available_players if p["position"] == pos]
            if pos_players:
                position_values[pos] = {
                    "avg_value": float(np.mean([p.get("auction_value", 0) for p in pos_players])),
                    "avg_points": float(np.mean([p.get("projected_points", 0) for p in pos_players])),
                }

        total_needs = sum(position_needs.values())

        return {
            'rl_team_budget': rl_team["current_budget"],
            'opponent_budgets': [self.teams[t]["current_budget"] for t in self.teams if t != self.rl_team_name],
            'draft_turn': len(self.players_drafted),
            'teams': list(self.teams.keys()),
            'predicted_points_per_slot': {
                team: self.get_points_per_slot(team, num_roster_slots)
                for team in self.teams
            },
            'position_needs': position_needs,
            'position_counts': position_counts,
            'position_scarcity': position_scarcity,
            'position_values': position_values,
            'remaining_budget_per_need': rl_team["current_budget"] / max(1, total_needs),
            'draft_progress': len(self.players_drafted) / (len(self.teams) * num_roster_slots),
            'total_team_points': sum(
                p["projected_points"] for p in rl_team["roster"].values() if p
            ),
        }

    def get_points_per_slot(self, team_name: str, num_slots: int) -> list:
        """Get fixed-size array of projected points per roster slot."""
        points = [0.0] * num_slots
        for slot_idx, (slot_name, player) in enumerate(self.teams[team_name]["roster"].items()):
            if slot_idx >= num_slots:
                break
            if player is not None:
                points[slot_idx] = player["projected_points"]
        return points

    def get_draft_results(self) -> dict:
        """Get results of the draft."""
        results = {
            "teams": {},
            "players_drafted": self.players_drafted.copy(),
            "available_players": self.available_players.copy(),
        }
        for team_name, team in self.teams.items():
            results["teams"][team_name] = {
                "roster": team["roster"].copy(),
                "remaining_budget": team["current_budget"],
                "position_counts": {
                    pos: {"required": pc["required"], "filled": pc["filled"]}
                    for pos, pc in team["position_counts"].items()
                }
            }
        return results

    def check_if_roster_is_valid(self, team_name: str) -> bool:
        """Raise InvalidRosterException if position counts are over the required limit."""
        team = self.teams[team_name]
        for position, pc in team["position_counts"].items():
            if pc["filled"] > pc["required"]:
                msg = (
                    f"Too many {position} players: "
                    f"has {pc['filled']}, required {pc['required']}.\n"
                )
                msg += self.get_team_roster_overview_by_slot(team_name)
                raise InvalidRosterException(team_name, msg)
        return True

    def get_team_roster_overview_by_slot(self, team_name: str) -> str:
        """Human-readable roster overview."""
        slots = []
        for position, player in self.teams[team_name]["roster"].items():
            if player is None:
                slots.append(f"\t- {position}:")
            elif 'BENCH' in position:
                slots.append(f"\t- {position}: {player['name']} ({player['position']}) (${player['bid_amount']})")
            else:
                slots.append(f"\t- {position}: {player['name']} (${player['bid_amount']})")
        return "Roster:\n" + "\n".join(slots)

    def get_positions_needed(self, team_name: str) -> list:
        """Get list of positions that still need to be filled for a team."""
        positions_needed = set()
        roster = self.teams[team_name]["roster"]

        if roster["QB"] is None:
            positions_needed.add("QB")

        rb_empty = sum(1 for slot in ["RB 1", "RB 2"] if roster[slot] is None)
        if rb_empty > 0:
            positions_needed.add("RB")

        wr_empty = sum(1 for slot in ["WR 1", "WR 2"] if roster[slot] is None)
        if wr_empty > 0:
            positions_needed.add("WR")

        if roster["TE"] is None:
            positions_needed.add("TE")

        if roster["D/ST"] is None:
            positions_needed.add("D/ST")

        if "K" in roster and roster["K"] is None:
            positions_needed.add("K")

        # FLEX slot — add positions we're still short on
        if roster["FLEX"] is None:
            rb_count = sum(1 for p in roster.values() if p and p["position"] == "RB")
            wr_count = sum(1 for p in roster.values() if p and p["position"] == "WR")
            te_count = sum(1 for p in roster.values() if p and p["position"] == "TE")

            if rb_count < 3:
                positions_needed.add("RB")
            if wr_count < 3:
                positions_needed.add("WR")
            if te_count < 2:
                positions_needed.add("TE")

        # Bench slots — accept any skill position
        bench_slots = [slot for slot in roster.keys() if "BENCH" in slot]
        if any(roster[slot] is None for slot in bench_slots):
            positions_needed.update(["QB", "RB", "WR", "TE"])

        return list(positions_needed)
