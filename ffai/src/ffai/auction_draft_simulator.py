import pandas as pd
import numpy as np
from pathlib import Path
import json
from ffai import get_logger
from ffai.data.espn_scraper import ESPNDraftScraper
import torch
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

        # Set data directory relative to this file
        self.data_dir = Path(__file__).parent / "data/raw"

        # Load data
        scraper = ESPNDraftScraper()
        self.draft_df, self.stats_df, self.weekly_df, self.predraft_df, self.settings = scraper.load_or_fetch_data(self.year)

        # Initialize teams based on league settings
        self.teams: dict[str, dict] = self.initialize_teams()
        self.nomination_order: list[str] = list(self.teams.keys())
        self.available_players: list[dict] = self.initialize_available_players()
        self.players_drafted: list[dict] = []

        # self.num_rl_players = 0
        # self.last_num_rl_players_debugged = 0

    def initialize_teams(self):
        """Initialize teams with mandatory slot budget reserves"""
        teams: dict[str, dict] = {}
        for i in range(12):
            team_name = f"Team {i+1}"

            # Create enumerated roster slots
            roster_slots = {
                "QB": None,
                "RB 1": None,
                "RB 2": None,
                "WR 1": None,
                "WR 2": None,
                "TE": None,
                "FLEX": None,  # RB/WR/TE
                "D/ST": None,  # Changed from "DEF" to "D/ST"
            }

            # Only some years have K slots
            if self.settings["position_slot_counts"].get("K"):
                roster_slots["K"] = None

            # Add enumerated bench slots based on league settings
            num_bench_slots = self.settings["position_slot_counts"].get("BE")
            for i in range(num_bench_slots):
                roster_slots[f"BENCH {i+1}"] = None

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

    def initialize_available_players(self) -> list[dict]:
        """Initialize available players with valid data only"""
        # Convert DataFrame to list of dicts and sort by projected points
        available_players = self.predraft_df.sort_values(
            by='projected_points',
            ascending=False
        ).to_dict('records')

        # Calculate total roster spots needed
        total_spots = sum(
            sum(position_count["required"] for position_count in team["position_counts"].values())
            for team in self.teams.values()
        )

        if len(available_players) < total_spots:
            raise ValueError(f"Not enough available players ({len(available_players)}) to fill all roster spots ({total_spots})")

        return available_players

    def simulate_draft(self):
        """Run auction draft simulation following design doc rules"""
        current_nominator_idx: int = 0
        round_num: int = 0
        total_reward: float = 0

        while not self.all_rosters_complete():
            round_num += 1
            if round_num > 500:  # Safety check
                logger.error("Draft exceeded maximum rounds")
                for team_name in self.teams.keys():
                    logger.warning(f"{team_name} roster: {self.get_team_roster_overview_by_slot(team_name)}")
                import pdb; pdb.set_trace()
                break

            # Get nominating team and their nomination
            nominating_team_name: str = self.nomination_order[current_nominator_idx]
            if self.teams[nominating_team_name]["roster_completed"]:
                logger.debug(f"{nominating_team_name} roster is complete, skipping nomination")
                current_nominator_idx = (current_nominator_idx + 1) % len(self.nomination_order)
                continue

            try:

                # If RL team is nominating, get nomination and update model
                if nominating_team_name == self.rl_team_name and self.rl_model:
                    state = self.get_state()
                    nominated_player = self.rl_model.nominate_player(state, self.available_players)
                    if not nominated_player:
                        # Penalize failing to nominate
                        reward = -5
                        total_reward += reward
                        self.rl_model.update(reward)
                        current_nominator_idx = (current_nominator_idx + 1) % len(self.nomination_order)
                        continue
                else:
                    # Other teams nominate normally
                    nominated_player = self.nominate_player(nominating_team_name)
                    if not nominated_player:
                        current_nominator_idx = (current_nominator_idx + 1) % len(self.nomination_order)
                        continue

                # Start bidding at $1
                current_bid: int = 1
                current_winner: str = nominating_team_name
                active_bidders: set[str] = set(self.teams.keys())

                # Teams bid until all pass
                while len(active_bidders) > 1:
                    highest_bid: int = current_bid
                    highest_bidder: str | None = None

                    # Each team decides if they want to bid
                    for team_name in list(active_bidders):
                        if team_name == current_winner:
                            continue

                        # Check if team should bid based on roster needs and budget
                        if not self.should_bid(team_name, nominated_player, current_bid):
                            active_bidders.remove(team_name)
                            continue

                        # Get team's max bid for this player
                        if team_name == self.rl_team_name and self.rl_model:
                            state = self.get_state()
                            max_bid = self.rl_model.get_bid(state, nominated_player, current_bid + 1, self.teams[team_name]["current_budget"])
                        else:
                            # Auto-draft teams bid predicted value ±10%
                            randomness = np.random.uniform(-0.1, 0.1)
                            max_bid = round(nominated_player.get('auction_value', 0) * (1 + randomness))

                        # Place bid if it's higher than current
                        if max_bid > current_bid:
                            highest_bid = current_bid + 1
                            highest_bidder = team_name

                    if highest_bidder:
                        current_bid = highest_bid
                        current_winner = highest_bidder
                    else:
                        break

                # Add player to winning team's roster
                logger.debug(f"{current_winner} wins {nominated_player['name']} for ${current_bid}")
                self.add_player_to_roster(current_winner, nominated_player, current_bid)

                # If RL team won the player, calculate reward
                if current_winner == self.rl_team_name and self.rl_model:
                    reward = 0
                    pos = nominated_player['position']
                    needed_positions = self.get_positions_needed(self.rl_team_name)
                    position_counts = self.teams[self.rl_team_name]["position_counts"]

                    # Big reward for first player at each required position
                    if pos in needed_positions:
                        if position_counts[pos]["filled"] == 0:
                            reward += 30  # Huge reward for first player at position
                        elif position_counts[pos]["filled"] < position_counts[pos]["required"]:
                            reward += 20  # Big reward for filling required slots
                        else:
                            reward += 5   # Small reward for depth

                    # Harsh penalties for bad roster construction
                    if needed_positions:  # If we still have needs
                        # Penalty for taking non-needed positions
                        if pos not in needed_positions:
                            reward -= 25

                        # Extra penalty for position hoarding
                        if position_counts[pos]["filled"] > position_counts[pos]["required"]:
                            reward -= position_counts[pos]["filled"] * 15  # Exponential penalty

                        # Extreme penalty for hoarding D/ST
                        if pos == "D/ST" and position_counts[pos]["filled"] >= 2:
                            reward -= 50

                    # Budget management penalties
                    remaining_budget = self.teams[self.rl_team_name]["current_budget"] - current_bid
                    min_needed = self.get_min_budget_needed_to_complete_roster(self.rl_team_name)
                    if remaining_budget < min_needed:
                        reward -= 40  # Big penalty for not leaving enough budget

                    total_reward += reward
                    self.rl_model.update(reward)

            except InvalidRosterException as e:
                # Immediate negative feedback for invalid roster moves
                if self.rl_model:
                    reward = -5
                    logger.debug(f"Invalid roster move: {e}, -5 reward")
                    total_reward += reward
                    self.rl_model.update(reward)
                logger.warning(f"Invalid roster move: {e}")
                break

            # Move to next nominator
            current_nominator_idx = (current_nominator_idx + 1) % len(self.nomination_order)

        # Return draft results and total reward
        return self.draft_completed, self.teams, total_reward

    def get_min_budget_needed_to_complete_roster(self, team_name: str) -> int:
        """Calculate minimum budget needed to complete roster"""
        min_needed = 0
        for position, position_count in self.teams[team_name]["position_counts"].items():
            if position_count["filled"] < position_count["required"]:
                unfilled = position_count["required"] - position_count["filled"]
                min_needed += unfilled
        return min_needed

    def add_player_to_roster(self, team_name: str, player: dict, bid_amount: int) -> None:
        """Add player to team's roster"""
        # update budget
        self.teams[team_name]["current_budget"] -= bid_amount

        player_data = {
            "player_id": player["player_id"],
            "name": player["name"],
            "position": player["position"],
            "projected_points": player["projected_points"],
            "bid_amount": bid_amount
        }

        # Find appropriate slot for player
        position = player["position"]
        roster = self.teams[team_name]["roster"]
        position_counts = self.teams[team_name]["position_counts"]

        # Add debug logging for RL team acquisitions
        # if team_name == self.rl_team_name:
        #     logger.debug(f"\nRL Team adding {player['name']} ({player['position']}) for ${bid_amount}")
        #     logger.debug("Current roster before add:")
        #     logger.debug(self.get_team_roster_overview_by_slot(team_name))

        #     # Log reward calculation details
        #     needed_positions = self.get_positions_needed(team_name)
        #     logger.debug(f"Needed positions before add: {needed_positions}")

        def find_empty_bench_slot():
            """Helper function to find first empty bench slot"""
            # Get number of bench slots from settings
            num_bench_slots = self.settings["position_slot_counts"].get("BE")
            for i in range(num_bench_slots):
                bench_key = f"BENCH {i+1}"
                if bench_key in roster and roster[bench_key] is None:
                    return bench_key
            raise InvalidRosterException(team_name, f"{team_name} has no empty bench slots for {player['name']} ({position}).\n{self.get_team_roster_overview_by_slot(team_name)}")

        # Try to fill primary position first
        if position == "QB":
            if roster["QB"] is None:
                roster["QB"] = player_data
                position_counts["QB"]["filled"] += 1
            else:
                bench_slot = find_empty_bench_slot()
                if bench_slot:
                    roster[bench_slot] = player_data
                    position_counts["BE"]["filled"] += 1

        elif position == "K":
            if roster["K"] is None:
                roster["K"] = player_data
                position_counts["K"]["filled"] += 1
            else:
                bench_slot = find_empty_bench_slot()
                if bench_slot:
                    roster[bench_slot] = player_data
                    position_counts["BE"]["filled"] += 1

        elif position == "D/ST":
            if roster["D/ST"] is None:
                roster["D/ST"] = player_data
                position_counts["D/ST"]["filled"] += 1
            else:
                bench_slot = find_empty_bench_slot()
                if bench_slot:
                    roster[bench_slot] = player_data
                    position_counts["BE"]["filled"] += 1

        elif position == "RB":
            if roster["RB 1"] is None:
                roster["RB 1"] = player_data
                position_counts["RB"]["filled"] += 1
            elif roster["RB 2"] is None:
                roster["RB 2"] = player_data
                position_counts["RB"]["filled"] += 1
            elif roster["FLEX"] is None and position_counts["RB/WR/TE"]["filled"] < position_counts["RB/WR/TE"]["required"]:
                roster["FLEX"] = player_data
                position_counts["RB/WR/TE"]["filled"] += 1
            else:
                bench_slot = find_empty_bench_slot()
                if bench_slot:
                    roster[bench_slot] = player_data
                    position_counts["BE"]["filled"] += 1

        elif position == "WR":
            if roster["WR 1"] is None:
                roster["WR 1"] = player_data
                position_counts["WR"]["filled"] += 1
            elif roster["WR 2"] is None:
                roster["WR 2"] = player_data
                position_counts["WR"]["filled"] += 1
            elif roster["FLEX"] is None and position_counts["RB/WR/TE"]["filled"] < position_counts["RB/WR/TE"]["required"]:
                roster["FLEX"] = player_data
                position_counts["RB/WR/TE"]["filled"] += 1
            else:
                bench_slot = find_empty_bench_slot()
                if bench_slot:
                    roster[bench_slot] = player_data
                    position_counts["BE"]["filled"] += 1

        elif position == "TE":
            if roster["TE"] is None:
                roster["TE"] = player_data
                position_counts["TE"]["filled"] += 1
            elif roster["FLEX"] is None and position_counts["RB/WR/TE"]["filled"] < position_counts["RB/WR/TE"]["required"]:
                roster["FLEX"] = player_data
                position_counts["RB/WR/TE"]["filled"] += 1
            else:
                bench_slot = find_empty_bench_slot()
                if bench_slot:
                    roster[bench_slot] = player_data
                    position_counts["BE"]["filled"] += 1

        # Remove player from available players
        self.available_players.remove(player)

        if team_name == self.rl_team_name:
            self.check_if_roster_is_valid(team_name)

            # Add more debug logging after successful add
            # logger.debug("\nRoster after add:")
            # logger.debug(self.get_team_roster_overview_by_slot(team_name))

            # self.num_rl_players += 1

    def should_bid(self, team, player, current_bid):
        """Determine if team should bid on player"""
        # First check if roster is already marked as complete
        if self.teams[team]["roster_completed"]:
            return False

        # Check if we need any more positions
        needed_positions = self.get_positions_needed(team)
        if not needed_positions:
            # Mark roster as complete if we don't need any positions
            self.teams[team]["roster_completed"] = True
            return False

        # Count how many defenses and kickers this team already has
        def_count = sum(
            1 for slot_name, player in self.teams[team]["roster"].items()
            if player is not None and player["position"] == "D/ST"
        )
        k_count = sum(
            1 for slot_name, player in self.teams[team]["roster"].items()
            if player is not None and player["position"] == "K"
        )

        # Don't bid on defense/kicker if team already has 2
        if (player["position"] == "D/ST" and def_count >= 2) or (player["position"] == "K" and k_count >= 2):
            return False

        # If team has unfilled required slots, only bid on players for those positions
        if needed_positions and player["position"] not in needed_positions:
            return False

        # ESPN auto-draft logic for budget
        max_bid = min(
            self.teams[team]["current_budget"],
            player.get("auction_value", 1)
        )
        can_afford = self.teams[team]["current_budget"] - current_bid >= self.get_min_budget_needed_to_complete_roster(team)

        return current_bid <= max_bid and can_afford

    def nominate_player(self, team_name: str) -> dict | None:
        """Select a player to nominate for auction"""
        if self.teams[team_name]["roster_completed"]:
            # logger.debug(f"{team_name} roster is complete, skipping nomination")
            return None

        positions_needed = self.get_positions_needed(team_name)

        if not positions_needed:
            # logger.debug(f"{team_name} roster is complete, skipping nomination")
            self.teams[team_name]["roster_completed"] = True
            return None

        # Get candidates for positions needed
        candidates = []
        for position in positions_needed:
            # Filter players by position and sort by projected points
            position_players = [p for p in self.available_players if p['position'] == position]
            position_players.sort(key=lambda x: x['projected_points'], reverse=True)
            # Take top 2 players from each position
            candidates.extend(position_players[:3])

        if not candidates:
            raise InvalidRosterException(team_name, f"{team_name} has no candidates for positions {positions_needed}")

        # Use RL model for RL team, random for others
        if team_name == self.rl_team_name and self.rl_model:
            state = self.get_state()
            player = self.rl_model.nominate_player(state, candidates)
        else:
            player = np.random.choice(candidates)

        return player

    def all_rosters_complete(self) -> bool:
        """Check if all teams have completed their rosters"""
        for team_name in self.teams.keys():
            if not self.teams[team_name]["roster_completed"]:
                return False

        logger.info("---- All rosters are complete! ----")
        self.draft_completed = True
        return True

    def get_rl_bid(self, player: dict) -> int:
        """Get bid amount from RL model"""
        if not self.rl_model:
            return 0

        # Get current state
        state = self.get_state()

        # Get min/max valid bids
        min_needed = self.get_min_budget_needed_to_complete_roster(self.rl_team_name)
        remaining_budget = self.teams[self.rl_team_name]["current_budget"]
        max_bid = remaining_budget - min_needed
        min_bid = 1

        # Get bid from model
        return self.rl_model.get_bid(state, player, min_bid, max_bid)

    def get_points_per_slot(self, team_name: str, num_slots: int) -> list[float]:
        """Get fixed-size array of points per roster slot"""
        # Initialize array with zeros
        points = [0.0] * num_slots

        # Fill in points for each roster slot
        slot_idx = 0
        roster = self.teams[team_name]["roster"]

        # Add points for each filled roster slot
        for slot_name, player in roster.items():
            if player is not None:  # Skip empty slots
                points[slot_idx] = player["projected_points"]
            slot_idx += 1
            if slot_idx >= num_slots:
                break

        return points

    def get_draft_results(self) -> dict:
        """Get results of the draft"""
        results = {
            "teams": {},
            "players_drafted": self.players_drafted.copy(),
            "available_players": self.available_players.copy()
        }

        for team_name, team in self.teams.items():
            results["teams"][team_name] = {
                "roster": team["roster"].copy(),
                "remaining_budget": team["current_budget"],
                "position_counts": {
                    pos: {
                        "required": position_count["required"],
                        "filled": position_count["filled"]
                    }
                    for pos, position_count in team["position_counts"].items()
                }
            }

        return results

    def get_state(self) -> dict:
        """Get current state for RL model"""
        # Calculate number of roster slots per team
        num_roster_slots = 14  # Fixed size based on standard roster slots

        # Get position needs for RL team
        rl_team = self.teams[self.rl_team_name]
        position_needs: dict[str, int] = {
            pos: position_count["required"] - position_count["filled"]
            for pos, position_count in rl_team["position_counts"].items()
        }

        # Count positions by type for RL team
        position_counts = {}
        for slot, player in rl_team["roster"].items():
            if player is not None:
                pos = player["position"]
                position_counts[pos] = position_counts.get(pos, 0) + 1

        # Calculate market scarcity for each position
        position_scarcity = {}
        for pos in ["QB", "RB", "WR", "TE", "D/ST", "K"]:
            available = len([p for p in self.available_players if p["position"] == pos])
            total_needed = sum(
                position_count["required"] - position_count["filled"]
                for team in self.teams.values()
                for _pos, position_count in team["position_counts"].items()
                if _pos == pos
            )
            position_scarcity[pos] = max(0, total_needed - available)

        # Get average values by position
        position_values = {}
        for pos in ["QB", "RB", "WR", "TE", "D/ST", "K"]:
            pos_players = [p for p in self.available_players if p["position"] == pos]
            if pos_players:
                position_values[pos] = {
                    "avg_value": np.mean([p.get("auction_value", 0) for p in pos_players]),
                    "avg_points": np.mean([p.get("projected_points", 0) for p in pos_players])
                }

        state = {
            # Existing state
            'rl_team_budget': self.teams[self.rl_team_name]["current_budget"],
            'opponent_budgets': [self.teams[t]["current_budget"] for t in self.teams if t != self.rl_team_name],
            'draft_turn': len(self.players_drafted),
            'teams': list(self.teams.keys()),
            'predicted_points_per_slot': {
                team: self.get_points_per_slot(team, num_roster_slots)
                for team in self.teams
            },
            'position_needs': position_needs,

            # New state information
            'position_counts': position_counts,  # Current count of each position on roster
            'position_scarcity': position_scarcity,  # How many more players needed vs available
            'position_values': position_values,  # Average values/points by position
            'remaining_budget_per_need': rl_team["current_budget"] / max(1, sum(position_needs.values())),  # Budget per remaining slot
            'draft_progress': len(self.players_drafted) / (len(self.teams) * num_roster_slots),  # Progress through draft
            'total_team_points': sum(player["projected_points"] for player in rl_team["roster"].values() if player),  # Total projected points
        }

        return state

    def check_if_roster_is_valid(self, team_name: str) -> bool:
        """
        Check if a team's roster meets all position requirements.

        Args:
            team_name: Name of the team to validate

        Returns:
            bool: True if roster meets all requirements, False otherwise

        Raises:
            InvalidRosterException: If the roster is invalid
        """
        team = self.teams[team_name]

        # Check each position count requirement
        for position, position_count in team["position_counts"].items():
            if position_count["filled"] > position_count["required"]:
                msg = f"needs {position_count['required'] - position_count['filled']} more {position}.\n"
                # msg += f"\nroster needs: {self.get_positions_needed(team_name)}"
                # msg += f"\nroster has: {', '.join([p['name'] + ' (' + p['position'] + ')' + ' ($' + str(p['bid_amount']) + ')' for p in team['roster']])}"
                msg += self.get_team_roster_overview_by_slot(team_name)
                raise InvalidRosterException(team_name, msg)

        return True

    def get_team_roster_overview_by_slot(self, team_name: str) -> str:
        """Get overview of a team's roster by slot"""
        slots = []

        for position, player in self.teams[team_name]["roster"].items():
            if player is None:
                slots.append(f"\t- {position}:")
            elif 'BENCH' in position:
                slots.append(f"\t- {position}: {player['name']} ({player['position']}) (${player['bid_amount']})")
            else:
                slots.append(f"\t- {position}: {player['name']} (${player['bid_amount']})")

        return "Roster:\n" + "\n".join(slots)

    def get_positions_needed(self, team_name: str) -> list[str]:
        """Get list of positions that still need to be filled for a team"""
        positions_needed = set()
        roster = self.teams[team_name]["roster"]

        # Check primary positions
        if roster["QB"] is None:
            positions_needed.add("QB")

        # Check RB slots
        rb_empty = sum(1 for slot in ["RB 1", "RB 2"] if roster[slot] is None)
        if rb_empty > 0:
            positions_needed.add("RB")

        # Check WR slots
        wr_empty = sum(1 for slot in ["WR 1", "WR 2"] if roster[slot] is None)
        if wr_empty > 0:
            positions_needed.add("WR")

        # Check TE slot
        if roster["TE"] is None:
            positions_needed.add("TE")

        # Check D/ST slot
        if roster["D/ST"] is None:
            positions_needed.add("D/ST")

        # Check K slot if it exists
        if "K" in roster and roster["K"] is None:
            positions_needed.add("K")

        # Check FLEX slot - only add positions we're short on
        if roster["FLEX"] is None:
            rb_count = sum(1 for slot, player in roster.items() if player and player["position"] == "RB")
            wr_count = sum(1 for slot, player in roster.items() if player and player["position"] == "WR")
            te_count = sum(1 for slot, player in roster.items() if player and player["position"] == "TE")

            # Add FLEX-eligible positions we're short on
            if rb_count < 3:  # 2 RB slots + 1 FLEX
                positions_needed.add("RB")
            if wr_count < 3:  # 2 WR slots + 1 FLEX
                positions_needed.add("WR")
            if te_count < 2:  # 1 TE slot + 1 FLEX
                positions_needed.add("TE")

        # Add bench needs if we have empty bench slots
        bench_slots = [slot for slot in roster.keys() if "BENCH" in slot]
        if any(roster[slot] is None for slot in bench_slots):
            positions_needed.update(["QB", "RB", "WR", "TE"])

        positions_needed = list(positions_needed)

        # if team_name == self.rl_team_name and self.num_rl_players != self.last_num_rl_players_debugged:
        #     logger.info(self.get_team_roster_overview_by_slot(team_name))
        #     logger.info(f"Needed positions: {positions_needed}")
        #     import pdb; pdb.set_trace()

            # self.last_num_rl_players_debugged = self.num_rl_players

        return positions_needed
