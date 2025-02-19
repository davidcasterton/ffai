import pandas as pd
import numpy as np
from pathlib import Path
import json
from ffai import get_logger
from ffai.data.espn_scraper import ESPNDraftScraper
import torch
import copy
import logging

logger = get_logger(__name__, level=logging.DEBUG)

class InvalidRosterException(Exception):
    """Exception raised when a team's roster is invalid."""
    def __init__(self, team_name: str, message: str):
        self.team_name = team_name
        self.message = message
        super().__init__(f"Invalid roster for {team_name}: {message}")

class AuctionDraftSimulator:
    def __init__(self, year, budget=200, rl_team="Team 1", rl_model=None):
        self.year = year
        self.budget = budget
        self.rl_team_name = rl_team
        self.rl_model = rl_model

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

    def initialize_teams(self):
        """Initialize teams with mandatory slot budget reserves"""
        teams: dict[str, dict] = {}
        for i in range(12):
            team_name = f"Team {i+1}"

            teams[team_name] = {
                "current_budget": copy.deepcopy(self.budget),
                "roster": [],
                "roster_completed": False,
                "slots": {
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
            sum(slot["required"] for slot in team["slots"].values())
            for team in self.teams.values()
        )

        if len(available_players) < total_spots:
            raise ValueError(f"Not enough available players ({len(available_players)}) to fill all roster spots ({total_spots})")

        return available_players

    def simulate_draft(self):
        """Run auction draft simulation following design doc rules"""
        current_nominator_idx: int = 0
        round_num: int = 0

        while not self.all_rosters_complete():
            round_num += 1
            if round_num > 500:  # Safety check
                logger.error("Draft exceeded maximum rounds")
                break

            # Get nominating team and their nomination
            nominating_team_name: str = self.nomination_order[current_nominator_idx]
            nominated_player: dict | None = self.nominate_player(nominating_team_name)

            if not nominated_player:
                if self.teams[nominating_team_name]["roster_completed"]:
                    current_nominator_idx = (current_nominator_idx + 1) % len(self.nomination_order)
                    continue
                else:
                    logger.warning(f"No valid nomination from {nominating_team_name}")
                    import pdb; pdb.set_trace()

            # Start bidding at $1
            current_bid: int = 1
            current_winner: str = nominating_team_name
            active_bidders: set[str] = set(self.teams.keys())

            # Teams bid until 11 pass
            while len(active_bidders) > 1:
                highest_bid: int = current_bid
                highest_bidder: str | None = None

                # Each team decides if they want to bid
                for team_name in list(active_bidders):
                    if team_name == current_winner:
                        continue

                    # Get team's max bid for this player
                    if team_name == self.rl_team_name:
                        max_bid = self.get_rl_bid(nominated_player)
                    else:
                        # Auto-draft teams bid predicted value ±10%
                        randomness = np.random.uniform(-0.1, 0.1)
                        max_bid = round(nominated_player['auction_value'] * (1 + randomness))

                    # Validate team can afford bid and complete roster
                    remaining_budget: int = self.teams[team_name]["current_budget"]
                    min_needed: int = self.get_min_budget_needed_to_complete_roster(team_name)
                    if max_bid >= current_bid + 1 and remaining_budget - (current_bid + 1) >= min_needed:
                        if current_bid + 1 > highest_bid:
                            highest_bid = current_bid + 1
                            highest_bidder = team_name
                    else:
                        active_bidders.remove(team_name)

                if highest_bidder:
                    current_bid = highest_bid
                    current_winner = highest_bidder
                else:
                    break

            # Add player to winning team's roster
            logger.debug(f"{current_winner} wins {nominated_player['name']} for ${current_bid} ({nominated_player['projected_points']} points)")
            self.add_player_to_roster(current_winner, nominated_player, current_bid)
            # Move to next nominator
            current_nominator_idx = (current_nominator_idx + 1) % len(self.nomination_order)

        logger.info("Draft complete")
        for team_name, team in self.teams.items():
            logger.debug(f"{team_name} roster: {team['roster']}")

        return self.teams

    def get_min_budget_needed_to_complete_roster(self, team_name: str) -> int:
        """Calculate minimum budget needed to complete roster"""
        min_needed = 0
        for position, slot in self.teams[team_name]["slots"].items():
            if slot["filled"] < slot["required"]:
                unfilled = slot["required"] - slot["filled"]
                min_needed += unfilled
        return min_needed

    def add_player_to_roster(self, team, player, bid_amount):
        """Add player to roster with budget management"""
        # update budget
        self.teams[team]["current_budget"] -= bid_amount

        # Add to roster
        self.teams[team]["roster"].append({
            "player_id": player["player_id"],
            "name": player["name"],
            "position": player["position"],
            "projected_points": player["projected_points"],
            "bid_amount": bid_amount
        })

        # Remove player from available players
        self.available_players.remove(player)

        # Update roster slots
        position = player["position"]
        slots = self.teams[team]["slots"]

        # Try to fill primary position first
        if position in slots and slots[position]["filled"] < slots[position]["required"]:
            slots[position]["filled"] += 1
        # Try FLEX for eligible positions
        elif position in ["RB", "WR", "TE"] and slots["RB/WR/TE"]["filled"] < slots["RB/WR/TE"]["required"]:
            slots["RB/WR/TE"]["filled"] += 1
        # Otherwise use bench
        else:
            slots["BE"]["filled"] += 1

        if team == self.rl_team_name:
            self.check_if_roster_is_valid(team)

    def nominate_player(self, team_name: str) -> dict | None:
        """Select a player to nominate for auction"""
        if self.teams[team_name]["roster_completed"]:
            logger.debug(f"{team_name} roster is complete, skipping nomination")
            return None

        # Get positional needs
        needs = []
        for position, slot in self.teams[team_name]["slots"].items():
            if position == "IR":
                continue
            if slot["filled"] < slot["required"]:
                if position == "RB/WR/TE":
                    needs.extend(["RB", "WR", "TE"])
                elif position == "BE":
                    needs = ["QB", "RB", "WR", "TE"]  # Include all positions for bench
                else:
                    needs.append(position)
        needs = list(set(needs))

        if not needs:
            logger.debug(f"{team_name} roster is complete, skipping nomination")
            self.teams[team_name]["roster_completed"] = True
            return None

        # Get candidates for positional needs
        candidates = []
        for position in needs:
            # Filter players by position and sort by projected points
            position_players = [p for p in self.available_players if p['position'] == position]
            position_players.sort(key=lambda x: x['projected_points'], reverse=True)
            # Take top 2 players from each position
            candidates.extend(position_players[:3])

        if not candidates:
            logger.debug(f"{team_name} has no candidates for needs {needs}, trying all available players")
            import pdb; pdb.set_trace()
            candidates = self.available_players

        # Use RL model for RL team, random for others
        if team_name == self.rl_team_name and self.rl_model:
            state = self.get_state()
            player = self.rl_model.nominate_player(state, candidates)
        else:
            player = np.random.choice(candidates)

        # logger.debug(f"{team_name} nominates {player['name']} ({player['position']}) - {player['projected_points']} points")
        return player

    def all_rosters_complete(self) -> bool:
        """Check if all teams have completed their rosters"""
        for team_name, team in self.teams.items():
            # Check each position slot
            for position, slot in team["slots"].items():
                if slot["filled"] < slot["required"]:
                    # logger.debug(f"{team_name} needs {slot['required'] - slot['filled']} more {position}")
                    return False
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
        for position, slot in self.teams[team_name]["slots"].items():
            if position in ["BE", "IR"]:
                continue

            # Get players in this position
            position_players = [
                p for p in self.teams[team_name]["roster"]
                if p["position"] == position
            ]

            # Sort by projected points
            position_players.sort(key=lambda x: x["projected_points"], reverse=True)

            # Fill slots for this position
            for i in range(slot["required"]):
                if i < len(position_players):
                    points[slot_idx] = position_players[i]["projected_points"]
                slot_idx += 1

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
                "slots": {
                    pos: {
                        "required": slot["required"],
                        "filled": slot["filled"]
                    }
                    for pos, slot in team["slots"].items()
                }
            }

        return results

    def get_state(self) -> dict:
        """Get current state for RL model"""
        # Calculate number of roster slots per team
        num_roster_slots = 14  # Fixed size based on standard roster slots

        # Get position needs for RL team
        rl_team = self.teams[self.rl_team_name]
        position_needs = {
            pos: slot["required"] - slot["filled"]
            for pos, slot in rl_team["slots"].items()
        }

        state = {
            'rl_team_budget': self.teams[self.rl_team_name]["current_budget"],
            'opponent_budgets': [self.teams[t]["current_budget"] for t in self.teams if t != self.rl_team_name],
            'draft_turn': len(self.players_drafted),
            'teams': list(self.teams.keys()),
            'predicted_points_per_slot': {
                team: self.get_points_per_slot(team, num_roster_slots)
                for team in self.teams
            },
            'position_needs': position_needs  # Add position needs to state
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

        # Check each position slot requirement
        for position, slot in team["slots"].items():
            if slot["filled"] > slot["required"]:
                msg = f"needs {slot['required'] - slot['filled']} more {position}.\nroster: {', '.join([p['name'] + ' (' + p['position'] + ')' + ' ($' + str(p['bid_amount']) + ')' for p in team['roster']])}"
                logger.debug(f"{team_name} roster invalid: {msg}")
                raise InvalidRosterException(team_name, msg)

        return True
