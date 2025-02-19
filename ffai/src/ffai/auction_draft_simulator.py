import pandas as pd
import numpy as np
from pathlib import Path
import json
import logging
from ffai.data.espn_scraper import ESPNDraftScraper
import torch

logger = logging.getLogger(__name__)

class AuctionDraftSimulator:
    def __init__(self, year, budget=200, rl_team="Team 1", rl_model=None):
        self.year = year
        self.budget = budget
        self.rl_team = rl_team
        self.rl_model = rl_model
        logger.info(f"Initializing AuctionDraftSimulator for year {year}")
        logger.info(f"RL Team: {rl_team}, Budget: ${budget}")

        # Set data directory relative to this file
        self.data_dir = Path(__file__).parent / "data/raw"

        # Load data and settings
        self.player_data = self.load_player_data()
        self.draft_data = self.load_draft_results()
        self.settings = self.load_league_settings()

        # Cache historical bids
        self.historical_bids = self.cache_historical_bids()

        # Initialize teams based on league settings
        self.teams = self.initialize_teams()
        self.nomination_order = list(self.teams.keys())
        self.current_nominator_idx = 0
        self.available_players = self.initialize_available_players()

        # Cache expensive calculations
        self.position_slots = self.settings["position_slot_counts"]
        self.state_cache = {}
        self.last_state_update = 0
        self.draft_round = 0

    def load_league_settings(self):
        """Load league settings from JSON"""
        settings_path = self.data_dir / f'league_settings_770280_{self.year}.json'
        with open(settings_path) as f:
            return json.load(f)

    def initialize_teams(self):
        """Initialize teams with mandatory slot budget reserves"""
        teams = {}
        for i in range(12):
            team_name = f"Team {i+1}"

            # Get required positions from league settings
            position_counts = self.settings["position_slot_counts"]

            # Calculate mandatory slot budgets based on required positions
            mandatory_budgets = {}

            # Core positions - MUST reserve enough for minimum bids
            if "QB" in position_counts:
                mandatory_budgets['QB'] = position_counts['QB']  # $1 per QB needed
            if "RB" in position_counts:
                mandatory_budgets['RB'] = position_counts['RB']
            if "WR" in position_counts:
                mandatory_budgets['WR'] = position_counts['WR']
            if "TE" in position_counts:
                mandatory_budgets['TE'] = position_counts['TE']
            if "D/ST" in position_counts:
                mandatory_budgets['D/ST'] = position_counts['D/ST']
            if "K" in position_counts:
                mandatory_budgets['K'] = position_counts['K']
            if "RB/WR/TE" in position_counts:
                mandatory_budgets['RB/WR/TE'] = position_counts['RB/WR/TE']

            # Reserve minimum $1 for each required roster spot
            reserved_budget = sum(mandatory_budgets.values())
            available_budget = self.budget - reserved_budget

            teams[team_name] = {
                "budget": available_budget,
                "reserved_budgets": mandatory_budgets,
                "roster": [],
                "slots": {
                    pos: {"max": count, "filled": 0}
                    for pos, count in position_counts.items()
                }
            }

            logger.info(f"Initialized {team_name} with ${available_budget} available budget")
            logger.info(f"Reserved budgets: {mandatory_budgets}")

        return teams

    def load_player_data(self):
        """Load player data from files"""
        scraper = ESPNDraftScraper()
        draft_df, stats_df, weekly_df, predraft_df, settings = scraper.load_or_fetch_data(self.year)

        # Merge predraft values with player stats
        player_data = pd.merge(
            stats_df,
            predraft_df[['player_id', 'auction_value', 'adp']],
            on='player_id',
            how='left'
        )

        # Scale down auction values to fit within budget constraints
        max_auction_value = player_data['auction_value'].max()
        if max_auction_value > self.budget * 0.4:  # Cap at 40% of budget
            scale_factor = (self.budget * 0.4) / max_auction_value
            player_data['auction_value'] = player_data['auction_value'] * scale_factor
            logger.info(f"Scaled auction values by factor {scale_factor:.2f}")

        # Fill missing auction values with reasonable defaults based on position
        default_values = {
            'QB': 5,
            'RB': 10,
            'WR': 8,
            'TE': 3,
            'K': 1,
            'D/ST': 1
        }
        player_data['auction_value'] = player_data.apply(
            lambda x: x['auction_value'] if pd.notna(x['auction_value']) else default_values.get(x['position'], 1),
            axis=1
        )

        # Add some logging to verify values
        for pos in ['QB', 'RB', 'WR', 'TE', 'K', 'D/ST']:
            pos_values = player_data[player_data['position'] == pos]['auction_value']
            if not pos_values.empty:
                logger.info(f"{pos} auction values - max: ${pos_values.max():.0f}, "
                           f"mean: ${pos_values.mean():.0f}")

        return player_data

    def load_draft_results(self):
        """Load draft results from CSV"""
        draft_results_path = self.data_dir / f'draft_results_770280_{self.year}.csv'
        return pd.read_csv(draft_results_path)

    def initialize_available_players(self):
        """Initialize available players with valid data only"""
        # Filter out players with missing or invalid data
        valid_players = self.player_data[
            self.player_data['projected_points'].notna() &
            self.player_data['position'].notna() &
            self.player_data['player_id'].notna() &
            self.player_data['name'].notna()
        ].copy()

        # Convert to list of dicts and validate data
        players = []
        for _, player in valid_players.iterrows():
            player_dict = player.to_dict()
            # Ensure all required fields are valid
            if (isinstance(player_dict.get('position'), str) and
                isinstance(player_dict.get('projected_points'), (int, float)) and
                not pd.isna(player_dict.get('player_id'))):
                players.append(player_dict)

        # Sort by projected points
        players.sort(key=lambda x: x['projected_points'], reverse=True)

        logger.info(f"Initialized {len(players)} valid players")
        return players

    def simulate_draft(self):
        """Run the draft simulation"""
        logger.info("Starting draft simulation")
        round_num = 0
        last_roster_count = 0  # Track total players drafted
        stalled_rounds = 0

        while True:
            round_num += 1
            logger.info(f"\nDraft Round {round_num}")

            # Get current total roster count
            current_roster_count = sum(len(team["roster"]) for team in self.teams.values())

            # Check if we made progress
            if current_roster_count > last_roster_count:
                stalled_rounds = 0
                last_roster_count = current_roster_count
            else:
                stalled_rounds += 1

            # Check completion conditions
            if self.all_rosters_complete():
                logger.info("Draft complete: All rosters filled")
                break

            if round_num > 300:
                logger.error("\nDraft exceeded maximum rounds. Diagnostic information:")
                self.log_draft_diagnostic()
                break

            if stalled_rounds > len(self.teams):
                logger.error("\nDraft stalled. Diagnostic information:")
                self.log_draft_diagnostic()
                break

            # Get nomination
            nominating_team = self.nomination_order[self.current_nominator_idx]
            nominated_player = self.select_nomination(nominating_team)

            if nominated_player:
                logger.info(f"{nominating_team} nominates {nominated_player['name']}")

                # Handle bidding
                if nominating_team == self.rl_team:
                    winning_team, winning_bid = self.handle_rl_bidding(nominated_player, self.rl_model)
                else:
                    winning_team, winning_bid = self.simulate_bidding(nominated_player, nominating_team)

                # Process result
                if winning_team and winning_bid > 0:
                    self.add_player_to_roster(winning_team, nominated_player, winning_bid)
                    logger.info(f"{winning_team} wins {nominated_player['name']} for ${winning_bid}")
                else:
                    logger.warning(f"Failed to process bid for {nominated_player['name']}")
                    stalled_rounds += 1
            else:
                logger.info(f"{nominating_team} has no valid nomination")
                stalled_rounds += 1

            # Move to next nominator
            self.current_nominator_idx = (self.current_nominator_idx + 1) % len(self.teams)

        self.log_final_rosters()
        return self.teams

    def log_draft_diagnostic(self):
        """Log detailed diagnostic information about draft state"""
        # Print available players
        logger.error(f"\nAvailable Players: {len(self.available_players)}")
        for pos in set(p["position"] for p in self.available_players):
            pos_players = [p for p in self.available_players if p["position"] == pos]
            logger.error(f"{pos}: {len(pos_players)} players available")
            # Show top 3 available by position
            for p in sorted(pos_players, key=lambda x: x["projected_points"], reverse=True)[:3]:
                logger.error(f"  {p['name']} - {p['projected_points']} projected points")

        # Print team status
        logger.error("\nTeam Status:")
        for team_name, team in self.teams.items():
            unfilled = {
                pos: slot["max"] - slot["filled"]
                for pos, slot in team["slots"].items()
                if pos not in ["BE", "IR"] and slot["filled"] < slot["max"]
            }

            if unfilled:
                logger.error(f"\n{team_name}:")
                logger.error(f"Budget: ${team['budget']}")
                logger.error("Missing positions:")
                for pos, count in unfilled.items():
                    logger.error(f"  {pos}: needs {count} more")

                # Show current roster
                logger.error("Current roster:")
                for player in team["roster"]:
                    logger.error(f"  {player['position']} - {player['name']} (${player['bid_amount']})")

                # Check if team can complete roster
                min_needed = sum(unfilled.values())
                if team['budget'] < min_needed:
                    logger.error(f"  Cannot complete roster: needs ${min_needed} but only has ${team['budget']}")

    def log_draft_status(self):
        """Log current draft status"""
        logger.info("\nDraft Status:")
        for team_name, team in self.teams.items():
            roster_size = len(team['roster'])
            remaining_budget = team['budget']
            logger.info(f"{team_name}: {roster_size} players, ${remaining_budget} remaining")

        logger.info(f"\nAvailable players: {len(self.available_players)}")

    def select_nomination(self, nominating_team):
        """Select a player to nominate for bidding"""
        team_info = self.teams[nominating_team]

        # Skip if team has no budget for even $1 bid
        if team_info["budget"] < 1:
            logger.info(f"{nominating_team} skipped - no budget")
            return None

        # Get unfilled required slots
        required_slots = {
            pos: slot["max"] - slot["filled"]
            for pos, slot in team_info["slots"].items()
            if pos not in ["BE", "IR"] and slot["filled"] < slot["max"]
        }

        if not required_slots:
            logger.info(f"{nominating_team} skipped - no required slots")
            return None

        # Find best available player for any required slot
        candidates = []
        for pos in required_slots:
            pos_players = [
                p for p in self.available_players
                if p["position"] == pos
            ]
            if pos_players:
                # Add best player for this position
                candidates.append(max(pos_players, key=lambda p: p["projected_points"]))

        if not candidates:
            logger.info(f"{nominating_team} skipped - no players for required slots")
            return None

        # Return best overall player
        return max(candidates, key=lambda p: p["projected_points"])

    def can_afford_player(self, team, player, budget=None):
        """Check if team can afford player while completing roster"""
        team_info = self.teams[team]
        if budget is None:
            budget = team_info["budget"]

        # Calculate minimum needed for remaining slots
        unfilled_slots = sum(
            slot["max"] - slot["filled"]
            for pos, slot in team_info["slots"].items()
            if pos not in ["BE", "IR"]
        )
        min_needed = max(0, unfilled_slots - 1)  # Subtract 1 for this player

        # Estimate max affordable bid
        max_affordable = budget - min_needed

        # Estimate player cost based on projected points
        estimated_cost = min(player["projected_points"], max_affordable)

        return estimated_cost >= 1

    def get_team_needs(self, team):
        """Calculate positional needs for a team"""
        needs = {}
        slots = self.teams[team]["slots"]

        # Calculate need score for each position
        # Higher score = greater need
        for position, slot in slots.items():
            if position == "BE" or position == "IR":
                continue

            filled = slot["filled"]
            maximum = slot["max"]

            # More urgent need if position is empty
            if filled == 0:
                needs[position] = 2.0
            # Some need if not at maximum
            elif filled < maximum:
                needs[position] = 1.0
            # No need if position is full
            else:
                needs[position] = 0.0

        return needs

    def score_nomination(self, player, team_needs):
        """Score a potential nomination based on team needs and value"""
        position = player["position"]
        team_info = self.teams[self.rl_team]

        # Base score is projected points
        score = float(player["projected_points"])

        # Adjust score based on position need
        if position in team_needs:
            need_multiplier = 1.0 + team_needs[position]
            score *= need_multiplier

        # Adjust for FLEX eligibility
        if (position in ["RB", "WR", "TE"] and
            "RB/WR/TE" in team_needs and
            team_needs["RB/WR/TE"] > 0):
            score *= 1.2

        # Adjust for budget considerations
        max_bid = min(
            team_info["budget"],
            self.historical_bids.get(player["player_id"], player["projected_points"]),
            team_info["budget"] * 0.7
        )

        # Penalize if likely too expensive
        if max_bid < player["projected_points"] * 0.5:
            score *= 0.5

        # Bonus for value picks
        historical_bid = self.historical_bids.get(player["player_id"])
        if historical_bid:
            value_ratio = player["projected_points"] / historical_bid
            score *= value_ratio

        return score

    def select_rl_nomination(self, available):
        """Select nomination for RL team using team needs and value scoring"""
        if not available:
            return None

        team_needs = self.get_team_needs(self.rl_team)

        try:
            # Score each available player
            player_scores = []
            for player in available[:10]:  # Limit to top 10 for efficiency
                score = self.score_nomination(player, team_needs)
                player_scores.append((player, score))

            # Return player with highest score
            if player_scores:
                return max(player_scores, key=lambda x: x[1])[0]

        except Exception as e:
            logger.error(f"Error in RL nomination: {str(e)}")
            # Fallback to simple selection
            if available:
                return max(available[:10],
                          key=lambda p: p['projected_points'])

        return None

    def simulate_bidding(self, player: dict, nominating_team: str) -> tuple:
        """Simulate bidding process for a player"""
        current_bid = 1  # Start at $1
        current_winner = nominating_team
        active_bidders = set(self.teams.keys())

        while len(active_bidders) > 1:
            # Remove teams that can't/won't bid
            for team in list(active_bidders):
                team_bid = self.get_bid_amount(team, player)
                if team_bid <= current_bid:
                    active_bidders.remove(team)

            # Get highest bid from remaining bidders
            if len(active_bidders) > 1:
                highest_bidder = None
                highest_bid = current_bid

                for team in active_bidders:
                    if team == current_winner:
                        continue

                    bid = self.get_bid_amount(team, player)
                    if bid > highest_bid:
                        highest_bid = bid
                        highest_bidder = team

                if highest_bidder:
                    current_bid = highest_bid
                    current_winner = highest_bidder
                else:
                    break
            else:
                break

        # Verify winner can afford the bid
        if current_bid > self.teams[current_winner]["budget"]:
            logger.warning(f"{current_winner} cannot afford ${current_bid} bid")
            return None, 0

        # Verify winner can still complete roster after bid
        remaining_budget = self.teams[current_winner]["budget"] - current_bid
        min_needed = self.min_needed_for_remaining_slots(current_winner)
        if remaining_budget < min_needed:
            logger.warning(f"{current_winner} needs ${min_needed} to complete roster but would only have ${remaining_budget}")
            return None, 0

        return current_winner, current_bid

    def handle_rl_bidding(self, player, rl_model=None):
        """Handle bidding with budget management"""
        if rl_model is None:
            return self.simulate_bidding(player, self.rl_team)

        current_bid = 1
        current_winner = self.rl_team

        # Only include teams that can actually bid
        active_bidders = {
            team for team in self.teams
            if self.can_add_player(team, player) and
               self.teams[team]["budget"] > 1
        }

        # Get RL bid once at start
        draft_state = self.get_draft_state()
        state_tensor = rl_model.get_state_tensor(draft_state, player)
        max_rl_bid = min(
            rl_model.get_bid_action(state_tensor, self.teams[self.rl_team]['budget']),
            self.teams[self.rl_team]['budget'] * 0.5  # Cap at 50% of budget
        )

        while len(active_bidders) > 1:
            any_bids = False

            # Other teams bid
            for team in list(active_bidders - {self.rl_team}):
                if not self.should_bid(team, player, current_bid):
                    active_bidders.remove(team)
                    continue

                current_bid += 1
                current_winner = team
                any_bids = True

            # RL team bids
            if self.rl_team in active_bidders:
                if current_bid > max_rl_bid:
                    active_bidders.remove(self.rl_team)
                elif current_bid <= max_rl_bid:
                    current_bid += 1
                    current_winner = self.rl_team
                    any_bids = True

            if not any_bids:
                break

        return current_winner, current_bid

    def should_bid(self, team, player, current_bid):
        """Determine if team should bid based on budget and needs"""
        team_info = self.teams[team]

        # Quick budget check
        if team_info["budget"] <= current_bid:
            return False

        # Don't bid if roster is full
        if all(slot["filled"] >= slot["max"] for slot in team_info["slots"].values()):
            return False

        # Check if player fills a needed slot
        if not self.can_add_player(team, player):
            return False

        # Late draft logic - be more aggressive with remaining budget
        if len(self.available_players) < 50:
            return current_bid <= team_info["budget"]

        # Normal draft logic
        max_bid = min(
            team_info["budget"],
            self.historical_bids.get(player["player_id"], player["projected_points"]),
            team_info["budget"] * 0.7
        )

        return current_bid <= max_bid

    def cache_historical_bids(self):
        """Cache historical bid amounts for quick lookup"""
        if self.draft_data is None:
            return {}

        bids = {}
        for _, row in self.draft_data.iterrows():
            bids[row["player_id"]] = row["bid_amount"]
        return bids

    def get_historical_bid(self, player_id):
        """Get historical draft value for a player from cache"""
        return self.historical_bids.get(player_id)

    def add_player_to_roster(self, team, player, bid_amount):
        """Add player to roster with budget management"""
        team_info = self.teams[team]
        position = player["position"]

        # Use reserved budget if available for this position
        if position in team_info["reserved_budgets"]:
            reserved = team_info["reserved_budgets"][position]
            if team_info["slots"][position]["filled"] < team_info["slots"][position]["max"]:
                if bid_amount <= reserved:
                    team_info["reserved_budgets"][position] = 0
                else:
                    team_info["reserved_budgets"][position] = 0
                    team_info["budget"] -= (bid_amount - reserved)
            else:
                team_info["budget"] -= bid_amount
        else:
            team_info["budget"] -= bid_amount

        # Add to roster
        team_info["roster"].append({
            "player_id": player["player_id"],
            "name": player["name"],
            "position": position,
            "projected_points": player["projected_points"],
            "bid_amount": bid_amount
        })

        self.update_roster_slots(team, player)
        self.available_players.remove(player)

    def update_roster_slots(self, team, player):
        """Update team's roster slots after adding a player"""
        position = player["position"]
        slots = self.teams[team]["slots"]

        # Try to fill primary position first
        if position in slots and slots[position]["filled"] < slots[position]["max"]:
            slots[position]["filled"] += 1
        # Try FLEX for eligible positions
        elif position in ["RB", "WR", "TE"] and slots["RB/WR/TE"]["filled"] < slots["RB/WR/TE"]["max"]:
            slots["RB/WR/TE"]["filled"] += 1
        # Otherwise use bench
        else:
            slots["BE"]["filled"] += 1

    def can_add_player(self, team, player):
        """Check if a team can add a player to their roster"""
        try:
            slots = self.teams[team]["slots"]
            position = player.get("position")

            # Quick validation without pandas
            if not position or not isinstance(position, str):
                return False

            # Budget checks
            if self.teams[team]["budget"] < 1:
                return False

            # Quick slot check
            if slots[position]["filled"] >= slots[position]["max"]:
                # Check FLEX for eligible positions
                if position in ["RB", "WR", "TE"] and slots["RB/WR/TE"]["filled"] < slots["RB/WR/TE"]["max"]:
                    return True
                # Check bench
                if slots["BE"]["filled"] < slots["BE"]["max"]:
                    return True
                return False

            return True

        except Exception as e:
            logger.error(f"Error checking if can add player: {str(e)}")
            return False

    def min_needed_for_remaining_slots(self, team):
        """Calculate minimum amount needed to fill remaining roster spots"""
        empty_slots = sum(
            slot["max"] - slot["filled"]
            for slot in self.teams[team]["slots"].values()
        )
        return empty_slots  # Minimum $1 per player

    def all_rosters_complete(self):
        """Check if draft should end"""
        # Quick check - no more players
        if not self.available_players:
            logger.info("Draft complete: No more available players")
            return True

        # Check each team's status
        for team_name, team in self.teams.items():
            # Get unfilled required slots
            unfilled = {
                pos: slot["max"] - slot["filled"]
                for pos, slot in team["slots"].items()
                if pos not in ["BE", "IR"] and slot["filled"] < slot["max"]
            }

            # If team has unfilled slots and any budget, draft continues
            if unfilled and team["budget"] >= 1:
                return False

        logger.info("Draft complete: All teams either complete or out of budget")
        return True

    def log_final_rosters(self):
        """Log final roster status for all teams"""
        logger.info("\nFinal Roster Status:")
        for team_name, team in self.teams.items():
            logger.info(f"\n{team_name}:")
            logger.info(f"Remaining budget: ${team['budget']}")
            logger.info("Roster slots:")
            for pos, slot in team["slots"].items():
                logger.info(f"  {pos}: {slot['filled']}/{slot['max']}")

            logger.info("Players:")
            for player in team["roster"]:
                logger.info(f"  {player['position']} - {player['name']} (${player['bid_amount']}) - Projected: {player['projected_points']:.1f}")

    def get_draft_results(self):
        """Return the results of the draft"""
        return self.teams

    def get_draft_state(self):
        """Get current state with caching"""
        # Always calculate initial state
        if self.draft_round == 0 or self.draft_round != self.last_state_update:
            # Basic state that's always included
            state = {
                'rl_team_budget': float(self.teams[self.rl_team]['budget']),
                'opponent_budgets': [float(self.teams[t]['budget']) for t in self.teams
                                   if t != self.rl_team],
                'draft_turn': self.current_nominator_idx,
                'teams': list(self.teams.keys()),
                'predicted_points_per_slot': {},
                'auction_spent_per_slot': {}
            }

            # Calculate points and spent for initial state or periodic updates
            if self.draft_round == 0 or self.draft_round % 5 == 0:
                for team in self.teams:
                    state['predicted_points_per_slot'][team] = self.get_points_per_slot(team)
                    state['auction_spent_per_slot'][team] = self.get_spent_per_slot(team)

            self.state_cache = state
            self.last_state_update = self.draft_round
            return state

        return self.state_cache

    def get_points_per_slot(self, team):
        """Get predicted points for each roster slot"""
        points = []
        for slot in self.settings["position_slot_counts"].keys():
            slot_points = []
            for player in self.teams[team]['roster']:
                if slot == "RB/WR/TE" and player['position'] in ["RB", "WR", "TE"]:
                    slot_points.append(player['projected_points'])
                elif player['position'] == slot:
                    slot_points.append(player['projected_points'])
            points.extend(slot_points + [0] * (self.settings["position_slot_counts"][slot] - len(slot_points)))
        return points

    def get_spent_per_slot(self, team):
        """Get amount spent on each roster slot"""
        spent = []
        for slot in self.settings["position_slot_counts"].keys():
            slot_spent = []
            for player in self.teams[team]['roster']:
                if slot == "RB/WR/TE" and player['position'] in ["RB", "WR", "TE"]:
                    slot_spent.append(player['bid_amount'])
                elif player['position'] == slot:
                    slot_spent.append(player['bid_amount'])
            spent.extend(slot_spent + [0] * (self.settings["position_slot_counts"][slot] - len(slot_spent)))
        return spent

    def can_complete_roster(self, team_name):
        """Check if team can theoretically complete roster"""
        team = self.teams[team_name]
        budget = team["budget"]

        # Get minimum cost to fill mandatory slots
        min_total = 0
        for pos, slot in team["slots"].items():
            if pos in ["BE", "IR"]:
                continue

            needed = slot["max"] - slot["filled"]
            if needed > 0:
                min_total += needed  # Minimum $1 per player

        return budget >= min_total

    def force_complete_rosters(self):
        """Fill remaining slots with minimum cost players"""
        for team_name, team in self.teams.items():
            if team["budget"] <= 0:
                continue

            unfilled = self.get_unfilled_slots(team_name)
            for pos in unfilled:
                candidates = [p for p in self.available_players
                            if p["position"] == pos and
                            p["projected_points"] > 0]
                if candidates:
                    player = min(candidates, key=lambda p: p["projected_points"])
                    self.add_player_to_roster(team_name, player, 1)

    def get_bid_amount(self, team_name: str, player: dict) -> int:
        """Determine bid amount for a player"""
        if team_name == self.rl_team and self.rl_model:
            return self.get_rl_bid(player)

        # Get team's remaining budget and needs
        team = self.teams[team_name]
        budget = team["budget"]
        needs = self.get_team_needs(team_name)

        # Calculate minimum needed budget for remaining required positions
        min_needed = sum(1 for pos, count in needs.items() if pos != 'BE')  # $1 per required position

        # Don't bid if it would prevent filling required positions
        if budget - min_needed < 1:
            # Desperation mode - if this is a needed position and costs $1, take it
            if player['position'] in needs and budget >= 1:
                return 1
            return 0

        # Base value is the predraft auction value
        base_value = player['auction_value']

        # Scale bid based on how many roster spots are left to fill
        total_slots = sum(slot["max"] for pos, slot in team["slots"].items() if pos not in ["BE", "IR"])
        filled_slots = sum(slot["filled"] for pos, slot in team["slots"].items() if pos not in ["BE", "IR"])
        remaining_slots = total_slots - filled_slots

        # Be more conservative early in the draft
        if remaining_slots > total_slots * 0.5:  # More than half slots remaining
            base_value *= 0.7  # Bid 70% of value early
        elif remaining_slots > total_slots * 0.25:  # More than quarter slots remaining
            base_value *= 0.85  # Bid 85% of value in middle

        # Add small randomness
        randomness = np.random.uniform(-0.1, 0.1)
        value = base_value * (1 + randomness)

        # Increase value for needed positions
        if player['position'] in needs:
            value *= 1.2  # 20% boost for needed positions

        # Position-specific caps as percentage of remaining flexible budget
        flexible_budget = budget - min_needed
        if player['position'] in ['K', 'D/ST']:
            value = min(value, flexible_budget * 0.1, 3)  # Max 10% of flexible budget or $3
        elif player['position'] == 'QB':
            value = min(value, flexible_budget * 0.25, 30)  # Max 25% of flexible budget or $30
        elif player['position'] == 'TE':
            value = min(value, flexible_budget * 0.3, 40)  # Max 30% of flexible budget or $40
        else:  # RB/WR
            value = min(value, flexible_budget * 0.4, 65)  # Max 40% of flexible budget or $65

        # Ensure we don't bid more than our flexible budget
        value = min(value, flexible_budget)

        return max(1, round(value))

    def nominate_player(self, team_name: str) -> dict:
        """Select a player to nominate for auction"""
        if team_name == self.rl_team and self.rl_model:
            return self.get_rl_nomination()

        team = self.teams[team_name]
        needs = self.get_team_needs(team_name)

        # Filter available players by team needs first
        needed_players = [p for p in self.available_players if p['position'] in needs]

        # If no needed players, use all available players
        candidates = needed_players if needed_players else self.available_players

        # Nominate based on team's budget situation
        budget_ratio = team["budget"] / self.budget
        if budget_ratio < 0.3:  # Low budget
            # Nominate cheaper players
            candidates.sort(key=lambda p: p['auction_value'])
            return candidates[0] if candidates else None
        elif budget_ratio > 0.7:  # High budget
            # Nominate valuable players
            candidates.sort(key=lambda p: p['auction_value'], reverse=True)
            return candidates[0] if candidates else None
        else:  # Medium budget
            # Mix of strategies
            if np.random.random() < 0.7:
                # Nominate needed positions
                return np.random.choice(candidates) if candidates else None
            else:
                # Nominate high value players to drain others' budgets
                all_sorted = sorted(self.available_players, key=lambda p: p['auction_value'], reverse=True)
                return all_sorted[0] if all_sorted else None

    def get_rl_bid(self, player: dict) -> int:
        """Get bid amount from RL model"""
        if not self.rl_model:
            return 0

        # Get current state
        state = self.get_draft_state()

        # Build state tensor as before...
        state_values = []
        state_values.append(float(player['player_id']))
        state_values.append(float(player['auction_value']))
        state_values.append(float(player['projected_points']))
        state_values.append(float(state['rl_team_budget']))
        state_values.extend(state['opponent_budgets'])
        state_values.append(float(state['draft_turn']))

        num_roster_slots = sum(self.settings["position_slot_counts"].values())
        for team in state['teams']:
            points = state['predicted_points_per_slot'][team][:num_roster_slots]
            points.extend([0] * (num_roster_slots - len(points)))
            spent = state['auction_spent_per_slot'][team][:num_roster_slots]
            spent.extend([0] * (num_roster_slots - len(spent)))
            state_values.extend(points)
            state_values.extend(spent)

        # Convert to tensor
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state_values).unsqueeze(0)

            # Handle training vs inference differently
            if self.rl_model.training:
                # During training, use duplicated tensor
                state_tensor = state_tensor.repeat(2, 1)
                output = self.rl_model(state_tensor)
                # Take mean of the duplicated predictions
                bid_value = output.mean().item()
            else:
                # During inference, just use single sample
                output = self.rl_model(state_tensor)
                bid_value = output.item()

        # Rest of the method remains the same...
        bid = max(1, round(bid_value))
        budget = self.teams[self.rl_team]["budget"]
        needs = self.get_team_needs(self.rl_team)
        min_needed = sum(1 for pos, count in needs.items() if pos != 'BE')
        flexible_budget = budget - min_needed
        bid = min(bid, flexible_budget)

        if player['position'] in ['K', 'D/ST']:
            bid = min(bid, 3)
        elif player['position'] == 'QB':
            bid = min(bid, 30)
        elif player['position'] == 'TE':
            bid = min(bid, 40)
        else:  # RB/WR
            bid = min(bid, 65)

        return max(1, round(bid))

    def get_rl_nomination(self) -> dict:
        """Get nomination from RL model"""
        if not self.rl_model:
            return self.nominate_player(self.rl_team)

        # For now, use same nomination logic as other teams
        # This can be enhanced later to use RL model for nominations
        return self.nominate_player(self.rl_team)
