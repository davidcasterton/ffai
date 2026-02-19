import pandas as pd
import numpy as np
from pathlib import Path
import json
from ffai import get_logger
from ffai.data.espn_scraper import ESPNDraftScraper, load_league_config
import copy
logger = get_logger(__name__)

class SeasonSimulator:
    def __init__(self, draft_results, year):
        self.draft_results = draft_results
        self.year = year
        # Set data directory relative to this file
        _cfg = load_league_config()
        _league_name = _cfg["league"]["league_name"]
        self.data_dir = Path(__file__).parent.parent / f"data/{_league_name}"

        # Load data
        scraper = ESPNDraftScraper()
        self.draft_df, self.stats_df, self.weekly_df, self.predraft_df, self.settings = scraper.load_or_fetch_data(self.year)

        # Initialize standings for each team
        self.standings = {team_name: 0 for team_name in self.draft_results.keys()}
        self.schedule = self.generate_season_schedule()
        self.weekly_results = {}

    def generate_season_schedule(self):
        """Generate a 17-week schedule where each team plays others fairly"""
        teams = list(self.draft_results.keys())
        schedule = {}

        # Ensure even number of teams
        if len(teams) % 2 != 0:
            raise ValueError(f"Need even number of teams, got {len(teams)}")

        # Round-robin scheduling algorithm
        for week in range(1, 18):
            matchups = []
            # Pair teams for this week
            for i in range(0, len(teams), 2):
                matchups.append((teams[i], teams[(i+1) % len(teams)]))
            schedule[week] = matchups

            # Rotate teams for next week (keeping first team fixed)
            teams = [teams[0]] + [teams[-1]] + teams[1:-1]

        return schedule

    def simulate_season(self):
        """Simulate entire season of weekly matchups"""
        for week in range(1, 18):
            self.simulate_week(week)

    def simulate_week(self, week):
        """Simulate one week of matchups"""
        self.weekly_results[week] = []

        for team1, team2 in self.schedule[week]:
            # Optimize rosters based on predictions
            roster1 = self.optimize_weekly_roster(team1, week)
            roster2 = self.optimize_weekly_roster(team2, week)

            # Calculate actual scores
            score1 = self.calculate_score(roster1, week)
            score2 = self.calculate_score(roster2, week)

            # Determine winner
            winner = team1 if score1 > score2 else team2
            self.standings[winner] += 1

            # Store results
            self.weekly_results[week].append({
                'team1': team1,
                'team1_score': score1,
                'team1_roster': roster1,
                'team2': team2,
                'team2_score': score2,
                'team2_roster': roster2,
                'winner': winner
            })

    def optimize_weekly_roster(self, team_name: str, week: int) -> dict:
        """Optimize weekly roster based on projected points for the given week"""
        # Create copy of roster to modify
        roster = copy.deepcopy(self.draft_results[team_name]["roster"])

        # Get weekly projections for each player
        for slot, player in roster.items():
            if player is not None:
                player_stats = self.weekly_df[
                    (self.weekly_df['week'] == week) &
                    (self.weekly_df['player_id'] == player['player_id'])
                ]
                if not player_stats.empty:
                    player['projected_points'] = player_stats['projected_points'].iloc[0]
                else:
                    player['projected_points'] = 0

        # Sort players by projected points for this week
        available_players = [p for p in roster.values() if p is not None]
        available_players.sort(key=lambda x: x.get('projected_points', 0), reverse=True)

        # Clear roster slots while preserving structure
        original_slots = roster.keys()
        roster = {slot: None for slot in original_slots}

        # Track used player IDs to avoid duplicates
        used_player_ids = set()

        # Fill required positions first
        for pos in ['QB', 'RB', 'WR', 'TE', 'D/ST']:
            pos_players = [p for p in available_players
                          if p['position'] == pos and p['player_id'] not in used_player_ids]
            if pos == 'RB' or pos == 'WR':
                # Fill RB1/RB2 or WR1/WR2
                for i, slot in enumerate([f'{pos} 1', f'{pos} 2']):
                    if i < len(pos_players) and slot in roster:
                        roster[slot] = pos_players[i]
                        used_player_ids.add(pos_players[i]['player_id'])
            else:
                # Fill single position slots
                if pos_players and pos in roster:
                    roster[pos] = pos_players[0]
                    used_player_ids.add(pos_players[0]['player_id'])

        # Fill FLEX with best remaining unused RB/WR/TE
        unused_flex_players = [
            p for p in available_players
            if p['position'] in ['RB', 'WR', 'TE']
            and p['player_id'] not in used_player_ids
        ]
        if unused_flex_players and 'FLEX' in roster:
            roster['FLEX'] = unused_flex_players[0]
            used_player_ids.add(unused_flex_players[0]['player_id'])

        # Fill bench with remaining unused players
        unused_players = [p for p in available_players
                         if p['player_id'] not in used_player_ids]
        bench_slots = [slot for slot in roster.keys() if 'BENCH' in slot]
        for slot, player in zip(bench_slots, unused_players):
            roster[slot] = player
            used_player_ids.add(player['player_id'])

        # logger.info(f"Weekly roster for {team_name} week {week}: {roster}")

        return roster

    def calculate_score(self, roster: dict, week: int) -> float:
        """Calculate actual score for a team's optimized roster"""
        score = 0
        for slot, player in roster.items():
            # Skip bench and empty slots
            if player is not None and not 'BENCH' in slot:
                # Try to get stats from weekly_df
                stats_mask = (
                    (self.weekly_df['week'] == week) &
                    (self.weekly_df['player_id'] == player['player_id'])
                )
                player_stats = self.weekly_df[stats_mask]

                if not player_stats.empty:
                    score += player_stats['points'].iloc[0]
                else:
                    # player did not score points this week
                    score += 0

        return round(score, 2)

    def get_standings(self):
        """Return final standings sorted by wins"""
        return sorted(self.standings.items(), key=lambda x: x[1], reverse=True)

    def get_weekly_results(self):
        """Return detailed results for each week"""
        return self.weekly_results

    def get_team_record(self, team):
        """Get win/loss record for a specific team"""
        wins = self.standings[team]
        losses = 17 - wins  # 17-week season
        return {'wins': wins, 'losses': losses}

    def log_season_results(self, rl_team_name: str = "Team 1"):
        """Log season results for RL team in human readable format"""
        logger.info(f"\nSeason Results for {rl_team_name}:")
        logger.info("-" * 50)

        total_wins = 0
        total_points_for = 0
        total_points_against = 0

        for week, matchups in self.weekly_results.items():
            # Find RL team's matchup
            rl_matchup = next(
                matchup for matchup in matchups
                if rl_team_name in (matchup['team1'], matchup['team2'])
            )

            # Determine if RL team was team1 or team2
            if rl_matchup['team1'] == rl_team_name:
                rl_score = rl_matchup['team1_score']
                opp_name = rl_matchup['team2']
                opp_score = rl_matchup['team2_score']
            else:
                rl_score = rl_matchup['team2_score']
                opp_name = rl_matchup['team1']
                opp_score = rl_matchup['team1_score']

            # Track totals
            total_points_for += rl_score
            total_points_against += opp_score
            if rl_matchup['winner'] == rl_team_name:
                total_wins += 1
                result = "WIN"
            else:
                result = "LOSS"

            # Log weekly result
            logger.info(
                f"Week {week:2d}: {result:4s} vs {opp_name:8s} "
                f"| {rl_score:5.1f} - {opp_score:5.1f}"
            )

        # Log season summary
        logger.info("-" * 50)
        logger.info(
            f"Season Total: {total_wins:2d}-{17-total_wins:2d} "
            f"| Points For: {total_points_for:6.1f} "
            f"| Points Against: {total_points_against:6.1f}"
        )
        logger.info("-" * 50)

if __name__ == "__main__":
    # Load draft results and weekly stats
    draft_results = {}  # Load from auction draft simulator
    simulator = SeasonSimulator(draft_results, 2024)
    simulator.simulate_season()
    simulator.log_RL_weekly_results()
    standings = simulator.get_standings()
    print(standings)
