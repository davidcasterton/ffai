import pandas as pd
import numpy as np
from pathlib import Path
import json
from ffai import get_logger
from ffai.data.espn_scraper import ESPNDraftScraper
logger = get_logger(__name__)

class SeasonSimulator:
    def __init__(self, draft_results, year):
        self.draft_results = draft_results
        self.year = year
        # Set data directory relative to this file
        self.data_dir = Path(__file__).parent / "data/raw"

        # Load data
        scraper = ESPNDraftScraper()
        self.draft_df, self.stats_df, self.weekly_df, self.predraft_df, self.settings = scraper.load_or_fetch_data(self.year)

        self.standings = {team: 0 for team in draft_results.keys()}
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

    def optimize_weekly_roster(self, team, week):
        """Set optimal lineup based on projected points"""
        roster = self.draft_results[team]['roster']
        optimized_roster = {
            pos: [] for pos in self.settings["position_slot_counts"].keys()
        }

        # Get projected points for each player
        players_with_projections = []
        for player in roster:
            # Try to get stats from weekly_df
            stats_mask = (
                (self.weekly_df['week'] == week) &
                (self.weekly_df['player_id'] == player['player_id'])
            )
            player_stats = self.weekly_df[stats_mask]

            if not player_stats.empty:
                projected = player_stats['projected_points'].iloc[0]
            else:
                projected = 0

            players_with_projections.append({
                **player,
                'projected_points': projected
            })

        # Sort by projected points
        players_with_projections.sort(key=lambda x: x['projected_points'], reverse=True)

        # Fill primary positions first
        remaining_players = []
        for player in players_with_projections:
            position = player['position']
            if position in optimized_roster and len(optimized_roster[position]) < self.settings["position_slot_counts"][position]:
                optimized_roster[position].append(player)
            else:
                remaining_players.append(player)

        # Fill FLEX with best remaining RB/WR/TE
        flex_eligible = [p for p in remaining_players if p['position'] in ['RB', 'WR', 'TE']]
        if flex_eligible and len(optimized_roster['RB/WR/TE']) < self.settings["position_slot_counts"]['RB/WR/TE']:
            best_flex = max(flex_eligible, key=lambda x: x['projected_points'])
            optimized_roster['RB/WR/TE'].append(best_flex)
            remaining_players.remove(best_flex)

        # Put remaining players on bench
        optimized_roster['BE'].extend(remaining_players)

        return optimized_roster

    def calculate_score(self, roster, week):
        """Calculate actual score for a team's optimized roster"""
        score = 0
        for position, players in roster.items():
            if position not in ['BE', 'IR']:  # Don't count bench or IR players
                for player in players:
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

if __name__ == "__main__":
    # Load draft results and weekly stats
    draft_results = {}  # Load from auction draft simulator
    simulator = SeasonSimulator(draft_results, 2024)
    simulator.simulate_season()
    standings = simulator.get_standings()
    print(standings)
