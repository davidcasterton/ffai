import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, Dict
import logging
from datetime import datetime
from espn_api.football import League
from espn_api.requests.espn_requests import ESPNInvalidLeague
import argparse
import json
from ffai import get_logger

# ESPN API Authentication
SWID = '{8DF64DA2-44CD-4383-87DA-FF2FECDFB6FD}'
ESPN_S2 = 'AEAxi0mL0zKA5Ck7gIl45WaDTC332wzUITx%2FewTe%2BckX3hpNwWVnUgrMstNR8l1tJVjb86s97U6E3JhHpEihdyrm5lIHR%2FnrdPiV%2BK%2F%2Bl2X0M8owhfjMMxLJ7WvjsbSBF6cH0oomwVvQUkZLWwXZ%2BiXu%2BYf8he8WX1JkZUdAQiKnejOld19RM87ZHiYDjoXGnu0yxEsWvfpqoAL8vxgWh%2B5YukiTZ1J5FeQo8PadRezprPktrVGN0hHYxE1Yd87yGtXpjZa4zczMm8JrAK7KGAjZ9xr%2B1RGtZcvVAAawVMN8kjfEglR50%2BDWUSIEpBlvCbcGwv7I6qtPdDAqLoNkX5ko'
LEAGUE_ID = "770280"

logger = get_logger(__name__)

class ESPNDraftScraper:
    def __init__(self, league_id: str = LEAGUE_ID, year: int = None):
        """Initialize scraper with league ID and optional year."""
        # Create absolute path relative to current file
        self.raw_data_dir = Path(__file__).parent / "raw"
        self.raw_data_dir.mkdir(parents=True, exist_ok=True)

        self.league_id = league_id
        self.year = year
        self.league = None  # Will be set by set_year()

        if year:
            self.set_year(year)

    def set_year(self, year: int) -> None:
        """Set the year and initialize League object."""
        self.year = year
        logger.info(f"Initializing league for year {year}")
        self.league = League(
            league_id=self.league_id,
            year=year,
            espn_s2=ESPN_S2,
            swid=SWID
        )

    def get_draft_results(self) -> pd.DataFrame:
        """Get draft results for current league and year"""
        picks = []
        for pick in self.league.draft:
            owner_info = pick.team.owners[0] if pick.team.owners else {}
            manager_id = owner_info.get('id', '').strip('{}') if owner_info.get('id') else None
            picks.append({
                "round": pick.round_num,
                "pick_number": pick.round_pick,
                "player_id": pick.playerId,
                "team_id": pick.team.team_id,
                "player_name": pick.playerName,
                "team_name": pick.team.team_name,
                "manager_display_name": owner_info.get('displayName'),
                "manager_first_name": owner_info.get('firstName'),
                "manager_last_name": owner_info.get('lastName'),
                "manager_id": manager_id,
                "bid_amount": pick.bid_amount if hasattr(pick, 'bid_amount') else None,
            })
        return pd.DataFrame(picks)

    def get_player_stats(self) -> pd.DataFrame:
        """Get end-of-season player statistics"""
        players = []
        for team in self.league.teams:
            owner_info = team.owners[0] if team.owners else {}
            manager_id = owner_info.get('id', '').strip('{}') if owner_info.get('id') else None
            for player in team.roster:
                players.append({
                    "player_id": player.playerId,
                    "name": player.name,
                    "position": player.position,
                    "pro_team": player.proTeam,
                    "total_points": player.total_points,
                    "projected_points": player.projected_total_points,
                    "fantasy_team": team.team_name,
                    "fantasy_team_id": team.team_id,
                    "manager_display_name": owner_info.get('displayName'),
                    "manager_first_name": owner_info.get('firstName'),
                    "manager_last_name": owner_info.get('lastName'),
                    "manager_id": manager_id
                })
        return pd.DataFrame(players)

    def get_league_settings(self) -> dict:
        """Get league roster and scoring settings."""
        settings = {
            'position_slot_counts': {k: v for k,v in self.league.settings.position_slot_counts.items() if v>0},
            'scoring_format': self.league.settings.scoring_format,
            'scoring_type': self.league.settings.scoring_type,
            'team_count': self.league.settings.team_count,
        }
        return settings

    def get_predraft_player_data(self) -> pd.DataFrame:
        """Get pre-draft player values and projections"""
        logger.info(f"Attempting to fetch pre-draft data for {self.year}...")

        try:
            # Get draft data to extract auction values
            draft_data = {}
            for pick in self.league.draft:
                try:
                    # Different versions of ESPN API have different pick structures
                    if hasattr(pick, 'playerId'):
                        player_id = pick.playerId
                    elif hasattr(pick, 'playerPoolEntry') and pick.playerPoolEntry:
                        player_id = pick.playerPoolEntry['id']
                    else:
                        continue

                    draft_data[player_id] = {
                        'auction_value': pick.bid_amount if hasattr(pick, 'bid_amount') else 0,
                        'draft_position': pick.round_num if hasattr(pick, 'round_num') else 0
                    }
                except AttributeError:
                    continue

            # Get all players through the League API
            players = []

            # Get all rostered players
            for team in self.league.teams:
                for player in team.roster:
                    try:
                        player_info = {
                            'player_id': player.playerId,
                            'name': player.name,
                            'position': player.position,
                            'projected_points': player.projected_total_points if hasattr(player, 'projected_total_points') else 0,
                            'auction_value': draft_data.get(player.playerId, {}).get('auction_value', 0),
                            'adp': draft_data.get(player.playerId, {}).get('draft_position', 0),
                            'team': player.proTeam if hasattr(player, 'proTeam') else None,
                            'status': player.injuryStatus if hasattr(player, 'injuryStatus') else None
                        }
                        players.append(player_info)
                    except AttributeError as e:
                        logger.debug(f"Skipping player due to missing attribute: {e}")
                        continue

            # Also get free agents for each position
            positions = ['QB', 'RB', 'WR', 'TE', 'K', 'D/ST']
            for pos in positions:
                try:
                    free_agents = self.league.free_agents(position=pos)
                    for player in free_agents:
                        try:
                            player_info = {
                                'player_id': player.playerId,
                                'name': player.name,
                                'position': player.position,
                                'projected_points': player.projected_total_points if hasattr(player, 'projected_total_points') else 0,
                                'auction_value': draft_data.get(player.playerId, {}).get('auction_value', 0),
                                'adp': draft_data.get(player.playerId, {}).get('draft_position', 0),
                                'team': player.proTeam if hasattr(player, 'proTeam') else None,
                                'status': player.injuryStatus if hasattr(player, 'injuryStatus') else None
                            }
                            players.append(player_info)
                        except AttributeError as e:
                            logger.debug(f"Skipping free agent due to missing attribute: {e}")
                            continue
                except Exception as e:
                    logger.debug(f"Error getting free agents for position {pos}: {e}")
                    continue

            if not players:
                logger.warning("No players found in league data")
                return None

            logger.info(f"Successfully extracted {len(players)} players")

            # Convert to DataFrame and sort by projected points
            df = pd.DataFrame(players)
            df = df.sort_values('projected_points', ascending=False)

            # Remove duplicates by player_id
            df = df.drop_duplicates(subset='player_id', keep='first')

            return df

        except Exception as e:
            logger.error(f"Error fetching pre-draft data: {str(e)}")
            logger.exception("Stack trace:")
            return None

    def load_or_fetch_data(self, year: int, force: bool = False) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
        """Load data from files if available, otherwise fetch from ESPN"""

        # Check if all required files exist
        files_exist = all(
            (self.raw_data_dir / f).exists() for f in [
                f"draft_results_{self.league_id}_{year}.csv",
                f"player_stats_{self.league_id}_{year}.csv",
                f"weekly_stats_{self.league_id}_{year}.csv",
                f"league_settings_{self.league_id}_{year}.json",
                f"predraft_values_{self.league_id}_{year}.csv"
            ]
        )

        logger.info(f"Initializing league for year {year}")

        if files_exist and not force:
            logger.info(f"Loading existing data files for league {self.league_id}, year {year}:")
            # Load from files without making HTTP requests
            draft_df = pd.read_csv(self.raw_data_dir / f"draft_results_{self.league_id}_{year}.csv")
            logger.debug(f"  Draft data: draft_results_{self.league_id}_{year}.csv")

            stats_df = pd.read_csv(self.raw_data_dir / f"player_stats_{self.league_id}_{year}.csv")
            logger.debug(f"  Stats data: player_stats_{self.league_id}_{year}.csv")

            weekly_df = pd.read_csv(self.raw_data_dir / f"weekly_stats_{self.league_id}_{year}.csv")
            logger.debug(f"  Weekly stats: weekly_stats_{self.league_id}_{year}.csv")

            with open(self.raw_data_dir / f"league_settings_{self.league_id}_{year}.json", 'r') as f:
                settings = json.load(f)
            logger.debug(f"  League settings: league_settings_{self.league_id}_{year}.json")

            predraft_df = pd.read_csv(self.raw_data_dir / f"predraft_values_{self.league_id}_{year}.csv")
            logger.debug(f"  Pre-draft data: predraft_values_{self.league_id}_{year}.csv")
        else:
            # Only make HTTP requests if we don't have cached data
            logger.info(f"Fetching data from ESPN for league {self.league_id}, year {year}")
            self.set_year(year)  # This makes the HTTP requests

            # Fetch and save data
            draft_df = self.get_draft_results()
            stats_df = self.get_player_stats()
            weekly_df = self.get_weekly_stats()
            settings = self.get_league_settings()
            predraft_df = self.get_predraft_player_data()

            # Add year to player stats
            stats_df['year'] = year

            # Filter out players who weren't in the NFL that year
            if 'total_points' in stats_df.columns:
                stats_df = stats_df[stats_df['total_points'].notna()]

            # Save all data
            self.save_data(draft_df, stats_df, weekly_df, predraft_df, settings, self.league_id, year)

        return draft_df, stats_df, weekly_df, predraft_df, settings

    def get_weekly_stats(self) -> pd.DataFrame:
        """Get weekly player statistics for entire season."""
        if self.year < 2019:
            logger.info(f"Weekly stats not available before 2019. Returning empty DataFrame for {self.year}")
            return pd.DataFrame(columns=[
                'week', 'player_id', 'player_name', 'position',
                'pro_team', 'points', 'projected_points', 'stats'
            ])

        weekly_stats = []

        # ESPN has data for weeks 1-17 (or 18 for newer seasons)
        max_week = 18 if self.year >= 2021 else 17

        logger.info(f"Fetching weekly stats for {self.year} season...")
        for week in range(1, max_week + 1):
            try:
                box_scores = self.league.box_scores(week)

                for game in box_scores:
                    # Get home team players
                    for player in game.home_lineup:
                        stats = {
                            'week': week,
                            'player_id': player.playerId,
                            'player_name': player.name,
                            'position': player.position,
                            'pro_team': player.proTeam,
                            'points': player.points,
                            'projected_points': player.projected_points,
                            'stats': player.stats  # This contains detailed stats like passing_yards, rushing_yards, etc.
                        }
                        weekly_stats.append(stats)

                    # Get away team players
                    for player in game.away_lineup:
                        stats = {
                            'week': week,
                            'player_id': player.playerId,
                            'player_name': player.name,
                            'position': player.position,
                            'pro_team': player.proTeam,
                            'points': player.points,
                            'projected_points': player.projected_points,
                            'stats': player.stats
                        }
                        weekly_stats.append(stats)

                logger.debug(f"Processed week {week}")
            except Exception as e:
                logger.warning(f"Error fetching week {week}: {e}")
                continue

        # Convert to DataFrame and return
        df = pd.DataFrame(weekly_stats)
        return df

    def save_data(self,
        draft_df: pd.DataFrame,
        stats_df: pd.DataFrame,
        weekly_df: pd.DataFrame,
        predraft_df: pd.DataFrame,
        settings: dict,
        league_id: str,
        year: int
    ) -> Tuple[str, str, str, str, str]:
        """Save all data to files"""
        # Update filenames to remove timestamp
        draft_filename = f"draft_results_{league_id}_{year}.csv"
        stats_filename = f"player_stats_{league_id}_{year}.csv"
        weekly_filename = f"weekly_stats_{league_id}_{year}.csv"
        settings_filename = f"league_settings_{league_id}_{year}.json"
        predraft_filename = f"predraft_values_{league_id}_{year}.csv"

        # Save files
        draft_df.to_csv(self.raw_data_dir / draft_filename, index=False)
        stats_df.to_csv(self.raw_data_dir / stats_filename, index=False)
        weekly_df.to_csv(self.raw_data_dir / weekly_filename, index=False)
        predraft_df.to_csv(self.raw_data_dir / predraft_filename, index=False)
        with open(self.raw_data_dir / settings_filename, 'w') as f:
            json.dump(settings, f, indent=2)

        logger.info(f"Saved data files:")
        logger.info(f"  Draft data: {draft_filename}")
        logger.info(f"  Stats data: {stats_filename}")
        logger.info(f"  Weekly stats: {weekly_filename}")
        logger.info(f"  Pre-draft data: {predraft_filename}")
        logger.info(f"  League settings: {settings_filename}")

        return draft_filename, stats_filename, weekly_filename, predraft_filename, settings_filename

def main():
    """Main function to run scraper"""
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--force", action="store_true", help="Force fetch new data")
    args = parser.parse_args()

    scraper = ESPNDraftScraper()

    # Get data for all seasons
    for year in range(2009, 2025):
        logger.info(f"Fetching {year} data...")

        try:
            # Get all data including pre-draft values
            draft_df, stats_df, weekly_df, predraft_df, settings = scraper.load_or_fetch_data(year, force=args.force)
            logger.info(f"Successfully fetched data for year {year}")

        except ESPNInvalidLeague as e:
            logger.error(f"Error fetching data for year {year}: {e}")
            continue

if __name__ == "__main__":
    main()
