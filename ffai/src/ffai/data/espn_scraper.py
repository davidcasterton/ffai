import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, Tuple, Dict
import logging
from espn_api.football import League
from espn_api.requests.espn_requests import ESPNInvalidLeague
import argparse
import json
import yaml
from ffai import get_logger

logger = get_logger(__name__)

# Default config path
_CONFIG_PATH = Path(__file__).parent.parent / "config" / "league.yaml"


def load_league_config(config_path: Optional[Path] = None) -> dict:
    """Load league config from yaml file."""
    path = config_path or _CONFIG_PATH
    if not path.exists():
        raise FileNotFoundError(
            f"League config not found at {path}. "
            "Copy config/league.yaml.example to config/league.yaml and fill in your credentials."
        )
    with open(path) as f:
        return yaml.safe_load(f)


def get_credentials(config_path: Optional[Path] = None) -> Tuple[str, str, str]:
    """Return (league_id, swid, espn_s2) from config file."""
    config = load_league_config(config_path)
    league_id = config["league"]["league_id"]
    swid = config["auth"]["swid"]
    espn_s2 = config["auth"]["espn_s2"]
    if swid == "REPLACE_WITH_SWID" or espn_s2 == "REPLACE_WITH_ESPN_S2":
        raise ValueError(
            "ESPN credentials not configured. Edit config/league.yaml with your SWID and espn_s2 cookies."
        )
    return league_id, swid, espn_s2


class ESPNDraftScraper:
    def __init__(self, league_id: str = None, year: int = None, config_path: Path = None):
        """Initialize scraper. Credentials loaded from config/league.yaml."""
        self.raw_data_dir = Path(__file__).parent / "raw"
        self.raw_data_dir.mkdir(parents=True, exist_ok=True)

        cfg_league_id, self.swid, self.espn_s2 = get_credentials(config_path)
        self.league_id = league_id or cfg_league_id
        self.year = year
        self.league = None

        if year:
            self.set_year(year)

    def set_year(self, year: int) -> None:
        """Set the year and initialize League object."""
        self.year = year
        logger.info(f"Initializing league for year {year}")
        self.league = League(
            league_id=self.league_id,
            year=year,
            espn_s2=self.espn_s2,
            swid=self.swid,
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
                    "manager_id": manager_id,
                })
        return pd.DataFrame(players)

    def get_league_settings(self) -> dict:
        """Get league roster and scoring settings."""
        settings = {
            'position_slot_counts': {k: v for k, v in self.league.settings.position_slot_counts.items() if v > 0},
            'scoring_format': self.league.settings.scoring_format,
            'scoring_type': self.league.settings.scoring_type,
            'team_count': self.league.settings.team_count,
        }
        return settings

    def get_predraft_player_data(self) -> pd.DataFrame:
        """Get pre-draft player values and projections"""
        logger.info(f"Attempting to fetch pre-draft data for {self.year}...")

        try:
            draft_data = {}
            for pick in self.league.draft:
                try:
                    if hasattr(pick, 'playerId'):
                        player_id = pick.playerId
                    elif hasattr(pick, 'playerPoolEntry') and pick.playerPoolEntry:
                        player_id = pick.playerPoolEntry['id']
                    else:
                        continue
                    draft_data[player_id] = {
                        'auction_value': pick.bid_amount if hasattr(pick, 'bid_amount') else 0,
                        'draft_position': pick.round_num if hasattr(pick, 'round_num') else 0,
                    }
                except AttributeError:
                    continue

            players = []
            for team in self.league.teams:
                for player in team.roster:
                    try:
                        players.append({
                            'player_id': player.playerId,
                            'name': player.name,
                            'position': player.position,
                            'projected_points': player.projected_total_points if hasattr(player, 'projected_total_points') else 0,
                            'auction_value': draft_data.get(player.playerId, {}).get('auction_value', 0),
                            'adp': draft_data.get(player.playerId, {}).get('draft_position', 0),
                            'team': player.proTeam if hasattr(player, 'proTeam') else None,
                            'status': player.injuryStatus if hasattr(player, 'injuryStatus') else None,
                        })
                    except AttributeError as e:
                        logger.debug(f"Skipping player due to missing attribute: {e}")
                        continue

            positions = ['QB', 'RB', 'WR', 'TE', 'K', 'D/ST']
            for pos in positions:
                try:
                    free_agents = self.league.free_agents(position=pos)
                    for player in free_agents:
                        try:
                            players.append({
                                'player_id': player.playerId,
                                'name': player.name,
                                'position': player.position,
                                'projected_points': player.projected_total_points if hasattr(player, 'projected_total_points') else 0,
                                'auction_value': draft_data.get(player.playerId, {}).get('auction_value', 0),
                                'adp': draft_data.get(player.playerId, {}).get('draft_position', 0),
                                'team': player.proTeam if hasattr(player, 'proTeam') else None,
                                'status': player.injuryStatus if hasattr(player, 'injuryStatus') else None,
                            })
                        except AttributeError as e:
                            logger.debug(f"Skipping free agent: {e}")
                            continue
                except Exception as e:
                    logger.debug(f"Error getting free agents for position {pos}: {e}")
                    continue

            if not players:
                logger.warning("No players found in league data")
                return None

            logger.info(f"Successfully extracted {len(players)} players")
            df = pd.DataFrame(players)
            df = df.sort_values('projected_points', ascending=False)
            df = df.drop_duplicates(subset='player_id', keep='first')
            return df

        except Exception as e:
            logger.error(f"Error fetching pre-draft data: {str(e)}")
            logger.exception("Stack trace:")
            return None

    def get_weekly_stats(self) -> pd.DataFrame:
        """Get weekly player statistics for entire season."""
        if self.year < 2019:
            logger.info(f"Weekly stats not available before 2019. Returning empty DataFrame for {self.year}")
            return pd.DataFrame(columns=['week', 'player_id', 'player_name', 'position', 'pro_team', 'points', 'projected_points', 'stats'])

        weekly_stats = []
        max_week = 18 if self.year >= 2021 else 17
        logger.info(f"Fetching weekly stats for {self.year} season...")

        for week in range(1, max_week + 1):
            try:
                box_scores = self.league.box_scores(week)
                for game in box_scores:
                    for player in game.home_lineup + game.away_lineup:
                        weekly_stats.append({
                            'week': week,
                            'player_id': player.playerId,
                            'player_name': player.name,
                            'position': player.position,
                            'pro_team': player.proTeam,
                            'points': player.points,
                            'projected_points': player.projected_points,
                            'stats': player.stats,
                        })
                logger.debug(f"Processed week {week}")
            except Exception as e:
                logger.warning(f"Error fetching week {week}: {e}")
                continue

        return pd.DataFrame(weekly_stats)

    def load_or_fetch_data(self, year: int, force: bool = False) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
        """Load data from cache if available, otherwise fetch from ESPN."""
        files_exist = all(
            (self.raw_data_dir / f).exists() for f in [
                f"draft_results_{self.league_id}_{year}.csv",
                f"player_stats_{self.league_id}_{year}.csv",
                f"weekly_stats_{self.league_id}_{year}.csv",
                f"league_settings_{self.league_id}_{year}.json",
                f"predraft_values_{self.league_id}_{year}.csv",
            ]
        )

        logger.info(f"Initializing league for year {year}")

        if files_exist and not force:
            logger.info(f"Loading cached data for league {self.league_id}, year {year}")
            draft_df = pd.read_csv(self.raw_data_dir / f"draft_results_{self.league_id}_{year}.csv")
            stats_df = pd.read_csv(self.raw_data_dir / f"player_stats_{self.league_id}_{year}.csv")
            weekly_df = pd.read_csv(self.raw_data_dir / f"weekly_stats_{self.league_id}_{year}.csv")
            with open(self.raw_data_dir / f"league_settings_{self.league_id}_{year}.json") as f:
                settings = json.load(f)
            predraft_df = pd.read_csv(self.raw_data_dir / f"predraft_values_{self.league_id}_{year}.csv")
        else:
            logger.info(f"Fetching data from ESPN for league {self.league_id}, year {year}")
            self.set_year(year)

            draft_df = self.get_draft_results()
            stats_df = self.get_player_stats()
            weekly_df = self.get_weekly_stats()
            settings = self.get_league_settings()
            predraft_df = self.get_predraft_player_data()

            stats_df['year'] = year
            if 'total_points' in stats_df.columns:
                stats_df = stats_df[stats_df['total_points'].notna()]

            self.save_data(draft_df, stats_df, weekly_df, predraft_df, settings, self.league_id, year)

        return draft_df, stats_df, weekly_df, predraft_df, settings

    def save_data(self, draft_df, stats_df, weekly_df, predraft_df, settings, league_id, year):
        """Save all data to cache files."""
        draft_df.to_csv(self.raw_data_dir / f"draft_results_{league_id}_{year}.csv", index=False)
        stats_df.to_csv(self.raw_data_dir / f"player_stats_{league_id}_{year}.csv", index=False)
        weekly_df.to_csv(self.raw_data_dir / f"weekly_stats_{league_id}_{year}.csv", index=False)
        predraft_df.to_csv(self.raw_data_dir / f"predraft_values_{league_id}_{year}.csv", index=False)
        with open(self.raw_data_dir / f"league_settings_{league_id}_{year}.json", 'w') as f:
            json.dump(settings, f, indent=2)
        logger.info(f"Saved data files for {league_id} / {year}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--force", action="store_true", help="Force fetch new data")
    parser.add_argument("--years", nargs="+", type=int, default=list(range(2009, 2025)))
    args = parser.parse_args()

    scraper = ESPNDraftScraper()
    for year in args.years:
        logger.info(f"Fetching {year} data...")
        try:
            scraper.load_or_fetch_data(year, force=args.force)
            logger.info(f"Successfully fetched data for year {year}")
        except ESPNInvalidLeague as e:
            logger.error(f"Error fetching data for year {year}: {e}")
            continue


if __name__ == "__main__":
    main()
