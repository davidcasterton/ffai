"""
Compute per-(position, year) strategic signals from ESPN draft history.

Output columns (one row per position+year):
    position, year,
    avg_bid, median_bid, top5_avg_bid,
    roi_mean              = mean(total_points / bid_amount)
    proj_accuracy_ratio   = mean(total_points / projected_points)
    budget_share_pct      = position_total_bid / league_total_bid
    winning_budget_share  = avg budget share for top-4 teams at this position
"""

import logging
from pathlib import Path

import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)

ALL_SEASONS = list(range(2009, 2025))
POSITIONS = ["QB", "RB", "WR", "TE", "K", "D/ST"]


def _default_espn_data_dir_and_id():
    from ffai.data.espn_scraper import load_league_config
    cfg = load_league_config()
    league_name = cfg["league"]["league_name"]
    league_id = cfg["league"]["league_id"]
    data_dir = Path(__file__).parent.parent / league_name
    return data_dir, league_id


def _load_year(year: int, data_dir: Path, league_id: str) -> pd.DataFrame | None:
    dr_path = data_dir / f"draft_results_{league_id}_{year}.csv"
    ps_path = data_dir / f"player_stats_{league_id}_{year}.csv"
    pv_path = data_dir / f"predraft_values_{league_id}_{year}.csv"

    if not dr_path.exists() or not ps_path.exists() or not pv_path.exists():
        return None

    dr = pd.read_csv(dr_path)
    ps = pd.read_csv(ps_path)[["player_id", "total_points"]]
    pv = pd.read_csv(pv_path)[["player_id", "position", "projected_points"]]

    for df in [dr, ps, pv]:
        df["player_id"] = df["player_id"].astype(str)

    merged = dr.merge(pv, on="player_id", how="left").merge(ps, on="player_id", how="left")
    merged["year"] = year
    merged["total_points"] = merged["total_points"].fillna(0.0)
    merged["projected_points"] = merged["projected_points"].fillna(0.0)
    return merged


def build_position_strategy(
    years: list[int] | None = None,
    data_dir: Path = None,
    league_id: str = None,
) -> pd.DataFrame:
    """
    Build per-(position, year) strategic signal DataFrame.
    """
    if data_dir is None or league_id is None:
        _dir, _id = _default_espn_data_dir_and_id()
        data_dir = data_dir or _dir
        league_id = league_id or _id
    years = years or ALL_SEASONS
    records = []

    for year in years:
        df = _load_year(year, data_dir, league_id)
        if df is None:
            continue

        league_total_bid = df["bid_amount"].sum()
        if league_total_bid == 0:
            continue

        # Team standings: rank teams by total_points of drafted players
        team_points = (
            df.groupby("team_name")["total_points"].sum()
            .sort_values(ascending=False)
        )
        top4_teams = set(team_points.head(4).index.tolist())

        for pos in POSITIONS:
            pos_df = df[df["position"] == pos]
            if pos_df.empty:
                continue

            pos_bid = pos_df["bid_amount"]
            pos_pts = pos_df["total_points"]
            pos_proj = pos_df["projected_points"]

            # ROI: points per dollar (skip $1 buys to reduce noise)
            roi_mask = pos_bid > 1
            roi_mean = float((pos_pts[roi_mask] / pos_bid[roi_mask]).mean()) if roi_mask.any() else np.nan

            # Projection accuracy
            proj_mask = pos_proj > 0
            proj_accuracy = float((pos_pts[proj_mask] / pos_proj[proj_mask]).mean()) if proj_mask.any() else np.nan

            # Winning budget share: avg position budget share for top-4 teams
            top4_shares = []
            for team in top4_teams:
                team_df = df[df["team_name"] == team]
                team_total = team_df["bid_amount"].sum()
                team_pos = team_df[team_df["position"] == pos]["bid_amount"].sum()
                if team_total > 0:
                    top4_shares.append(float(team_pos / team_total))
            winning_budget_share = float(np.mean(top4_shares)) if top4_shares else np.nan

            records.append({
                "position": pos,
                "year": year,
                "avg_bid": float(pos_bid.mean()),
                "median_bid": float(pos_bid.median()),
                "top5_avg_bid": float(pos_bid.nlargest(5).mean()),
                "roi_mean": roi_mean,
                "proj_accuracy_ratio": proj_accuracy,
                "budget_share_pct": float(pos_bid.sum() / league_total_bid),
                "winning_budget_share": winning_budget_share,
            })

    return pd.DataFrame(records)
