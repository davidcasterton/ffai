"""
Parse ESPN weekly stats CSVs (2019+) to extract per-player weekly fantasy points.

The 'stats' column in weekly_stats_770280_{year}.csv is a JSON string with nested
structure. This module flattens it into a clean DataFrame of
(player_id, year, week, actual_points) for use in consistency features.
"""

import json
import logging
from pathlib import Path

import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)

WEEKLY_SEASONS = list(range(2019, 2025))


def _default_espn_data_dir_and_id():
    from ffai.data.espn_scraper import load_league_config
    cfg = load_league_config()
    league_name = cfg["league"]["league_name"]
    league_id = cfg["league"]["league_id"]
    data_dir = Path(__file__).parent.parent / league_name
    return data_dir, league_id


def load_weekly_fantasy_points(
    years: list[int] | None = None,
    data_dir: Path = None,
    league_id: str = None,
) -> pd.DataFrame:
    """
    Load ESPN weekly stats and return a tidy DataFrame:
        player_id (str), year (int), week (int), actual_points (float)

    Only loads years >= 2019 (earlier files exist but JSON stats field is sparse).
    """
    if data_dir is None or league_id is None:
        _dir, _id = _default_espn_data_dir_and_id()
        data_dir = data_dir or _dir
        league_id = league_id or _id
    years = years or WEEKLY_SEASONS
    years = [y for y in years if y >= 2019]

    frames = []
    for year in years:
        path = data_dir / f"weekly_stats_{league_id}_{year}.csv"
        if not path.exists():
            logger.warning(f"Missing weekly stats file: {path}")
            continue

        df = pd.read_csv(path)
        df["player_id"] = df["player_id"].astype(str)
        df["year"] = year

        frames.append(df[["player_id", "year", "week", "points"]].copy())

    if not frames:
        return pd.DataFrame(columns=["player_id", "year", "week", "actual_points"])

    result = pd.concat(frames, ignore_index=True)
    result = result.rename(columns={"points": "actual_points"})
    result["actual_points"] = pd.to_numeric(result["actual_points"], errors="coerce").fillna(0.0)
    return result


def compute_weekly_consistency(
    weekly_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    From a tidy (player_id, year, week, actual_points) DataFrame, compute
    per-(player_id, year) consistency features using only prior-year weekly data.

    Returns DataFrame with columns:
        player_id, year, weekly_pts_cv, floor_pts, ceiling_pts, weeks_active
    """
    records = []

    for (pid, year), grp in weekly_df.groupby(["player_id", "year"]):
        pts = grp["actual_points"].values
        active = pts[pts > 0]

        cv = float(np.std(active) / np.mean(active)) if len(active) >= 2 and np.mean(active) > 0 else np.nan
        floor = float(np.percentile(active, 25)) if len(active) >= 4 else np.nan
        ceiling = float(np.percentile(active, 75)) if len(active) >= 4 else np.nan
        weeks = int(len(active))

        records.append({
            "player_id": pid,
            "year": year,
            "weekly_pts_cv": cv,
            "floor_pts": floor,
            "ceiling_pts": ceiling,
            "weeks_active": weeks,
        })

    if not records:
        return pd.DataFrame(columns=["player_id", "year", "weekly_pts_cv", "floor_pts", "ceiling_pts", "weeks_active"])

    return pd.DataFrame(records)
