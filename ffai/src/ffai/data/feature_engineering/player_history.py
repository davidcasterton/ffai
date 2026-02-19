"""
Build multi-year lookback features per (player_id, year) from ESPN data.

All features are lookback-safe: they only use data from years < target_year.
Rookies get NaN for all lookback features.

Output columns (per player, per year):
    pts_3yr_avg        mean(total_points, Y-1 to Y-3)
    pts_3yr_std        std(total_points, Y-1 to Y-3)
    pts_1yr_val        total_points[Y-1]
    yoy_pct_change     (pts[Y-1] - pts[Y-2]) / pts[Y-2], clipped [-1, 1]
    years_in_league    count of prior seasons with data
    proj_ratio_3yr_avg mean(total_points[Y-k] / projected_points[Y-k], k=1..3)
    proj_bias_1yr      total_points[Y-1] - projected_points[Y-1]
"""

import logging
from pathlib import Path

import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)

FAVREFIGNEWTON_DIR = Path(__file__).parent.parent / "favrefignewton"
LEAGUE_ID = "770280"
ALL_SEASONS = list(range(2009, 2025))


def load_espn_season_data(
    years: list[int] | None = None,
    data_dir: Path = FAVREFIGNEWTON_DIR,
) -> pd.DataFrame:
    """
    Load and concatenate ESPN draft_results + player_stats + predraft_values
    for all years, returning a tidy per-(player_id, year) DataFrame with:
        player_id, year, total_points, projected_points, position
    """
    years = years or ALL_SEASONS
    frames = []

    for year in years:
        dr_path = data_dir / f"draft_results_{LEAGUE_ID}_{year}.csv"
        ps_path = data_dir / f"player_stats_{LEAGUE_ID}_{year}.csv"
        pv_path = data_dir / f"predraft_values_{LEAGUE_ID}_{year}.csv"

        if not dr_path.exists() or not ps_path.exists() or not pv_path.exists():
            logger.warning(f"Missing ESPN data for year {year}, skipping")
            continue

        dr = pd.read_csv(dr_path)[["player_id", "bid_amount"]]
        ps = pd.read_csv(ps_path)[["player_id", "total_points"]]
        pv = pd.read_csv(pv_path)[["player_id", "position", "projected_points"]]

        dr["player_id"] = dr["player_id"].astype(str)
        ps["player_id"] = ps["player_id"].astype(str)
        pv["player_id"] = pv["player_id"].astype(str)

        merged = pv.merge(ps, on="player_id", how="left").merge(dr, on="player_id", how="left")
        merged["year"] = year
        merged["total_points"] = merged["total_points"].fillna(0.0)
        merged["projected_points"] = merged["projected_points"].fillna(0.0)

        frames.append(merged[["player_id", "year", "position", "total_points", "projected_points"]])

    if not frames:
        return pd.DataFrame(columns=["player_id", "year", "position", "total_points", "projected_points"])

    return pd.concat(frames, ignore_index=True)


def build_player_history(
    season_df: pd.DataFrame,
    target_years: list[int] | None = None,
) -> pd.DataFrame:
    """
    From a tidy per-(player_id, year) season DataFrame, compute lookback features
    for each (player_id, target_year). Returns one row per (player_id, target_year).

    Args:
        season_df: DataFrame with columns [player_id, year, total_points, projected_points]
        target_years: years to compute features for (defaults to all years in season_df)

    Returns:
        DataFrame with columns:
            player_id, year, pts_3yr_avg, pts_3yr_std, pts_1yr_val,
            yoy_pct_change, years_in_league, proj_ratio_3yr_avg, proj_bias_1yr
    """
    if target_years is None:
        target_years = sorted(season_df["year"].unique().tolist())

    # Index historical data per player for fast lookups
    history = season_df.set_index(["player_id", "year"])

    records = []
    for year in target_years:
        # Players who appear in this target year's data
        year_players = season_df[season_df["year"] == year]["player_id"].unique()

        for pid in year_players:
            # Lookback years: Y-1, Y-2, Y-3
            y1, y2, y3 = year - 1, year - 2, year - 3

            def pts(y):
                try:
                    return float(history.at[(pid, y), "total_points"])
                except KeyError:
                    return np.nan

            def proj(y):
                try:
                    p = float(history.at[(pid, y), "projected_points"])
                    return p if p > 0 else np.nan
                except KeyError:
                    return np.nan

            p1, p2, p3 = pts(y1), pts(y2), pts(y3)
            pr1, pr2, pr3 = proj(y1), proj(y2), proj(y3)

            available = [v for v in [p1, p2, p3] if not np.isnan(v)]
            years_in_league = len(available)

            pts_3yr_avg = float(np.mean(available)) if available else np.nan
            pts_3yr_std = float(np.std(available)) if len(available) >= 2 else np.nan
            pts_1yr_val = p1  # nan if no prior year

            if not np.isnan(p1) and not np.isnan(p2) and p2 > 0:
                yoy_pct_change = float(np.clip((p1 - p2) / p2, -1.0, 1.0))
            else:
                yoy_pct_change = np.nan

            # Projection ratio: actual / projected for each lookback year
            ratios = []
            for actual, projected in [(p1, pr1), (p2, pr2), (p3, pr3)]:
                if not np.isnan(actual) and not np.isnan(projected) and projected > 0:
                    ratios.append(actual / projected)
            proj_ratio_3yr_avg = float(np.mean(ratios)) if ratios else np.nan

            proj_bias_1yr = float(p1 - pr1) if (not np.isnan(p1) and not np.isnan(pr1)) else np.nan

            records.append({
                "player_id": pid,
                "year": year,
                "pts_3yr_avg": pts_3yr_avg,
                "pts_3yr_std": pts_3yr_std,
                "pts_1yr_val": pts_1yr_val,
                "yoy_pct_change": yoy_pct_change,
                "years_in_league": float(years_in_league),
                "proj_ratio_3yr_avg": proj_ratio_3yr_avg,
                "proj_bias_1yr": proj_bias_1yr,
            })

    return pd.DataFrame(records)
