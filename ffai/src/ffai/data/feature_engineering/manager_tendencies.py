"""
Compute per-manager bidding profile from ESPN draft history.

Avoids using bid_amount directly as a value proxy (since auction_value == bid_amount
in the ESPN data). Instead, computes relative metrics:
  - budget share by position (rb_budget_share, etc.)
  - bid per projected point by position (bid_per_proj_pt_rb, etc.)
  - high_bid_rate: fraction of picks with bid > $30
  - dollar_one_rate: fraction of picks costing $1

Output: one row per manager_id.
"""

import logging
from pathlib import Path

import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)

ALL_SEASONS = list(range(2009, 2025))
POSITIONS = ["QB", "RB", "WR", "TE"]


def _default_espn_data_dir_and_id():
    from ffai.data.espn_scraper import load_league_config
    cfg = load_league_config()
    league_name = cfg["league"]["league_name"]
    league_id = cfg["league"]["league_id"]
    data_dir = Path(__file__).parent.parent / league_name
    return data_dir, league_id


def load_all_draft_data(
    years: list[int] | None = None,
    data_dir: Path = None,
    league_id: str = None,
) -> pd.DataFrame:
    """Load and concatenate draft_results + predraft_values for all years."""
    if data_dir is None or league_id is None:
        _dir, _id = _default_espn_data_dir_and_id()
        data_dir = data_dir or _dir
        league_id = league_id or _id
    years = years or ALL_SEASONS
    frames = []
    for year in years:
        dr_path = data_dir / f"draft_results_{league_id}_{year}.csv"
        pv_path = data_dir / f"predraft_values_{league_id}_{year}.csv"
        if not dr_path.exists() or not pv_path.exists():
            continue

        dr = pd.read_csv(dr_path)
        pv = pd.read_csv(pv_path)[["player_id", "position", "projected_points"]]

        dr["player_id"] = dr["player_id"].astype(str)
        pv["player_id"] = pv["player_id"].astype(str)

        merged = dr.merge(pv, on="player_id", how="left")
        merged["year"] = year
        frames.append(merged)

    if not frames:
        return pd.DataFrame()

    return pd.concat(frames, ignore_index=True)


def build_manager_tendencies(
    draft_df: pd.DataFrame | None = None,
    years: list[int] | None = None,
    data_dir: Path = None,
    league_id: str = None,
) -> pd.DataFrame:
    """
    Compute per-manager bidding tendencies from all available draft years.

    Args:
        draft_df: pre-loaded combined draft DataFrame (optional, loads from disk if None)
        years: years to include (default: all available)
        data_dir: path to favrefignewton data directory

    Returns:
        DataFrame with one row per manager_id and tendency columns.
    """
    if draft_df is None:
        draft_df = load_all_draft_data(years=years, data_dir=data_dir, league_id=league_id)

    if draft_df.empty:
        return pd.DataFrame()

    records = []

    for mgr_id, grp in draft_df.groupby("manager_id"):
        total_bid = grp["bid_amount"].sum()
        total_picks = len(grp)

        if total_bid == 0 or total_picks == 0:
            continue

        row = {
            "manager_id": mgr_id,
            "manager_name": grp["manager_display_name"].iloc[0],
            "seasons_active": grp["year"].nunique(),
        }

        # Budget share and bid efficiency by position
        for pos in POSITIONS:
            pos_grp = grp[grp["position"] == pos]
            pos_bid = pos_grp["bid_amount"].sum()
            row[f"{pos.lower()}_budget_share"] = float(pos_bid / total_bid) if total_bid > 0 else 0.0

            # bid per projected point (where projection > 0)
            valid = pos_grp[pos_grp["projected_points"] > 0].copy()
            if len(valid) > 0:
                row[f"bid_per_proj_pt_{pos.lower()}"] = float(
                    (valid["bid_amount"] / valid["projected_points"]).mean()
                )
            else:
                row[f"bid_per_proj_pt_{pos.lower()}"] = np.nan

        row["high_bid_rate"] = float((grp["bid_amount"] > 30).mean())
        row["dollar_one_rate"] = float((grp["bid_amount"] == 1).mean())

        # Recent 3-season budget shares (for temporal weighting)
        if "year" in grp.columns:
            recent_years = sorted(grp["year"].unique())[-3:]
            recent = grp[grp["year"].isin(recent_years)]
            recent_total = recent["bid_amount"].sum()
            for pos in ["WR", "RB"]:
                pos_recent = recent[recent["position"] == pos]["bid_amount"].sum()
                key = f"recent_{pos.lower()}_share"
                row[key] = float(pos_recent / recent_total) if recent_total > 0 else 0.0

        records.append(row)

    return pd.DataFrame(records)
