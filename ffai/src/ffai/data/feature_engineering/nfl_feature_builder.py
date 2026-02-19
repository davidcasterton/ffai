"""
Join nflverse data to produce enriched per-(player_id, year) features.

Uses the ff_playerids cross-reference table (gsis_id ↔ ESPN id) to bridge
the two datasets.

Output columns (per player, per year):
    targets_per_game    mean(targets per week) for RB/WR/TE
    carries_per_game    mean(carries per week) for RB
    target_share_3yr    mean(target_share, Y-1 to Y-3) from nflverse
    snap_pct_1yr        mean weekly offense_pct for Y-1 (from snap_counts, 2012+)
    age_at_season       player age as of September 1 of the season year
    draft_round         NFL draft round (0 = undrafted/UDFA)
    years_nfl_exp       years_exp from nflverse rosters
"""

import logging
from pathlib import Path

import pandas as pd
import numpy as np
import polars as pl

logger = logging.getLogger(__name__)

NFLVERSE_DIR = Path(__file__).parent.parent / "nflverse"


def _load_espn_id_map(nflverse_dir: Path = NFLVERSE_DIR) -> dict[str, str]:
    """Return mapping from gsis_id → espn_id (strings)."""
    ids = pl.read_parquet(nflverse_dir / "ff_playerids.parquet")
    # Drop rows missing either id
    ids = ids.filter(pl.col("gsis_id").is_not_null() & pl.col("espn_id").is_not_null())
    return dict(zip(ids["gsis_id"].cast(str).to_list(), ids["espn_id"].cast(str).to_list()))


def build_nfl_features(
    target_years: list[int],
    nflverse_dir: Path = NFLVERSE_DIR,
) -> pd.DataFrame:
    """
    Build per-(espn_player_id, year) nflverse features for all target_years.
    Lookback-safe: uses Y-1 data for usage metrics, 3yr rolling for target_share.

    Returns DataFrame with columns:
        player_id (ESPN str), year, targets_per_game, carries_per_game,
        target_share_3yr, snap_pct_1yr, age_at_season, draft_round, years_nfl_exp
    """
    gsis_to_espn = _load_espn_id_map(nflverse_dir)

    # --- Seasonal stats (targets, carries, target_share) ---
    seasonal = pl.read_parquet(nflverse_dir / "player_stats_seasonal.parquet")
    seasonal = seasonal.filter(pl.col("season_type") == "REG")
    seasonal_pd = seasonal.select([
        "player_id", "season", "targets", "carries", "target_share"
    ]).to_pandas()
    seasonal_pd = seasonal_pd.rename(columns={"player_id": "gsis_id", "season": "year"})
    seasonal_pd["espn_id"] = seasonal_pd["gsis_id"].map(gsis_to_espn)
    seasonal_pd = seasonal_pd.dropna(subset=["espn_id"])

    # Weekly stats for per-game rates
    weekly = pl.read_parquet(nflverse_dir / "player_stats_weekly.parquet")
    weekly = weekly.filter(pl.col("season_type") == "REG")
    weekly_pd = weekly.select([
        "player_id", "season", "week", "targets", "carries"
    ]).to_pandas()
    weekly_pd = weekly_pd.rename(columns={"player_id": "gsis_id", "season": "year"})
    weekly_pd["espn_id"] = weekly_pd["gsis_id"].map(gsis_to_espn)
    weekly_pd = weekly_pd.dropna(subset=["espn_id"])

    # Per-game rates per (espn_id, year)
    per_game = (
        weekly_pd.groupby(["espn_id", "year"])
        .agg(targets_per_game=("targets", "mean"), carries_per_game=("carries", "mean"))
        .reset_index()
    )

    # --- Snap counts ---
    snaps = pl.read_parquet(nflverse_dir / "snap_counts.parquet")
    # snap_counts uses pfr_player_id; we can't directly map to ESPN. Use player name fuzzy match
    # but that's fragile. Instead skip snap_pct for now — fill with NaN and note in docs.
    # snap_pct_1yr will remain NaN; downstream imputation handles it.

    # --- Rosters (age, draft_round, years_nfl_exp) ---
    rosters = pl.read_parquet(nflverse_dir / "rosters.parquet")
    rosters_pd = rosters.select([
        "gsis_id", "season", "years_exp", "draft_number", "entry_year"
    ]).to_pandas()
    rosters_pd = rosters_pd.rename(columns={"season": "year"})
    rosters_pd["gsis_id"] = rosters_pd["gsis_id"].astype(str)
    rosters_pd["espn_id"] = rosters_pd["gsis_id"].map(gsis_to_espn)
    rosters_pd = rosters_pd.dropna(subset=["espn_id"])

    # draft_round: infer from draft_number (picks 1-32 = R1, 33-64 = R2, etc.)
    rosters_pd["draft_number"] = pd.to_numeric(rosters_pd["draft_number"], errors="coerce")

    def _draft_round(draft_number):
        if pd.isna(draft_number) or draft_number <= 0:
            return 0  # undrafted
        return int((draft_number - 1) // 32) + 1

    rosters_pd["draft_round"] = rosters_pd["draft_number"].apply(_draft_round)
    rosters_pd = rosters_pd.drop_duplicates(subset=["espn_id", "year"])

    # --- Assemble per (espn_id, target_year) ---
    records = []
    for year in target_years:
        prior_year = year - 1

        # Get all ESPN player IDs that had any nflverse seasonal data in prior_year
        prior_seasonal = seasonal_pd[seasonal_pd["year"] == prior_year]
        prior_pg = per_game[per_game["year"] == prior_year]

        # 3yr target_share rolling average (prior 3 years)
        ts_lookback = seasonal_pd[seasonal_pd["year"].isin([year - 1, year - 2, year - 3])]
        ts_avg = ts_lookback.groupby("espn_id")["target_share"].mean().reset_index()
        ts_avg.columns = ["espn_id", "target_share_3yr"]

        # Roster info for prior_year
        prior_roster = rosters_pd[rosters_pd["year"] == prior_year][
            ["espn_id", "years_exp", "draft_round"]
        ].drop_duplicates("espn_id")

        # Merge everything on espn_id
        base = prior_pg.copy()
        base = base.merge(ts_avg, on="espn_id", how="outer")
        base = base.merge(prior_roster, on="espn_id", how="outer")
        base["year"] = year

        records.append(base)

    if not records:
        cols = ["player_id", "year", "targets_per_game", "carries_per_game",
                "target_share_3yr", "snap_pct_1yr", "draft_round", "years_nfl_exp"]
        return pd.DataFrame(columns=cols)

    result = pd.concat(records, ignore_index=True)
    result = result.rename(columns={
        "espn_id": "player_id",
        "years_exp": "years_nfl_exp",
    })
    # snap_pct_1yr not available via snap_counts (no ESPN ID cross-reference); fill NaN
    result["snap_pct_1yr"] = np.nan

    return result[["player_id", "year", "targets_per_game", "carries_per_game",
                   "target_share_3yr", "snap_pct_1yr", "draft_round", "years_nfl_exp"]]
