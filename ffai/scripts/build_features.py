#!/usr/bin/env python3
"""
Build all processed feature CSVs from ESPN league data + nflverse data.

Writes to ffai/src/ffai/data/{league_name}_processed/:
  player_history_{league_id}.csv     — per-(player_id, year) lookback features
  manager_tendencies_{league_id}.csv — per-manager bidding profile
  position_strategy_{league_id}.csv  — per-(position, year) strategic signals

League name and ID are read from config/league.yaml.

Usage:
    python scripts/build_features.py [--league-id ID] [--years 2009-2024]
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import pandas as pd
from ffai.data.espn_scraper import load_league_config

_cfg = load_league_config()
_league_name = _cfg["league"]["league_name"]
_league_id = _cfg["league"]["league_id"]

PROCESSED_DIR = Path(__file__).parent.parent / f"src/ffai/data/{_league_name}_processed"
FAVREFIGNEWTON_DIR = Path(__file__).parent.parent / f"src/ffai/data/{_league_name}"
NFLVERSE_DIR = Path(__file__).parent.parent / "src/ffai/data/nflverse"


def build_player_history(years: list[int], league_id: str) -> pd.DataFrame:
    from ffai.data.feature_engineering.player_history import load_espn_season_data, build_player_history as _build
    from ffai.data.feature_engineering.weekly_stats_parser import load_weekly_fantasy_points, compute_weekly_consistency
    from ffai.data.feature_engineering.nfl_feature_builder import build_nfl_features

    print("  Loading ESPN season data...")
    season_df = load_espn_season_data(years=years, data_dir=FAVREFIGNEWTON_DIR, league_id=league_id)

    print("  Building multi-year lookback features...")
    history = _build(season_df, target_years=years)

    print("  Computing weekly consistency features (ESPN, 2019+)...")
    weekly_raw = load_weekly_fantasy_points(years=years, data_dir=FAVREFIGNEWTON_DIR, league_id=league_id)
    # weekly_raw is per prior year; shift: we want Y-1 consistency for a player in year Y
    weekly_raw["lookback_year"] = weekly_raw["year"] + 1
    consistency = compute_weekly_consistency(weekly_raw)
    consistency = consistency.rename(columns={"year": "data_year"})
    consistency["year"] = consistency["data_year"] + 1  # features available for next year

    history = history.merge(
        consistency[["player_id", "year", "weekly_pts_cv", "floor_pts", "ceiling_pts", "weeks_active"]],
        on=["player_id", "year"], how="left"
    )

    print("  Building nflverse usage + demographic features...")
    nfl_feats = build_nfl_features(target_years=years, nflverse_dir=NFLVERSE_DIR)
    history = history.merge(nfl_feats, on=["player_id", "year"], how="left")

    # Attach position from season_df for imputation reference
    pos_map = season_df.groupby("player_id")["position"].first().reset_index()
    history = history.merge(pos_map, on="player_id", how="left")

    return history


def main() -> None:
    parser = argparse.ArgumentParser(description="Build processed feature CSVs for ffai")
    parser.add_argument("--league-id", default=_league_id)
    parser.add_argument("--years", default="2009-2024")
    args = parser.parse_args()

    raw = args.years
    if "-" in raw and "," not in raw:
        start, end = raw.split("-")
        years = list(range(int(start), int(end) + 1))
    else:
        years = [int(s.strip()) for s in raw.split(",")]

    lid = args.league_id
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Building features for league {lid}, years {years[0]}–{years[-1]}")

    # --- Player history ---
    print("\n[1/3] player_history...")
    history = build_player_history(years, lid)
    out = PROCESSED_DIR / f"player_history_{lid}.csv"
    history.to_csv(out, index=False)
    print(f"  Wrote {len(history):,} rows → {out}")

    # --- Manager tendencies ---
    print("\n[2/3] manager_tendencies...")
    from ffai.data.feature_engineering.manager_tendencies import build_manager_tendencies
    tendencies = build_manager_tendencies(years=years, data_dir=FAVREFIGNEWTON_DIR)
    out = PROCESSED_DIR / f"manager_tendencies_{lid}.csv"
    tendencies.to_csv(out, index=False)
    print(f"  Wrote {len(tendencies):,} rows → {out}")

    # --- Position strategy ---
    print("\n[3/3] position_strategy...")
    from ffai.data.feature_engineering.position_strategy import build_position_strategy
    strategy = build_position_strategy(years=years, data_dir=FAVREFIGNEWTON_DIR)
    out = PROCESSED_DIR / f"position_strategy_{lid}.csv"
    strategy.to_csv(out, index=False)
    print(f"  Wrote {len(strategy):,} rows → {out}")

    print("\nDone. All features written to", PROCESSED_DIR)


if __name__ == "__main__":
    main()
