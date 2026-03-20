#!/usr/bin/env python3
"""
Build all processed feature CSVs from ESPN league data + nflverse data.

Writes to ffai/src/ffai/data/{league_name}_processed/:
  player_history_{league_id}.csv     — per-(player_id, year) lookback features
  manager_tendencies_{league_id}.csv — per-manager bidding profile
  position_strategy_{league_id}.csv  — per-(position, year) strategic signals

League name and ID are read from config/league.yaml.

Budget normalization: before any feature engineering runs, bid_amount values
are scaled to $200-equivalent using budget_by_year.csv (written by
sanity_check_data.py). If that file does not exist, budgets are detected
inline from each year's median team spend.

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


def _load_budget_map() -> dict[int, int]:
    """Load per-year detected budgets from budget_by_year.csv if available."""
    budget_path = PROCESSED_DIR / "budget_by_year.csv"
    if budget_path.exists():
        bdf = pd.read_csv(budget_path)
        return {int(row["year"]): int(row["detected_budget"]) for _, row in bdf.iterrows()}
    return {}


def _detect_budget_from_df(draft_df: pd.DataFrame) -> int:
    """Inline budget detection when budget_by_year.csv is absent."""
    team_spend = draft_df.groupby("manager_id")["bid_amount"].sum()
    return 1000 if team_spend.median() > 300 else 200


def normalize_draft_bids(years: list[int], league_id: str, budget_map: dict[int, int]) -> dict[int, pd.DataFrame]:
    """
    Load each year's draft_results CSV and normalize bid_amount to $200-equivalent.

    Returns a dict mapping year → normalized DataFrame.
    """
    normalized: dict[int, pd.DataFrame] = {}
    for year in years:
        dr_path = FAVREFIGNEWTON_DIR / f"draft_results_{league_id}_{year}.csv"
        if not dr_path.exists():
            continue
        df = pd.read_csv(dr_path)
        budget = budget_map.get(year) or _detect_budget_from_df(df)
        if budget != 200:
            norm_factor = 200.0 / budget
            df["bid_amount"] = (df["bid_amount"] * norm_factor).round(2)
            print(f"    Year {year}: detected budget=${budget}, normalized bids by {norm_factor:.3f}×")
        normalized[year] = df
    return normalized


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

    # --- Budget normalization: load or detect per-year budgets, normalize bid_amounts ---
    print("\n[0/3] Normalizing bid amounts to $200-equivalent...")
    budget_map = _load_budget_map()
    normalized_dfs = normalize_draft_bids(years, lid, budget_map)
    non_200_years = [
        y for y in normalized_dfs
        if (budget_map.get(y) or _detect_budget_from_df(
            pd.read_csv(FAVREFIGNEWTON_DIR / f"draft_results_{lid}_{y}.csv")
        )) != 200
    ]
    if non_200_years:
        print(f"  Normalized {len(non_200_years)} non-$200 year(s): {non_200_years}")
    else:
        print("  All years have $200 budgets — no normalization needed.")

    # Build normalized combined draft_df (manager_tendencies and position_strategy use it)
    normalized_frames = list(normalized_dfs.values())

    # --- Player history (bid_amount not in output — no normalization needed) ---
    print("\n[1/3] player_history...")
    history = build_player_history(years, lid)
    out = PROCESSED_DIR / f"player_history_{lid}.csv"
    history.to_csv(out, index=False)
    print(f"  Wrote {len(history):,} rows → {out}")

    # --- Manager tendencies (pass normalized draft_df directly) ---
    print("\n[2/3] manager_tendencies...")
    from ffai.data.feature_engineering.manager_tendencies import build_manager_tendencies
    if normalized_frames:
        # Enrich with predraft projections (needed for bid_per_proj_pt)
        enriched_frames = []
        for year, norm_df in normalized_dfs.items():
            pv_path = FAVREFIGNEWTON_DIR / f"predraft_values_{lid}_{year}.csv"
            if pv_path.exists():
                pv = pd.read_csv(pv_path)[["player_id", "position", "projected_points"]]
                pv["player_id"] = pv["player_id"].astype(str)
                norm_df = norm_df.copy()
                norm_df["player_id"] = norm_df["player_id"].astype(str)
                merged = norm_df.merge(pv, on="player_id", how="left")
                merged["year"] = year
                enriched_frames.append(merged)
        if enriched_frames:
            combined_draft_df = pd.concat(enriched_frames, ignore_index=True)
            tendencies = build_manager_tendencies(draft_df=combined_draft_df)
        else:
            tendencies = build_manager_tendencies(years=years, data_dir=FAVREFIGNEWTON_DIR)
    else:
        tendencies = build_manager_tendencies(years=years, data_dir=FAVREFIGNEWTON_DIR)
    out = PROCESSED_DIR / f"manager_tendencies_{lid}.csv"
    tendencies.to_csv(out, index=False)
    print(f"  Wrote {len(tendencies):,} rows → {out}")

    # --- Position strategy (pass normalized draft_df directly) ---
    print("\n[3/3] position_strategy...")
    from ffai.data.feature_engineering.position_strategy import build_position_strategy
    if normalized_frames:
        # Build per-year DataFrames with stats merged in (position_strategy needs total_points)
        ps_frames = []
        for year, norm_df in normalized_dfs.items():
            ps_path = FAVREFIGNEWTON_DIR / f"player_stats_{lid}_{year}.csv"
            pv_path = FAVREFIGNEWTON_DIR / f"predraft_values_{lid}_{year}.csv"
            if not ps_path.exists() or not pv_path.exists():
                continue
            ps = pd.read_csv(ps_path)[["player_id", "total_points"]]
            pv = pd.read_csv(pv_path)[["player_id", "position", "projected_points"]]
            for df in [ps, pv]:
                df["player_id"] = df["player_id"].astype(str)
            merged = norm_df.copy()
            merged["player_id"] = merged["player_id"].astype(str)
            # Drop position/projected_points columns from norm_df if present to avoid conflicts
            for col in ["position", "projected_points", "total_points"]:
                if col in merged.columns:
                    merged = merged.drop(columns=[col])
            merged = merged.merge(pv, on="player_id", how="left").merge(ps, on="player_id", how="left")
            merged["year"] = year
            merged["total_points"] = merged["total_points"].fillna(0.0)
            merged["projected_points"] = merged["projected_points"].fillna(0.0)
            ps_frames.append(merged)
        if ps_frames:
            combined_ps_df = pd.concat(ps_frames, ignore_index=True)
            strategy = build_position_strategy(draft_df=combined_ps_df)
        else:
            strategy = build_position_strategy(years=years, data_dir=FAVREFIGNEWTON_DIR)
    else:
        strategy = build_position_strategy(years=years, data_dir=FAVREFIGNEWTON_DIR)
    out = PROCESSED_DIR / f"position_strategy_{lid}.csv"
    strategy.to_csv(out, index=False)
    print(f"  Wrote {len(strategy):,} rows → {out}")

    print("\nDone. All features written to", PROCESSED_DIR)


if __name__ == "__main__":
    main()
