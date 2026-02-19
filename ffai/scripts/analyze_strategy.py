#!/usr/bin/env python3
"""
Strategic analysis of 16 years of league draft + season data.

Answers:
  1. Position ROI over 16 years: avg bid, avg points, points-per-dollar by position+year
  2. Manager tendencies: who overbids/underbids by position; budget efficiency ranking
  3. Projection bias: systematic ESPN bias by position (actual / projected)
  4. Winning patterns: WR vs RB budget allocation for top-4 vs bottom-8 teams;
     Spearman correlation of position budget share with final standing

Usage:
    python scripts/analyze_strategy.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import pandas as pd
import numpy as np
from scipy import stats as scipy_stats
from ffai.data.espn_scraper import load_league_config

_cfg = load_league_config()
_league_name = _cfg["league"]["league_name"]
LEAGUE_ID = _cfg["league"]["league_id"]
FAVREFIGNEWTON = Path(__file__).parent.parent / f"src/ffai/data/{_league_name}"
PROCESSED = Path(__file__).parent.parent / f"src/ffai/data/{_league_name}_processed"
POSITIONS = ["QB", "RB", "WR", "TE"]
ALL_YEARS = list(range(2009, 2025))


def load_all_draft_data() -> pd.DataFrame:
    frames = []
    for year in ALL_YEARS:
        dr_path = FAVREFIGNEWTON / f"draft_results_{LEAGUE_ID}_{year}.csv"
        ps_path = FAVREFIGNEWTON / f"player_stats_{LEAGUE_ID}_{year}.csv"
        pv_path = FAVREFIGNEWTON / f"predraft_values_{LEAGUE_ID}_{year}.csv"
        if not (dr_path.exists() and ps_path.exists() and pv_path.exists()):
            continue
        dr = pd.read_csv(dr_path)
        ps = pd.read_csv(ps_path)[["player_id", "total_points"]]
        pv = pd.read_csv(pv_path)[["player_id", "position", "projected_points"]]
        for df in [dr, ps, pv]:
            df["player_id"] = df["player_id"].astype(str)
        merged = dr.merge(pv, on="player_id", how="left").merge(ps, on="player_id", how="left")
        merged["year"] = year
        merged["total_points"] = merged["total_points"].fillna(0.0)
        merged["projected_points"] = merged["projected_points"].fillna(0.0)
        frames.append(merged)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def section(title: str) -> None:
    print(f"\n{'='*70}")
    print(f"  {title}")
    print(f"{'='*70}")


def analyze_position_roi(df: pd.DataFrame) -> None:
    section("1. Position ROI (all years combined)")
    roi_rows = []
    for pos in POSITIONS:
        pos_df = df[(df["position"] == pos) & (df["bid_amount"] > 1)]
        if pos_df.empty:
            continue
        roi_rows.append({
            "Position": pos,
            "N picks": len(pos_df),
            "Avg bid $": pos_df["bid_amount"].mean(),
            "Avg actual pts": pos_df["total_points"].mean(),
            "Avg proj pts": pos_df["projected_points"].mean(),
            "Pts/Dollar": (pos_df["total_points"] / pos_df["bid_amount"]).mean(),
            "Actual/Proj ratio": (pos_df["total_points"] / pos_df["projected_points"].clip(lower=1)).mean(),
        })

    roi_df = pd.DataFrame(roi_rows)
    with pd.option_context("display.float_format", "{:.2f}".format):
        print(roi_df.to_string(index=False))

    section("1b. Points/Dollar by position, last 5 years (2020-2024)")
    for pos in POSITIONS:
        recent = df[(df["position"] == pos) & (df["year"] >= 2020) & (df["bid_amount"] > 1)]
        if recent.empty:
            continue
        print(f"  {pos}: {(recent['total_points'] / recent['bid_amount']).mean():.2f} pts/$")


def analyze_manager_tendencies(df: pd.DataFrame) -> None:
    section("2. Manager Tendencies")
    mt_path = PROCESSED / f"manager_tendencies_{LEAGUE_ID}.csv"
    if not mt_path.exists():
        print("  Run build_features.py first.")
        return

    mt = pd.read_csv(mt_path)
    print("\n  Budget efficiency (WR pts/dollar vs RB pts/dollar — higher is better):")

    rows = []
    for _, row in mt.iterrows():
        mgr_data = df[df["manager_id"] == row["manager_id"]]
        eff_rows = {}
        for pos in ["WR", "RB"]:
            pos_data = mgr_data[(mgr_data["position"] == pos) & (mgr_data["bid_amount"] > 1)]
            if len(pos_data) > 0:
                eff_rows[f"{pos}_pts_per_dollar"] = (pos_data["total_points"] / pos_data["bid_amount"]).mean()
            else:
                eff_rows[f"{pos}_pts_per_dollar"] = np.nan
        rows.append({
            "Manager": row["manager_name"][:22],
            "RB $%": row.get("rb_budget_share", np.nan),
            "WR $%": row.get("wr_budget_share", np.nan),
            "RB pts/$": eff_rows.get("RB_pts_per_dollar", np.nan),
            "WR pts/$": eff_rows.get("WR_pts_per_dollar", np.nan),
            "$1 rate": row.get("dollar_one_rate", np.nan),
            "Seasons": row.get("seasons_active", np.nan),
        })

    eff_df = pd.DataFrame(rows).sort_values("WR pts/$", ascending=False)
    with pd.option_context("display.float_format", "{:.3f}".format):
        print(eff_df.to_string(index=False))


def analyze_projection_bias(df: pd.DataFrame) -> None:
    section("3. ESPN Projection Bias by Position")
    print("  actual_pts / projected_pts ratio — values > 1 mean ESPN under-projects\n")

    for pos in POSITIONS:
        pos_df = df[(df["position"] == pos) & (df["projected_points"] > 10)]
        if pos_df.empty:
            continue
        ratio = pos_df["total_points"] / pos_df["projected_points"]
        print(f"  {pos:<4}  mean={ratio.mean():.3f}  median={ratio.median():.3f}  "
              f"std={ratio.std():.3f}  N={len(pos_df)}")

    section("3b. Projection bias trend (2019-2024)")
    recent = df[df["year"] >= 2019]
    for pos in POSITIONS:
        pos_df = recent[(recent["position"] == pos) & (recent["projected_points"] > 10)]
        if pos_df.empty:
            continue
        ratio = pos_df["total_points"] / pos_df["projected_points"]
        print(f"  {pos:<4}  mean={ratio.mean():.3f}  → recommend scaling ESPN projection by {ratio.mean():.2f}x")


def analyze_winning_patterns(df: pd.DataFrame) -> None:
    section("4. Winning Patterns — Budget Allocation vs Final Standing")

    results = []
    for year in df["year"].unique():
        yr = df[df["year"] == year]
        total_by_team = yr.groupby("team_name")["total_points"].sum()
        total_by_team = total_by_team.sort_values(ascending=False)
        teams = total_by_team.index.tolist()
        n = len(teams)

        team_budget_total = yr.groupby("team_name")["bid_amount"].sum()
        for i, team in enumerate(teams):
            standing = i + 1  # 1 = best
            group = "top4" if standing <= 4 else "bottom8"
            total_bid = team_budget_total.get(team, 1)
            for pos in ["WR", "RB", "QB", "TE"]:
                pos_bid = yr[(yr["team_name"] == team) & (yr["position"] == pos)]["bid_amount"].sum()
                results.append({
                    "year": year, "team": team,
                    "standing": standing, "group": group,
                    "position": pos,
                    "budget_share": float(pos_bid / total_bid) if total_bid > 0 else 0.0,
                })

    res_df = pd.DataFrame(results)

    print("\n  Avg budget share by group (top-4 vs bottom-8):")
    pivot = res_df.pivot_table(
        index="position", columns="group", values="budget_share", aggfunc="mean"
    )
    with pd.option_context("display.float_format", "{:.3f}".format):
        print(pivot.to_string())

    print("\n  Spearman correlation: budget share vs standing (negative = higher share → better rank):")
    for pos in ["WR", "RB", "QB", "TE"]:
        sub = res_df[res_df["position"] == pos]
        if len(sub) < 10:
            continue
        rho, pval = scipy_stats.spearmanr(sub["budget_share"], sub["standing"])
        sig = "**" if pval < 0.01 else ("*" if pval < 0.05 else "")
        print(f"  {pos:<4}  ρ={rho:+.3f}  p={pval:.3f} {sig}")


def main() -> None:
    print("Loading all draft + season data...")
    df = load_all_draft_data()
    if df.empty:
        print(f"No data found. Check {FAVREFIGNEWTON}/ directory.")
        return
    print(f"  Loaded {len(df):,} picks across {df['year'].nunique()} seasons")

    analyze_position_roi(df)
    analyze_manager_tendencies(df)
    analyze_projection_bias(df)
    analyze_winning_patterns(df)

    print("\n\nAnalysis complete.")


if __name__ == "__main__":
    main()
