#!/usr/bin/env python3
"""
Sanity-check ESPN draft data for budget anomalies.

For each year, detects whether the league used a non-$200 auction budget,
reports bid distribution statistics, and writes budget_by_year.csv which
build_features.py reads for normalization.

Usage:
    python scripts/sanity_check_data.py [--years 2009-2024]
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import pandas as pd
import numpy as np
from ffai.data.espn_scraper import load_league_config

_cfg = load_league_config()
_league_name = _cfg["league"]["league_name"]
_league_id = _cfg["league"]["league_id"]

DATA_DIR = Path(__file__).parent.parent / f"src/ffai/data/{_league_name}"
PROCESSED_DIR = Path(__file__).parent.parent / f"src/ffai/data/{_league_name}_processed"


def detect_budget(draft_df: pd.DataFrame) -> int:
    """Infer auction budget from median team spend."""
    team_spend = draft_df.groupby("manager_id")["bid_amount"].sum()
    median_spend = team_spend.median()
    return 1000 if median_spend > 300 else 200


def analyze_year(year: int, league_id: str) -> dict | None:
    dr_path = DATA_DIR / f"draft_results_{league_id}_{year}.csv"
    if not dr_path.exists():
        return None

    df = pd.read_csv(dr_path)
    if df.empty or "bid_amount" not in df.columns:
        return None

    detected_budget = detect_budget(df)
    team_spend = df.groupby("manager_id")["bid_amount"].sum()
    median_team_spend = float(team_spend.median())

    bids = df["bid_amount"]
    dollar_one_count = int((bids == 1).sum())
    total_picks = len(bids)

    result = {
        "year": year,
        "detected_budget": detected_budget,
        "median_team_spend": round(median_team_spend, 1),
        "total_picks": total_picks,
        "bid_p5": round(float(np.percentile(bids, 5)), 1),
        "bid_median": round(float(np.median(bids)), 1),
        "bid_p95": round(float(np.percentile(bids, 95)), 1),
        "bid_max": round(float(bids.max()), 1),
        "dollar_one_pct": round(100.0 * dollar_one_count / total_picks, 1),
    }
    return result


def flag_manager_anomalies(years: list[int], league_id: str) -> None:
    """Print managers whose bid_per_proj_pt varies >3× across years."""
    frames = []
    for year in years:
        dr_path = DATA_DIR / f"draft_results_{league_id}_{year}.csv"
        pv_path = DATA_DIR / f"predraft_values_{league_id}_{year}.csv"
        if not dr_path.exists() or not pv_path.exists():
            continue
        dr = pd.read_csv(dr_path)
        pv = pd.read_csv(pv_path)[["player_id", "position", "projected_points"]]
        dr["player_id"] = dr["player_id"].astype(str)
        pv["player_id"] = pv["player_id"].astype(str)
        merged = dr.merge(pv, on="player_id", how="left")
        merged["year"] = year
        # Normalize bid_amount to $200-equivalent for this check
        budget = detect_budget(dr)
        norm = 200.0 / budget
        merged["bid_norm"] = merged["bid_amount"] * norm
        frames.append(merged)

    if not frames:
        return

    all_df = pd.concat(frames, ignore_index=True)

    print("\n[Manager anomaly check]")
    flagged = []
    for mgr_id, grp in all_df.groupby("manager_id"):
        name = grp["manager_display_name"].iloc[0] if "manager_display_name" in grp.columns else str(mgr_id)
        rb = grp[(grp["position"] == "RB") & (grp["projected_points"] > 0)].copy()
        if len(rb) < 3:
            continue
        rb["bpppt"] = rb["bid_norm"] / rb["projected_points"]
        yearly = rb.groupby("year")["bpppt"].mean()
        if len(yearly) < 2:
            continue
        med = yearly.median()
        if med == 0:
            continue
        if yearly.max() > 3.0 * med:
            flagged.append({
                "manager": name,
                "median_bpppt_rb": round(med, 3),
                "max_bpppt_rb": round(yearly.max(), 3),
                "worst_year": int(yearly.idxmax()),
            })

    if flagged:
        print(f"  {'Manager':<20} {'median_bpppt_rb':>16} {'max_bpppt_rb':>13} {'worst_year':>11}")
        for f in sorted(flagged, key=lambda x: -x["max_bpppt_rb"]):
            print(f"  {f['manager']:<20} {f['median_bpppt_rb']:>16.3f} {f['max_bpppt_rb']:>13.3f} {f['worst_year']:>11}")
    else:
        print("  No manager anomalies detected (all within 3× median).")


def main() -> None:
    parser = argparse.ArgumentParser(description="Sanity-check ESPN draft data for budget anomalies")
    parser.add_argument("--years", default="2009-2024")
    args = parser.parse_args()

    raw = args.years
    if "-" in raw and "," not in raw:
        start, end = raw.split("-")
        years = list(range(int(start), int(end) + 1))
    else:
        years = [int(s.strip()) for s in raw.split(",")]

    print(f"Sanity-checking {_league_name} (id={_league_id}), years {years[0]}–{years[-1]}")
    print(f"Data dir: {DATA_DIR}\n")

    rows = []
    for year in years:
        r = analyze_year(year, _league_id)
        if r is not None:
            rows.append(r)

    if not rows:
        print("No draft data found.")
        return

    df = pd.DataFrame(rows)

    # --- Per-year table ---
    print("[Per-year budget detection]")
    print(f"  {'Year':>4}  {'Budget':>7}  {'MedSpend':>9}  {'Picks':>6}  "
          f"{'bid_p5':>6}  {'median':>7}  {'bid_p95':>8}  {'max':>6}  {'$1 pct':>7}")
    for _, row in df.iterrows():
        budget_flag = "  *** NON-$200 ***" if row["detected_budget"] != 200 else ""
        print(f"  {int(row['year']):>4}  ${row['detected_budget']:>6}  "
              f"${row['median_team_spend']:>8.1f}  {int(row['total_picks']):>6}  "
              f"${row['bid_p5']:>5.1f}  ${row['bid_median']:>6.1f}  "
              f"${row['bid_p95']:>7.1f}  ${row['bid_max']:>5.1f}  "
              f"{row['dollar_one_pct']:>6.1f}%{budget_flag}")

    non_200 = df[df["detected_budget"] != 200]
    print(f"\nSummary: {len(non_200)} of {len(df)} years had non-$200 budgets.")
    if not non_200.empty:
        print("Non-standard years:", list(non_200["year"].astype(int)))

    # --- Manager anomaly check ---
    flag_manager_anomalies(years, _league_id)

    # --- Write budget_by_year.csv ---
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    out_path = PROCESSED_DIR / "budget_by_year.csv"
    df[["year", "detected_budget", "median_team_spend"]].to_csv(out_path, index=False)
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
