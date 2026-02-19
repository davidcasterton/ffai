#!/usr/bin/env python3
"""
Download and cache nflverse data into ffai/src/ffai/data/nflverse/.

Downloads:
  player_stats_seasonal.parquet  - per-player season rushing/receiving/passing totals
  player_stats_weekly.parquet    - per-player per-week stats (2012+)
  rosters.parquet                - player demographics (age, position, team, draft round)
  draft_picks.parquet            - all-time draft picks with pick number and round
  snap_counts.parquet            - weekly snap count % by player (2012+)
  ff_playerids.parquet           - cross-reference table (gsis_id ↔ ESPN id, etc.)

Usage:
    python scripts/fetch_nfl_data.py [--seasons 2009-2024]

Data is cached in data/nflverse/ and gitignored.
"""

import sys
import argparse
from pathlib import Path

# Ensure src/ is on path when run from repo root
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

NFLVERSE_DIR = Path(__file__).parent.parent / "src/ffai/data/nflverse"

DEFAULT_SEASONS = list(range(2009, 2025))
SNAP_COUNT_SEASONS = list(range(2012, 2025))  # snap counts available from 2012


def fetch_all(seasons: list[int]) -> None:
    from nflreadpy.load_stats import load_player_stats
    from nflreadpy.load_rosters import load_rosters
    from nflreadpy.load_draft_picks import load_draft_picks
    from nflreadpy.load_snap_counts import load_snap_counts
    from nflreadpy.load_ffverse import load_ff_playerids

    NFLVERSE_DIR.mkdir(parents=True, exist_ok=True)

    snap_seasons = [s for s in seasons if s >= 2012]

    datasets = [
        ("player_stats_seasonal.parquet", lambda: load_player_stats(seasons=seasons, summary_level="reg")),
        ("player_stats_weekly.parquet",   lambda: load_player_stats(seasons=seasons, summary_level="week")),
        ("rosters.parquet",              lambda: load_rosters(seasons=seasons)),
        ("draft_picks.parquet",          lambda: load_draft_picks()),
        ("snap_counts.parquet",          lambda: load_snap_counts(seasons=snap_seasons)),
        ("ff_playerids.parquet",         lambda: load_ff_playerids()),
    ]

    for filename, loader in datasets:
        out_path = NFLVERSE_DIR / filename
        print(f"  Fetching {filename}...", end=" ", flush=True)
        try:
            df = loader()
            df.write_parquet(out_path)
            print(f"done ({len(df):,} rows → {out_path.name})")
        except Exception as exc:
            print(f"FAILED: {exc}")

    print(f"\nAll data written to {NFLVERSE_DIR}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch nflverse data for ffai")
    parser.add_argument(
        "--seasons",
        type=str,
        default="2009-2024",
        help="Season range, e.g. 2009-2024 or comma-separated list",
    )
    args = parser.parse_args()

    raw = args.seasons
    if "-" in raw and "," not in raw:
        start, end = raw.split("-")
        seasons = list(range(int(start), int(end) + 1))
    else:
        seasons = [int(s.strip()) for s in raw.split(",")]

    print(f"Fetching nflverse data for seasons {seasons[0]}–{seasons[-1]} ...")
    fetch_all(seasons)


if __name__ == "__main__":
    main()
