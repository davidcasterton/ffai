#!/usr/bin/env python3
"""
Collect historical ESPN draft data for multiple years.

Stage 1 of the training pipeline.

Usage:
    python scripts/collect_data.py --years 2009-2024
    python scripts/collect_data.py --years 2020 2021 2022 --force
"""

import argparse
import sys
from pathlib import Path
import logging

# Ensure src/ is on the path when running from the scripts/ directory
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from ffai import get_logger
from ffai.data.espn_scraper import ESPNDraftScraper

logger = get_logger(__name__)


def parse_years(years_arg: list) -> list:
    """Parse years argument, supporting ranges like '2009-2024' or individual years."""
    result = []
    for arg in years_arg:
        if '-' in str(arg) and not str(arg).startswith('-'):
            parts = str(arg).split('-')
            start, end = int(parts[0]), int(parts[1])
            result.extend(range(start, end + 1))
        else:
            result.append(int(arg))
    return sorted(set(result))


def main():
    parser = argparse.ArgumentParser(description="Collect ESPN draft data for multiple years")
    parser.add_argument(
        "--years", nargs="+", default=["2009-2024"],
        help="Years to collect (e.g., 2009-2024 or 2020 2021 2022)"
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Re-fetch data even if cached files exist"
    )
    parser.add_argument(
        "--config", type=Path, default=None,
        help="Path to league.yaml config (default: config/league.yaml)"
    )
    args = parser.parse_args()

    years = parse_years(args.years)
    logger.info(f"Collecting data for years: {years}")

    scraper = ESPNDraftScraper(config_path=args.config)
    success_count = 0
    failure_count = 0

    for year in years:
        logger.info(f"\n{'='*50}")
        logger.info(f"Fetching {year}...")
        try:
            draft_df, stats_df, weekly_df, predraft_df, settings = scraper.load_or_fetch_data(
                year, force=args.force
            )
            logger.info(
                f"  Year {year}: "
                f"{len(draft_df)} picks, "
                f"{len(stats_df)} players, "
                f"{len(weekly_df)} weekly records"
            )
            success_count += 1
        except Exception as e:
            logger.error(f"  Failed to fetch {year}: {e}")
            failure_count += 1

    logger.info(f"\nComplete: {success_count} years fetched, {failure_count} failed")


if __name__ == "__main__":
    main()
