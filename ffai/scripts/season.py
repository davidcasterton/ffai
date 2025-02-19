import argparse
from ffai.auction_draft_simulator import AuctionDraftSimulator
from ffai.season_simulator import SeasonSimulator
from ffai.data.espn_scraper import ESPNDraftScraper

def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Run the fantasy football season simulator.')
    parser.add_argument('-y', '--year', type=int, required=True, help='The year for which to load data')

    # Parse arguments
    args = parser.parse_args()

    # Load data
    scraper = ESPNDraftScraper()
    draft_df, stats_df, weekly_df, predraft_df, settings = scraper.load_or_fetch_data(args.year)

    # Initialize and run the auction draft simulator
    auction_simulator = AuctionDraftSimulator(year=args.year)
    auction_simulator.simulate_draft()
    draft_results = auction_simulator.get_draft_results()

    # Initialize and run the season simulator
    season_simulator = SeasonSimulator(draft_results=draft_results, year=args.year)
    season_simulator.simulate_season()

    # Print the results of each week's matchups
    for week, results in season_simulator.get_weekly_results().items():
        print(f"\nWeek {week} Results:")
        for matchup in results:
            print(f"  {matchup['team1']} ({matchup['team1_score']}) vs {matchup['team2']} ({matchup['team2_score']}) - Winner: {matchup['winner']}")

if __name__ == "__main__":
    main()
