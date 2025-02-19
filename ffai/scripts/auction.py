import argparse
from ffai.auction_draft_simulator import AuctionDraftSimulator

def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Run the fantasy football auction draft simulator.')
    parser.add_argument('-y', '--year', type=int, required=True, help='The year for which to load data')

    # Parse arguments
    args = parser.parse_args()

    # Initialize and run the auction draft simulator
    simulator = AuctionDraftSimulator(year=args.year)
    simulator.simulate_draft()

    # Print the team rosters
    simulator.print_team_rosters()

if __name__ == "__main__":
    main()

