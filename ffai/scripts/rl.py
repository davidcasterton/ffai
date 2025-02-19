import argparse
from pathlib import Path
from ffai.auction_draft_simulator import AuctionDraftSimulator
from ffai.season_simulator import SeasonSimulator
from ffai.rl_model import RLModel, train_rl_model
from ffai import setup_logger
import logging

logger = setup_logger(__name__)

def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Run the RL training for fantasy football.')
    parser.add_argument('-y', '--year', type=int, required=True, help='The year for which to load data')
    parser.add_argument('-e', '--episodes', type=int, default=1000, help='Number of training episodes')
    parser.add_argument('-c', '--checkpoint-dir', type=str, default='checkpoints/auction_draft',
                        help='Directory for saving/loading checkpoints')
    parser.add_argument('--checkpoint-frequency', type=int, default=10,
                        help='Save checkpoint every N episodes')
    parser.add_argument('--load-checkpoint', type=str,
                        help='Path to specific checkpoint to load')
    parser.add_argument('--no-load', action='store_true',
                        help='Start fresh without loading latest checkpoint')
    parser.add_argument('--debug', action='store_true',
                        help='Enable debug logging')

    # Parse arguments
    args = parser.parse_args()

    # Set up logging
    log_level = logging.DEBUG if args.debug else logging.INFO
    logger = setup_logger(__name__, level=log_level)

    # Create checkpoint directory
    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Training for year {args.year}")
    logger.info(f"Using checkpoint directory: {checkpoint_dir}")

    try:
        # Initialize model
        model = RLModel(checkpoint_dir=checkpoint_dir)

        # Load checkpoint if requested
        if not args.no_load:
            if args.load_checkpoint:
                logger.info(f"Loading specified checkpoint: {args.load_checkpoint}")
                model.load_checkpoint(Path(args.load_checkpoint))
            else:
                logger.info("Attempting to load latest checkpoint...")
                model.load_checkpoint()

        # Initialize simulators with model
        auction_simulator = AuctionDraftSimulator(year=args.year, rl_model=model)

        # Run initial draft to get valid draft results
        auction_simulator.simulate_draft()
        initial_draft_results = auction_simulator.get_draft_results()

        # Initialize season simulator with valid draft results
        season_simulator = SeasonSimulator(draft_results=initial_draft_results, year=args.year)

        # Train model
        train_rl_model(
            model=model,
            draft_simulator=auction_simulator,
            season_simulator=season_simulator,
            num_episodes=args.episodes,
            checkpoint_frequency=args.checkpoint_frequency
        )

    except Exception as e:
        logger.error(f"Error during training: {str(e)}")
        raise

if __name__ == "__main__":
    main()
