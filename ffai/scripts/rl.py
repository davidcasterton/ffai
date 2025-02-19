import argparse
from pathlib import Path
from ffai.auction_draft_simulator import AuctionDraftSimulator
from ffai.season_simulator import SeasonSimulator
from ffai.rl_model import RLModel, train_rl_model
from ffai import get_logger
import logging

def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Run the RL training for fantasy football.')
    parser.add_argument('-y', '--year', type=int, required=True, help='The year for which to load data')
    parser.add_argument('-e', '--episodes', type=int, default=1000, help='Number of training episodes')
    parser.add_argument('-c', '--checkpoint-dir', type=str, default='checkpoints',
                        help='Directory for saving/loading checkpoints')
    parser.add_argument('--checkpoint-frequency', type=int, default=10,
                        help='Save checkpoint every N episodes')
    parser.add_argument('--load-checkpoint', type=str,
                        help='Path to specific checkpoint to load')
    parser.add_argument('--no-load', action='store_true',
                        help='Do not load existing checkpoint')
    parser.add_argument('--debug', action='store_true',
                        help='Enable debug logging')

    # Parse arguments
    args = parser.parse_args()

    # Configure logging based on debug flag
    log_level = logging.DEBUG if args.debug else logging.INFO
    logging.getLogger().setLevel(log_level)

    logger = get_logger(__name__)
    logger.info(f"Training for year {args.year}")

    # Create checkpoint directory
    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Using checkpoint directory: {checkpoint_dir}")

    try:
        # Initialize model
        model = RLModel(checkpoint_dir=checkpoint_dir)

        # Load specific checkpoint if requested
        if args.load_checkpoint and not args.no_load:
            logger.info(f"Loading specified checkpoint: {args.load_checkpoint}")
            model.load_checkpoint(args.load_checkpoint)
        # Train model
        train_rl_model(
            model=model,
            num_episodes=args.episodes,
            checkpoint_frequency=args.checkpoint_frequency
        )

    except Exception as e:
        logger.error(f"Error during training: {str(e)}")
        raise

if __name__ == "__main__":
    main()
