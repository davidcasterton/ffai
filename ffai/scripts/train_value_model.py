#!/usr/bin/env python3
"""
Train the PlayerValueModel (supervised, two-head regression).

Stage 2 of the training pipeline.

Usage:
    python scripts/train_value_model.py
    python scripts/train_value_model.py --train-years 2009-2022 --val-year 2023 --test-year 2024
    python scripts/train_value_model.py --config config/league.yaml
"""

import argparse
import sys
from pathlib import Path
import logging
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from ffai import get_logger
from ffai.data.espn_scraper import ESPNDraftScraper
from ffai.data.preprocessor import FantasyDataPreprocessor
from ffai.value_model.player_value_model import PlayerValueModel
from ffai.value_model.value_trainer import ValueModelTrainer

logger = get_logger(__name__)

DEFAULT_TRAINING_CONFIG = Path(__file__).parent.parent / "src/ffai/config/training.yaml"
DEFAULT_CHECKPOINT_DIR = Path("checkpoints/value_model")
DEFAULT_PREPROCESSOR_DIR = Path("checkpoints/preprocessor")


def parse_years(years_arg: list) -> list:
    result = []
    for arg in years_arg:
        arg = str(arg)
        if '-' in arg and not arg.startswith('-'):
            parts = arg.split('-')
            start, end = int(parts[0]), int(parts[1])
            result.extend(range(start, end + 1))
        else:
            result.append(int(arg))
    return sorted(set(result))


def main():
    parser = argparse.ArgumentParser(description="Train PlayerValueModel on historical ESPN data")
    parser.add_argument(
        "--train-years", nargs="+", default=["2009-2022"],
        help="Training years (default: 2009-2022)"
    )
    parser.add_argument("--val-year", type=int, default=2023, help="Validation year")
    parser.add_argument("--test-year", type=int, default=2024, help="Test year")
    parser.add_argument("--config", type=Path, default=None, help="Path to league.yaml")
    parser.add_argument("--training-config", type=Path, default=DEFAULT_TRAINING_CONFIG)
    parser.add_argument("--checkpoint-dir", type=Path, default=DEFAULT_CHECKPOINT_DIR)
    parser.add_argument("--preprocessor-dir", type=Path, default=DEFAULT_PREPROCESSOR_DIR)
    parser.add_argument("--lr", type=float, default=None, help="Override learning rate")
    parser.add_argument("--epochs", type=int, default=None, help="Override num_epochs")
    parser.add_argument("--batch-size", type=int, default=None, help="Override batch_size")
    parser.add_argument("--device", default="cpu", help="Training device (cpu or cuda)")
    args = parser.parse_args()

    # Load training config
    with open(args.training_config) as f:
        training_cfg = yaml.safe_load(f)

    vm_cfg = training_cfg["value_model"]
    lr = args.lr or vm_cfg.get("learning_rate", 1e-3)
    epochs = args.epochs or vm_cfg.get("num_epochs", 100)
    batch_size = args.batch_size or vm_cfg.get("batch_size", 64)
    patience = vm_cfg.get("early_stopping_patience", 10)

    train_years = parse_years(args.train_years)
    logger.info(f"Train years: {train_years}, Val: {args.val_year}, Test: {args.test_year}")

    # Load data
    scraper = ESPNDraftScraper(config_path=args.config)
    preprocessor = FantasyDataPreprocessor()

    logger.info("Loading training data...")
    train_year_data = []
    for year in train_years:
        try:
            draft_df, stats_df, _, _, _ = scraper.load_or_fetch_data(year)
            train_year_data.append((year, draft_df, stats_df))
        except Exception as e:
            logger.warning(f"Skipping year {year}: {e}")

    if not train_year_data:
        logger.error("No training data loaded. Run collect_data.py first.")
        sys.exit(1)

    logger.info(f"Loaded {len(train_year_data)} training years")

    # Process training data (fits encoders)
    logger.info("Processing training data...")
    train_data = preprocessor.process_multi_year(train_year_data)

    # Process validation data
    logger.info(f"Loading validation data ({args.val_year})...")
    val_draft, val_stats, _, _, _ = scraper.load_or_fetch_data(args.val_year)
    val_data = preprocessor.process_draft_data(val_draft, val_stats, year=args.val_year)

    # Process test data
    logger.info(f"Loading test data ({args.test_year})...")
    test_draft, test_stats, _, _, _ = scraper.load_or_fetch_data(args.test_year)
    test_data = preprocessor.process_draft_data(test_draft, test_stats, year=args.test_year)

    # Save preprocessor state (for use during RL training and inference)
    args.preprocessor_dir.mkdir(parents=True, exist_ok=True)
    preprocessor.save(args.preprocessor_dir)
    logger.info(f"Preprocessor saved to {args.preprocessor_dir}")

    # Build model
    logger.info(f"Building model: {preprocessor.num_players} players, {preprocessor.num_positions} positions")
    model = PlayerValueModel(
        num_players=preprocessor.num_players,
        num_positions=preprocessor.num_positions,
        player_embedding_dim=vm_cfg.get("player_embedding_dim", 256),
        position_embedding_dim=vm_cfg.get("position_embedding_dim", 64),
        numerical_hidden_dim=vm_cfg.get("numerical_hidden_dim", 128),
    )

    # Train
    trainer = ValueModelTrainer(
        model=model,
        learning_rate=lr,
        batch_size=batch_size,
        num_epochs=epochs,
        early_stopping_patience=patience,
        checkpoint_dir=args.checkpoint_dir,
        device=args.device,
    )

    logger.info("Training value model...")
    history = trainer.train(train_data, val_data)

    # Evaluate on test set
    logger.info("Evaluating on test set...")
    test_metrics = trainer.evaluate(test_data)

    logger.info(
        f"\nFinal test metrics:\n"
        f"  Points RMSE: {test_metrics['points_rmse']:.2f} pts\n"
        f"  Dollar MAE:  ${test_metrics['dollar_mae']:.2f}\n"
        f"\nTarget: RMSE ~5-6 pts, MAE ~$5-8"
    )

    logger.info(f"Best model saved to {args.checkpoint_dir / 'best_model.pt'}")


if __name__ == "__main__":
    main()
