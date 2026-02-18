#!/usr/bin/env python3
"""
Launch advisory mode — monitors ESPN draft and prints bidding recommendations.

The human controls the draft; the agent provides recommendations.

Usage:
    python scripts/advisory_draft.py --team-id 1
    python scripts/advisory_draft.py --team-id 3 --value-model checkpoints/value_model/best_model.pt
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from ffai import get_logger
from ffai.interfaces.advisory import AdvisoryMode
from ffai.rl.ppo_agent import PPOAgent
from ffai.value_model.player_value_model import PlayerValueModel

logger = get_logger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Advisory mode for live ESPN draft")
    parser.add_argument("--team-id", type=int, required=True, help="Your ESPN team ID")
    parser.add_argument("--config", type=Path, default=None, help="Path to league.yaml")
    parser.add_argument("--value-model", type=Path, default=None, help="Path to value model checkpoint")
    parser.add_argument("--ppo-checkpoint", type=Path, default=None, help="Path to PPO agent checkpoint")
    parser.add_argument("--budget", type=float, default=200.0)
    parser.add_argument("--poll-interval", type=float, default=2.5)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    # Load value model
    value_model = None
    if args.value_model and args.value_model.exists():
        value_model = PlayerValueModel.load(args.value_model, device=args.device)
        logger.info(f"Loaded value model from {args.value_model}")
    else:
        logger.warning("No value model loaded. Using projected_points as proxy.")

    # Load PPO agent
    ppo_agent = None
    if args.ppo_checkpoint and args.ppo_checkpoint.exists():
        ppo_agent = PPOAgent(device=args.device)
        ppo_agent.load(args.ppo_checkpoint)
        logger.info(f"Loaded PPO agent from {args.ppo_checkpoint}")
    else:
        logger.warning("No PPO agent loaded. Using fair_value * 0.95 heuristic.")

    advisory = AdvisoryMode(
        config_path=args.config,
        value_model=value_model,
        ppo_agent=ppo_agent,
        rl_team_id=args.team_id,
        budget=args.budget,
        poll_interval=args.poll_interval,
    )

    advisory.run()


if __name__ == "__main__":
    main()
