#!/usr/bin/env python3
"""
Launch autonomous mode — PPO agent controls ESPN draft room via Playwright.

WARNING: Read the docs in interfaces/autonomous.py before running.
Always test in a mock/test league before using in a real draft.

Usage:
    python scripts/autonomous_draft.py --team-id 1
    python scripts/autonomous_draft.py --team-id 3 --ppo-checkpoint checkpoints/ppo/ppo_checkpoint_2000.pt
"""

import argparse
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from ffai import get_logger
from ffai.interfaces.autonomous import AutonomousMode
from ffai.rl.ppo_agent import PPOAgent
from ffai.value_model.player_value_model import PlayerValueModel

logger = get_logger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Autonomous mode — PPO agent controls ESPN draft")
    parser.add_argument("--team-id", type=int, required=True, help="Your ESPN team ID")
    parser.add_argument("--config", type=Path, default=None, help="Path to league.yaml")
    parser.add_argument("--value-model", type=Path, default=None)
    parser.add_argument("--ppo-checkpoint", type=Path, default=None, help="Path to PPO checkpoint")
    parser.add_argument("--budget", type=float, default=200.0)
    parser.add_argument("--poll-interval", type=float, default=2.5)
    parser.add_argument("--headless", action="store_true", help="Run browser headlessly (not recommended)")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    # Load value model
    value_model = None
    if args.value_model and args.value_model.exists():
        value_model = PlayerValueModel.load(args.value_model, device=args.device)
        logger.info(f"Loaded value model from {args.value_model}")

    # Load PPO agent
    if not args.ppo_checkpoint or not args.ppo_checkpoint.exists():
        logger.error("PPO checkpoint required for autonomous mode. Train first with train_rl.py.")
        sys.exit(1)

    ppo_agent = PPOAgent(device=args.device)
    ppo_agent.load(args.ppo_checkpoint)
    logger.info(f"Loaded PPO agent from {args.ppo_checkpoint}")

    print("\n" + "="*60)
    print("  AUTONOMOUS DRAFT MODE")
    print("  The PPO agent will control your draft automatically.")
    print("  Keep the browser window visible to monitor.")
    print("  Close the browser to abort at any time.")
    print("="*60 + "\n")

    auto = AutonomousMode(
        config_path=args.config,
        value_model=value_model,
        ppo_agent=ppo_agent,
        rl_team_id=args.team_id,
        budget=args.budget,
        poll_interval=args.poll_interval,
        headless=args.headless,
    )

    asyncio.run(auto.run())


if __name__ == "__main__":
    main()
