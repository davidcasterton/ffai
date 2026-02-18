#!/usr/bin/env python3
"""
PPO RL Training Script — three-phase curriculum.

Phase 1 (heuristic baseline, 500 episodes):
    Value model output used as heuristic (bid = fair_value * 0.95).
    No PPO learning. Validates integration before adding RL.

Phase 2 (PPO training, 2000 episodes):
    PPO agent learns to optimize bidding strategy.
    Monitor: avg_standing_position should trend from ~6 → ~3-4 by episode 1500.

Phase 3 (self-play, 3000 episodes):
    Some opponent teams are replaced with frozen PPO checkpoints.
    Prevents overfitting to the fixed ±10% ESPN opponent heuristic.

Usage:
    python scripts/train_rl.py --phase 1 --episodes 500
    python scripts/train_rl.py --phase 2 --episodes 2000
    python scripts/train_rl.py --phase 3 --episodes 3000
"""

import argparse
import sys
import json
from pathlib import Path
import numpy as np
import logging
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from ffai import get_logger
from ffai.data.espn_scraper import ESPNDraftScraper
from ffai.data.preprocessor import FantasyDataPreprocessor
from ffai.value_model.player_value_model import PlayerValueModel
from ffai.rl.ppo_agent import PPOAgent
from ffai.rl.state_builder import build_state
from ffai.rl.reward import mid_draft_reward, terminal_reward
from ffai.simulation.auction_draft_simulator import AuctionDraftSimulator
from ffai.simulation.season_simulator import SeasonSimulator

logger = get_logger(__name__)

DEFAULT_TRAINING_CONFIG = Path(__file__).parent.parent / "src/ffai/config/training.yaml"
DEFAULT_YEAR = 2024


class HeuristicAgent:
    """
    Phase 1 heuristic: bid 95% of fair_value from the value model.
    No gradient updates — used to validate simulation + value model integration.
    """

    def __init__(self, value_model: PlayerValueModel, budget: float = 200.0):
        self.value_model = value_model
        self.budget = budget

    def nominate_player(self, state: dict, available_players: list):
        if not available_players:
            return None
        # Nominate highest VORP-dollar player
        return max(available_players, key=lambda p: p.get("VORP_dollar", p.get("auction_value", 0)))

    def get_bid(self, state: dict, player: dict, min_bid: float, max_bid: float) -> float:
        fair_value = player.get("VORP_dollar", player.get("auction_value", 1))
        bid = max(min_bid, min(fair_value * 0.95, max_bid))
        return float(bid)

    def update(self, reward: float):
        pass  # No learning in phase 1


class PPODraftInterface:
    """
    Wraps PPOAgent to conform to the RLModel interface expected
    by AuctionDraftSimulator (nominate_player, get_bid, update).
    """

    def __init__(self, ppo_agent: PPOAgent, budget: float = 200.0):
        self.agent = ppo_agent
        self.budget = budget
        self._last_state = None
        self._last_action = None
        self._last_log_prob = None
        self._last_value = None

    def nominate_player(self, state: dict, available_players: list):
        if not available_players:
            return None
        # Nominate player with highest VORP dollar
        return max(available_players, key=lambda p: p.get("VORP_dollar", p.get("auction_value", 0)))

    def get_bid(self, state: dict, player: dict, min_bid: float, max_bid: float) -> float:
        current_budget = float(state.get("rl_team_budget", self.budget))
        state_tensor = build_state(state, current_player=player, current_bid=min_bid)

        bid_amount, action_frac, value = self.agent.get_bid_action(
            state_tensor, current_budget, min_bid=min_bid
        )

        # Store for later buffer insertion
        self._last_state = state_tensor
        self._last_action = action_frac
        self._last_value = value
        # Compute log_prob
        with __import__('torch').no_grad():
            _, lp = self.agent.actor.get_action(state_tensor.unsqueeze(0).to(self.agent.device))
            self._last_log_prob = lp.squeeze()

        return float(bid_amount)

    def update(self, reward: float):
        """Store transition in PPO rollout buffer (called per pick by simulator)."""
        if self._last_state is not None:
            self.agent.store(
                state=self._last_state,
                action=self._last_action,
                log_prob=self._last_log_prob,
                reward=reward,
                value=self._last_value,
                done=False,
            )
            self._last_state = None


def run_episode(
    rl_model,
    year: int,
    phase: int,
) -> dict:
    """Run one draft + season episode. Returns metrics dict."""
    simulator = AuctionDraftSimulator(year=year, rl_model=rl_model)
    draft_completed, draft_results, draft_reward = simulator.simulate_draft()

    if not draft_completed:
        return {"draft_completed": False, "standing": 11, "wins": 0, "draft_reward": draft_reward}

    season_sim = SeasonSimulator(draft_results=draft_results["teams"], year=year)
    season_sim.simulate_season()
    standings = season_sim.get_standings()
    weekly_results = season_sim.get_weekly_results()

    standing_pos = next(i for i, (team, _) in enumerate(standings) if team == simulator.rl_team_name)
    wins = sum(
        1 for week in weekly_results.values()
        for match in week
        if match.get("winner") == simulator.rl_team_name
    )

    season_reward = terminal_reward(standing_pos, wins)

    return {
        "draft_completed": True,
        "standing": standing_pos + 1,  # 1-indexed
        "wins": wins,
        "draft_reward": draft_reward,
        "season_reward": season_reward,
    }


def main():
    parser = argparse.ArgumentParser(description="Train PPO RL agent for auction draft bidding")
    parser.add_argument("--phase", type=int, choices=[1, 2, 3], default=2)
    parser.add_argument("--episodes", type=int, default=None)
    parser.add_argument("--year", type=int, default=DEFAULT_YEAR)
    parser.add_argument("--ppo-lr", type=float, default=3e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--training-config", type=Path, default=DEFAULT_TRAINING_CONFIG)
    parser.add_argument("--value-model-path", type=Path, default=Path("checkpoints/value_model/best_model.pt"))
    parser.add_argument("--checkpoint-dir", type=Path, default=Path("checkpoints/ppo"))
    parser.add_argument("--checkpoint-frequency", type=int, default=50)
    parser.add_argument("--load-latest", action="store_true")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    with open(args.training_config) as f:
        training_cfg = yaml.safe_load(f)

    phase_episodes = {
        1: training_cfg["training"]["phase1_episodes"],
        2: training_cfg["training"]["phase2_episodes"],
        3: training_cfg["training"]["phase3_episodes"],
    }
    num_episodes = args.episodes or phase_episodes[args.phase]

    logger.info(f"Phase {args.phase}: {num_episodes} episodes, year={args.year}")

    # Load value model if available
    value_model = None
    if args.value_model_path.exists():
        value_model = PlayerValueModel.load(args.value_model_path, device=args.device)
        logger.info(f"Loaded value model from {args.value_model_path}")
    else:
        logger.warning(
            f"Value model not found at {args.value_model_path}. "
            "Using raw projected_points. Run train_value_model.py first."
        )

    # Build RL model
    if args.phase == 1:
        rl_model = HeuristicAgent(value_model) if value_model else HeuristicAgent.__new__(HeuristicAgent)
        if value_model is None:
            # Minimal heuristic without value model
            rl_model.nominate_player = lambda state, players: (
                max(players, key=lambda p: p.get("projected_points", 0)) if players else None
            )
            rl_model.get_bid = lambda state, player, min_b, max_b: min(
                player.get("auction_value", 1) * 0.95, max_b
            )
            rl_model.update = lambda reward: None
    else:
        ppo_cfg = training_cfg["ppo"]
        ppo_agent = PPOAgent(
            lr_actor=args.ppo_lr,
            lr_critic=args.ppo_lr * 3,
            gamma=args.gamma,
            gae_lambda=args.gae_lambda,
            clip_epsilon=ppo_cfg.get("clip_epsilon", 0.2),
            num_epochs=ppo_cfg.get("num_epochs_per_update", 4),
            batch_size=ppo_cfg.get("batch_size", 64),
            checkpoint_dir=args.checkpoint_dir,
            device=args.device,
        )

        if args.load_latest:
            loaded = ppo_agent.load_latest()
            if loaded:
                logger.info("Loaded latest checkpoint")

        rl_model = PPODraftInterface(ppo_agent)

    # Training loop
    episode_metrics = []
    standings_window = []

    for episode in range(1, num_episodes + 1):
        metrics = run_episode(rl_model, year=args.year, phase=args.phase)
        episode_metrics.append(metrics)
        standings_window.append(metrics["standing"])

        if args.phase >= 2 and isinstance(rl_model, PPODraftInterface):
            # Signal end of episode to compute GAE
            rl_model.agent.finish_episode(last_value=0.0)

            # PPO update every 10 episodes
            if episode % 10 == 0 and rl_model.agent.buffer.ready:
                update_metrics = rl_model.agent.update()
                logger.debug(f"PPO update: {update_metrics}")

            # Checkpoint
            if episode % args.checkpoint_frequency == 0:
                rl_model.agent.save()

        # Log progress
        recent_standing = np.mean(standings_window[-50:]) if standings_window else 0
        logger.info(
            f"Episode {episode:4d}/{num_episodes} | "
            f"Standing: {metrics['standing']:2d} | "
            f"Wins: {metrics['wins']:2d} | "
            f"Draft reward: {metrics.get('draft_reward', 0):+.3f} | "
            f"Avg standing (50): {recent_standing:.1f}"
        )

    # Summary
    if episode_metrics:
        standings = [m["standing"] for m in episode_metrics if m["draft_completed"]]
        wins_list = [m["wins"] for m in episode_metrics if m["draft_completed"]]
        logger.info(
            f"\nTraining complete ({num_episodes} episodes):\n"
            f"  Avg standing: {np.mean(standings):.1f} (target: <6)\n"
            f"  Avg wins: {np.mean(wins_list):.1f}/17\n"
            f"  Best standing: {min(standings)}\n"
            f"  Top-4 rate: {sum(1 for s in standings if s <= 4) / len(standings):.1%}"
        )


if __name__ == "__main__":
    main()
