import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from ffai.auction_draft_simulator import AuctionDraftSimulator, InvalidRosterException
from ffai.season_simulator import SeasonSimulator
import numpy as np
from collections import deque
import random
import json
from pathlib import Path
import logging
from ffai import get_logger

logger = get_logger(__name__)

class RLModel:
    def __init__(self, checkpoint_dir=None):
        """Initialize RL model

        Args:
            checkpoint_dir: Directory for saving/loading checkpoints
        """
        # Convert to absolute Path object if provided
        self.checkpoint_dir = Path(checkpoint_dir).absolute() if checkpoint_dir else None

        # Input features:
        # - Player: [id, auction_value, projected_points] (3)
        # - Team budgets: [rl_team, 11 opponents] (12)
        # - Draft progress: [current_pick] (1)
        # - Points per slot: [12 teams * 14 slots] (168)
        # - Position needs: [QB, RB, WR, TE, FLEX, D/ST, K, BENCH] (6)
        # Total: 190 features
        self.model = nn.Sequential(
            nn.Linear(190, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)  # Output: bid amount
        )
        self.optimizer = optim.Adam(self.model.parameters())
        self.training = True

        # Load latest checkpoint if available
        if self.checkpoint_dir and self.checkpoint_dir.exists():
            checkpoints = list(self.checkpoint_dir.glob('checkpoint_*.pt'))
            if checkpoints:  # Only try to load if checkpoints exist
                latest = max(checkpoints, key=lambda x: x.stat().st_mtime)
                logger.info(f"Loading checkpoint: {latest}")
                try:
                    self.load_checkpoint(str(latest))  # Pass string path to avoid Path concatenation
                except Exception as e:
                    logger.warning(f"Failed to load checkpoint {latest}: {e}")
                    logger.info("Starting with fresh model")
            else:
                logger.info(f"No checkpoints found in {self.checkpoint_dir}, starting fresh")

    def nominate_player(self, state, available_players):
        """Choose a player to nominate"""
        with torch.no_grad():
            best_value = float('-inf')
            best_player = None

            for player in available_players:
                # Get predicted value for this player
                state_tensor = self.prepare_state_tensor(state, player)
                value = self.model(state_tensor).item()

                if value > best_value:
                    best_value = value
                    best_player = player

            return best_player

    def get_bid(self, state, player, min_bid, max_bid):
        """Decide how much to bid on a player"""
        with torch.no_grad():
            state_tensor = self.prepare_state_tensor(state, player)
            bid = self.model(state_tensor).item()
            # Ensure bid is within valid range
            return max(min_bid, min(bid, max_bid))

    def prepare_state_tensor(self, state, player):
        """Convert state dict and player into tensor"""
        state_values = []

        # Player info (3 values)
        state_values.append(float(player['player_id']))
        state_values.append(float(player.get('auction_value', 0)))
        state_values.append(float(player['projected_points']))

        # Budget info (12 values)
        state_values.append(float(state['rl_team_budget']))
        state_values.extend([float(b) for b in state['opponent_budgets']])

        # Draft progress (1 value)
        state_values.append(float(state['draft_turn']))

        # Points per slot (168 values)
        for team in state['teams']:
            points = state['predicted_points_per_slot'][team]
            # Pad or truncate to exactly 14 slots
            points = points[:14] + [0.0] * (14 - len(points))
            state_values.extend(points)

        # Position needs (6 values)
        position_needs = [
            float(state['position_needs'].get('QB', 0)),
            float(state['position_needs'].get('RB', 0)),
            float(state['position_needs'].get('WR', 0)),
            float(state['position_needs'].get('TE', 0)),
            float(state['position_needs'].get('D/ST', 0)),
            float(state['position_needs'].get('K', 0))
        ]
        state_values.extend(position_needs)

        return torch.FloatTensor(state_values).unsqueeze(0)

    def update(self, reward):
        """Update model weights based on reward"""
        if not self.training:
            return

        # Convert reward to float tensor
        loss = -torch.tensor(float(reward), dtype=torch.float32, requires_grad=True)

        # Backpropagate loss
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()

    def save_checkpoint(self, path):
        """Save model checkpoint"""
        if not self.checkpoint_dir:
            return

        # Handle path as string to avoid Path concatenation issues
        full_path = str(self.checkpoint_dir / path)
        Path(full_path).parent.mkdir(parents=True, exist_ok=True)

        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }, full_path)
        logger.info(f"Saved checkpoint: {full_path}")

    def load_checkpoint(self, path):
        """Load model checkpoint"""
        # If path is not absolute and we have a checkpoint_dir, make it relative to that
        if not Path(path).is_absolute() and self.checkpoint_dir:
            full_path = str(self.checkpoint_dir / path)
        else:
            full_path = str(path)

        logger.info(f"Loading checkpoint from: {full_path}")
        checkpoint = torch.load(full_path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

def calculate_reward(draft_results, standings, weekly_results, rl_team):
    """Calculate reward based on season performance"""
    reward = 0

    # Weekly wins reward (+1 per win)
    for week in weekly_results.values():
        for match in week:
            if match['winner'] == rl_team:
                reward += 1

    # Final standing reward
    standing_position = next(i for i, (team, _) in enumerate(standings) if team == rl_team)
    if standing_position == 0:  # 1st place
        reward += 10
    elif standing_position == 1:  # 2nd place
        reward += 5
    elif standing_position <= 3:  # 3rd or 4th
        reward += 2

    return reward

def train_rl_model(model, num_episodes=1000, checkpoint_frequency=10):
    """Train RL model on draft and season simulations"""
    logger = get_logger(__name__)

    for episode in range(num_episodes):
        # Reset simulators for new episode
        draft_simulator = AuctionDraftSimulator(year=2024, rl_model=model)

        try:
            # Run draft simulation
            draft_results = draft_simulator.simulate_draft()

            # Initialize season simulator with draft results
            season_simulator = SeasonSimulator(draft_results=draft_results, year=2024)

            # Simulate season and get reward
            season_simulator.simulate_season()
            standings = season_simulator.get_standings()
            weekly_results = season_simulator.get_weekly_results()

            # Calculate reward based on season performance
            reward = calculate_reward(draft_results, standings, weekly_results, draft_simulator.rl_team_name)
        except InvalidRosterException as e:
            logger.error(f"Invalid roster for {draft_simulator.rl_team_name}: {e}")
            reward = -10

        # Update model weights based on reward
        model.update(reward)

        # Save checkpoint
        if episode % checkpoint_frequency == 0:
            model.save_checkpoint(f'checkpoint_{episode}.pt')

        logger.info(f'Episode {episode} complete - Reward: {reward}')

    return model

if __name__ == "__main__":
    # Create model with checkpoints directory
    model = RLModel()

    # Train model
    train_rl_model(model, None, None, num_episodes=1000, checkpoint_frequency=10)
