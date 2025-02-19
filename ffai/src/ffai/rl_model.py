import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from ffai.auction_draft_simulator import AuctionDraftSimulator
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
        self.checkpoint_dir = Path(checkpoint_dir) if checkpoint_dir else None

        # Input features:
        # - Player: [id, auction_value, projected_points] (3)
        # - Team budgets: [rl_team, 11 opponents] (12)
        # - Draft progress: [current_pick] (1)
        # - Points per slot: [12 teams * 14 slots] (168)
        # Total: 184 features
        self.model = nn.Sequential(
            nn.Linear(184, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)  # Output: bid amount
        )
        self.optimizer = optim.Adam(self.model.parameters())
        self.training = True

        # Load latest checkpoint if available
        if self.checkpoint_dir and self.checkpoint_dir.exists():
            checkpoints = list(self.checkpoint_dir.glob('*.pt'))
            if checkpoints:  # Only try to load if checkpoints exist
                latest = max(checkpoints, key=lambda x: x.stat().st_mtime)  # Get most recently modified
                logger.info(f"Loading checkpoint: {latest}")
                self.load_checkpoint(latest)
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
        state_values.append(float(player['auction_value']))
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

        return torch.FloatTensor(state_values).unsqueeze(0)

    def update(self, reward):
        """Update model based on season results"""
        if not self.training:
            return

        loss = -torch.tensor(reward, requires_grad=True)  # Negative reward for gradient descent
        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()

    def save_checkpoint(self, path):
        """Save model checkpoint"""
        if self.checkpoint_dir:
            path = self.checkpoint_dir / path
            path.parent.mkdir(parents=True, exist_ok=True)

        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }, path)
        logger.info(f"Saved checkpoint: {path}")

    def load_checkpoint(self, path):
        """Load model checkpoint"""
        if self.checkpoint_dir:
            path = self.checkpoint_dir / path

        checkpoint = torch.load(path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        logger.info(f"Loaded checkpoint: {path}")

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

def train_rl_model(model, draft_simulator, season_simulator, num_episodes=1000, checkpoint_frequency=10):
    """Train with safety checks and timeouts"""
    for episode in range(num_episodes):
        try:
            draft_simulator = AuctionDraftSimulator(year=2024, rl_model=model)
            draft_round = 0
            max_rounds = 1000

            while not draft_simulator.all_rosters_complete():
                if draft_round >= max_rounds:
                    logger.warning("Draft exceeded maximum rounds")
                    break

                nominated_player = draft_simulator.select_nomination(draft_simulator.rl_team)
                if nominated_player is None:
                    logger.warning("No valid nomination")
                    break

                # Rest of training loop...
                draft_round += 1

        except Exception as e:
            logger.error(f"Error in episode {episode}: {str(e)}")
            continue

if __name__ == "__main__":
    # Create model with checkpoints directory
    model = RLModel()

    # Train model
    train_rl_model(model, None, None, num_episodes=1000, checkpoint_frequency=10)
