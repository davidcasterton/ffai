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

class FFAIModel(nn.Module):
    def __init__(self, state_size, action_size):
        super(FFAIModel, self).__init__()

        # Separate processing paths for different state components
        self.budget_net = nn.Sequential(
            nn.Linear(2, 32),  # [rl_budget, remaining_budget_per_need]
            nn.ReLU()
        )

        self.opponent_net = nn.Sequential(
            nn.Linear(11, 32),  # opponent budgets
            nn.ReLU()
        )

        self.position_net = nn.Sequential(
            nn.Linear(18, 64),  # 6 positions * 3 features (counts, scarcity, needs)
            nn.ReLU()
        )

        self.value_net = nn.Sequential(
            nn.Linear(12, 32),  # 6 positions * 2 values (avg_value, avg_points)
            nn.ReLU()
        )

        # Progress and points features
        self.progress_net = nn.Sequential(
            nn.Linear(2, 16),  # [draft_progress, total_team_points]
            nn.ReLU()
        )

        # Combine all features
        combined_size = 32 + 32 + 64 + 32 + 16  # Sum of all intermediate layers

        self.combined_net = nn.Sequential(
            nn.Linear(combined_size, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, action_size)
        )

    def forward(self, x):
        """Process state features through specialized networks before combining"""
        # Unpack the input tensor into its components
        budget_features = x[:, :2]  # First 2 features
        opponent_features = x[:, 2:13]  # Next 11 features
        position_features = x[:, 13:31]  # Next 18 features (6 positions * 3 features)
        value_features = x[:, 31:43]  # Next 12 features (6 positions * 2 features)
        progress_features = x[:, 43:45]  # Last 2 features

        # Process each feature type through its specialized network
        budget_out = self.budget_net(budget_features)
        opponent_out = self.opponent_net(opponent_features)
        position_out = self.position_net(position_features)
        value_out = self.value_net(value_features)
        progress_out = self.progress_net(progress_features)

        # Combine all processed features
        combined = torch.cat([
            budget_out,
            opponent_out,
            position_out,
            value_out,
            progress_out
        ], dim=1)

        # Final processing through combined network
        return self.combined_net(combined)

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
        self.model = FFAIModel(190, 1)
        self.optimizer = optim.Adam(self.model.parameters())
        self.training = True
        self.episode_losses = []  # Track losses for each episode
        self.running_reward = 0
        self.alpha = 0.1  # For exponential moving average

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

        # Update running reward average
        self.running_reward = (1 - self.alpha) * self.running_reward + self.alpha * reward

        # Backpropagate loss
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Track loss
        loss_value = loss.item()
        self.episode_losses.append(loss_value)

        # Log detailed metrics every update
        logger.debug(f"Update - Loss: {loss_value:.2f}, Running Reward: {self.running_reward:.2f}")

        return loss_value

    def save_checkpoint(self, path, metrics=None):
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

        if metrics:
            with open(f"{full_path}.metrics", 'w') as f:
                json.dump(metrics, f)

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

    # Track metrics
    episode_rewards = []
    episode_draft_rewards = []
    episode_season_rewards = []

    for episode in range(num_episodes):
        episode_start_loss = np.mean(model.episode_losses[-100:]) if model.episode_losses else 0

        # Reset simulator for new episode
        draft_simulator = AuctionDraftSimulator(year=2024, rl_model=model)

        # Run draft simulation
        draft_completed, draft_results, draft_reward = draft_simulator.simulate_draft()
        episode_draft_rewards.append(draft_reward)

        if not draft_completed:
            logger.warning(f'Draft not completed for episode {episode}')
            continue

        # Simulate season for additional reward
        season_simulator = SeasonSimulator(draft_results=draft_results, year=2024)
        season_simulator.simulate_season()
        standings = season_simulator.get_standings()
        weekly_results = season_simulator.get_weekly_results()

        # Calculate rewards
        season_reward = calculate_reward(draft_results, standings, weekly_results, draft_simulator.rl_team_name)
        episode_season_rewards.append(season_reward)

        total_reward = draft_reward + season_reward
        episode_rewards.append(total_reward)

        # Update model
        model.update(season_reward)

        # Calculate metrics
        episode_end_loss = np.mean(model.episode_losses[-100:])
        avg_total_reward = np.mean(episode_rewards[-100:])
        avg_draft_reward = np.mean(episode_draft_rewards[-100:])
        avg_season_reward = np.mean(episode_season_rewards[-100:])

        # Log detailed metrics every episode
        logger.info(
            f'Episode {episode:4d} - '
            f'Loss: {episode_end_loss:8.2f} -> {episode_start_loss:8.2f} | '
            f'Running Reward: {model.running_reward:8.2f} | '
            f'Avg Rewards - Total: {avg_total_reward:6.2f}, '
            f'Draft: {avg_draft_reward:6.2f}, '
            f'Season: {avg_season_reward:6.2f}'
        )

        # Save checkpoint with metrics
        if episode % checkpoint_frequency == 0:
            checkpoint_path = f'checkpoint_{episode}.pt'
            metrics = {
                'episode': episode,
                'running_reward': model.running_reward,
                'avg_total_reward': avg_total_reward,
                'avg_draft_reward': avg_draft_reward,
                'avg_season_reward': avg_season_reward,
                'recent_losses': model.episode_losses[-100:],
            }
            model.save_checkpoint(checkpoint_path, metrics)

    return model

if __name__ == "__main__":
    # Create model with checkpoints directory
    model = RLModel()

    # Train model
    train_rl_model(model, None, None, num_episodes=1000, checkpoint_frequency=10)
