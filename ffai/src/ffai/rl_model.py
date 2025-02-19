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

logger = logging.getLogger(__name__)

class RLModel(nn.Module):
    def __init__(self, checkpoint_dir=None):
        super(RLModel, self).__init__()

        self.checkpoint_dir = Path(checkpoint_dir) if checkpoint_dir else Path('checkpoints')
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Calculate state size based on league settings
        self.state_size = self.calculate_state_size()

        # Initialize network and training parameters
        self.initialize_network()
        self.initialize_training_params()

    def initialize_network(self):
        """Initialize neural network architecture"""
        self.fc1 = nn.Linear(self.state_size, 256)
        self.bn1 = nn.BatchNorm1d(256)
        self.fc2 = nn.Linear(256, 128)
        self.bn2 = nn.BatchNorm1d(128)
        self.fc3 = nn.Linear(128, 64)
        self.bn3 = nn.BatchNorm1d(64)
        self.value = nn.Linear(64, 1)

    def initialize_training_params(self):
        """Initialize training parameters"""
        self.memory = deque(maxlen=10000)
        self.gamma = 0.95
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.batch_size = 32

        self.optimizer = optim.Adam(self.parameters(), lr=self.learning_rate)
        self.criterion = nn.MSELoss()

        # Training statistics
        self.episodes_trained = 0
        self.total_rewards = []
        self.avg_losses = []

    def calculate_state_size(self):
        """Calculate state size based on league settings"""
        # Use path relative to the package
        settings_path = Path(__file__).parent / "data/raw/league_settings_770280_2024.json"
        with open(settings_path) as f:
            settings = json.load(f)

        num_roster_slots = sum(settings["position_slot_counts"].values())

        return (
            1 +    # RL team budget
            11 +   # Opponent budgets
            1 +    # Draft turn
            12 * num_roster_slots +  # Predicted points per roster slot per team
            12 * num_roster_slots +  # Auction $ per roster slot per team
            1 +    # Nominated player ID
            1 +    # Nominated player predicted value
            1      # Nominated player predicted points
        )

    def forward(self, x):
        """Forward pass through the network"""
        x = torch.relu(self.bn1(self.fc1(x)))
        x = torch.relu(self.bn2(self.fc2(x)))
        x = torch.relu(self.bn3(self.fc3(x)))
        return self.value(x)

    def get_state_tensor(self, draft_state, player):
        """Convert draft state to tensor format with safety checks"""
        try:
            state = []

            # Ensure budget values are valid
            rl_budget = max(0, float(draft_state.get('rl_team_budget', 0)))
            state.append(rl_budget)

            # Add opponent budgets with safety
            opp_budgets = draft_state.get('opponent_budgets', [])
            state.extend([max(0, float(b)) for b in opp_budgets[:11]])  # Limit to 11 opponents

            # Add player info safely
            state.append(max(0, float(player.get('projected_points', 0))))

            # Add position encoding (fixed size)
            pos_encoding = {
                'QB': [1,0,0,0,0],
                'RB': [0,1,0,0,0],
                'WR': [0,0,1,0,0],
                'TE': [0,0,0,1,0],
                'D/ST': [0,0,0,0,1]
            }.get(player.get('position', ''), [0,0,0,0,0])
            state.extend(pos_encoding)

            # Add team state info with limits
            if draft_state.get('predicted_points_per_slot'):
                teams = draft_state.get('teams', [])[:12]  # Limit to 12 teams
                for team in teams:
                    points = draft_state['predicted_points_per_slot'].get(team, [])
                    spent = draft_state['auction_spent_per_slot'].get(team, [])
                    # Limit to 16 slots per team
                    state.extend((points + [0] * 16)[:16])
                    state.extend((spent + [0] * 16)[:16])

            return torch.tensor(state, dtype=torch.float32)
        except Exception as e:
            logger.error(f"Error creating state tensor: {str(e)}")
            # Return safe default tensor
            return torch.zeros(self.state_size, dtype=torch.float32)

    def get_bid_action(self, state, available_budget):
        """Get bid amount with safety checks"""
        try:
            if available_budget < 1:
                return 0

            if random.random() < self.epsilon:
                # Safer random bid
                max_bid = min(available_budget, 50)  # Cap random bids
                return random.randint(1, max_bid)

            with torch.no_grad():
                bid_value = self.forward(state)
                # Ensure reasonable bid range
                return min(max(1, int(bid_value.item())),
                          min(available_budget, 100))  # Cap network bids
        except Exception as e:
            logger.error(f"Error getting bid action: {str(e)}")
            return 0

    def remember(self, state, action, reward, next_state, done):
        """Store experience in replay memory"""
        self.memory.append((state, action, reward, next_state, done))

    def replay(self):
        """Train on batch from replay memory"""
        if len(self.memory) < self.batch_size:
            return

        minibatch = random.sample(self.memory, self.batch_size)

        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                target = reward + self.gamma * self.forward(next_state).item()

            current_prediction = self.forward(state)
            target_f = current_prediction.clone()
            target_f[0] = target

            self.optimizer.zero_grad()
            loss = self.criterion(current_prediction, target_f)
            loss.backward()
            self.optimizer.step()

        # Decay epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def save_checkpoint(self, episode, reward, loss):
        """Save model checkpoint with training state"""
        checkpoint = {
            'episode': episode,
            'model_state_dict': self.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'memory': list(self.memory),
            'episodes_trained': self.episodes_trained,
            'total_rewards': self.total_rewards,
            'avg_losses': self.avg_losses,
            'reward': reward,
            'loss': loss
        }

        checkpoint_path = self.checkpoint_dir / f'model_checkpoint_ep{episode}.pt'
        torch.save(checkpoint, checkpoint_path)

        # Save latest checkpoint reference
        latest_path = self.checkpoint_dir / 'latest.txt'
        with open(latest_path, 'w') as f:
            f.write(str(checkpoint_path))

        print(f"Saved checkpoint: {checkpoint_path}")

    def load_checkpoint(self, checkpoint_path=None):
        """Load model checkpoint and training state"""
        if checkpoint_path is None:
            # Try to load latest checkpoint
            latest_path = self.checkpoint_dir / 'latest.txt'
            if not latest_path.exists():
                print("No checkpoint found")
                return False

            with open(latest_path, 'r') as f:
                checkpoint_path = Path(f.read().strip())

        if not checkpoint_path.exists():
            print(f"Checkpoint not found: {checkpoint_path}")
            return False

        checkpoint = torch.load(checkpoint_path)

        # Load model and optimizer state
        self.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        # Restore training state
        self.epsilon = checkpoint['epsilon']
        self.memory = deque(checkpoint['memory'], maxlen=10000)
        self.episodes_trained = checkpoint['episodes_trained']
        self.total_rewards = checkpoint['total_rewards']
        self.avg_losses = checkpoint['avg_losses']

        print(f"Loaded checkpoint from episode {checkpoint['episode']}")
        print(f"Epsilon: {self.epsilon:.3f}")
        print(f"Last reward: {checkpoint['reward']}")
        print(f"Last loss: {checkpoint['loss']:.6f}")

        return True

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
    model = RLModel(checkpoint_dir='checkpoints/auction_draft')

    # Load latest checkpoint if it exists
    model.load_checkpoint()

    # Train model
    train_rl_model(model, None, None, num_episodes=1000, checkpoint_frequency=10)
