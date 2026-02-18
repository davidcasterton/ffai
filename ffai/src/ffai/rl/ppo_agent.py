"""
Actor-Critic PPO Agent for the auction draft bidding problem.

Replaces the broken rl_model.py (DQN with zero gradients and wrong action space).

Key fixes:
1. Continuous Gaussian policy (not discrete DQN) — bid amount is a real number
2. Proper PPO clip loss with mini-batch updates
3. Full episode rollout collection via RolloutBuffer before any gradient update
4. Separate actor and critic networks
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal
import numpy as np
from pathlib import Path
from typing import Optional, Tuple
import logging

from ffai.rl.state_builder import STATE_DIM
from ffai.rl.replay_buffer import RolloutBuffer

logger = logging.getLogger(__name__)


def _mlp(in_dim: int, hidden_dims: list, out_dim: int, activation=nn.ReLU) -> nn.Sequential:
    layers = []
    prev = in_dim
    for h in hidden_dims:
        layers.extend([nn.Linear(prev, h), nn.LayerNorm(h), activation()])
        prev = h
    layers.append(nn.Linear(prev, out_dim))
    return nn.Sequential(*layers)


class AuctionDraftActor(nn.Module):
    """
    Policy network. Outputs:
    - bid_mean: mean of Gaussian bid distribution (scaled to [0, budget])
    - bid_log_std: log standard deviation (learned parameter)
    - nomination_logits: scores over available players for nomination
    """

    LOG_STD_MIN = -4.0
    LOG_STD_MAX = 1.0

    def __init__(self, state_dim: int = STATE_DIM, hidden_dims: list = None):
        super().__init__()
        hidden_dims = hidden_dims or [256, 128]
        self.state_encoder = _mlp(state_dim, hidden_dims, hidden_dims[-1])
        self.bid_mean_head = nn.Linear(hidden_dims[-1], 1)
        self.bid_log_std = nn.Parameter(torch.zeros(1))  # global learned std

    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            state: (batch, STATE_DIM)
        Returns:
            bid_mean: (batch, 1) ∈ [0, 1] (fraction of budget)
            bid_std: (batch, 1) > 0
        """
        encoded = self.state_encoder(state)
        bid_mean = torch.sigmoid(self.bid_mean_head(encoded))  # [0, 1]
        bid_log_std = self.bid_log_std.clamp(self.LOG_STD_MIN, self.LOG_STD_MAX)
        bid_std = bid_log_std.exp().expand_as(bid_mean)
        return bid_mean, bid_std

    def get_action(
        self,
        state: torch.Tensor,
        deterministic: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Sample a bid fraction and compute its log probability.

        Returns:
            action: (batch, 1) bid fraction ∈ [0, 1]
            log_prob: (batch,) log π(a|s)
        """
        bid_mean, bid_std = self(state)
        dist = Normal(bid_mean, bid_std)

        if deterministic:
            action = bid_mean
        else:
            action = dist.sample()

        action = action.clamp(0.0, 1.0)
        log_prob = dist.log_prob(action).squeeze(-1)
        return action, log_prob

    def evaluate_action(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Evaluate log_prob and entropy for a stored action (used in PPO update)."""
        bid_mean, bid_std = self(state)
        dist = Normal(bid_mean, bid_std)
        log_prob = dist.log_prob(action).squeeze(-1)
        entropy = dist.entropy().squeeze(-1)
        return log_prob, entropy


class AuctionDraftCritic(nn.Module):
    """Value function V(s) for advantage estimation."""

    def __init__(self, state_dim: int = STATE_DIM, hidden_dims: list = None):
        super().__init__()
        hidden_dims = hidden_dims or [256, 128]
        self.net = _mlp(state_dim, hidden_dims, 1)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """Returns V(s) of shape (batch, 1)."""
        return self.net(state)


class PPOAgent:
    """
    Proximal Policy Optimization agent for continuous bid amounts.

    Workflow per training cycle:
    1. Run N episodes, collecting (s, a, log_π, r, V(s), done) into RolloutBuffer
    2. After each episode: call buffer.finish_episode(last_value)
    3. After collecting enough episodes: call update()
    4. Clear buffer and repeat
    """

    def __init__(
        self,
        state_dim: int = STATE_DIM,
        actor_hidden: list = None,
        critic_hidden: list = None,
        lr_actor: float = 3e-4,
        lr_critic: float = 1e-3,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_epsilon: float = 0.2,
        num_epochs: int = 4,
        batch_size: int = 64,
        entropy_coef: float = 0.01,
        value_coef: float = 0.5,
        checkpoint_dir: Optional[Path] = None,
        device: str = 'cpu',
    ):
        self.gamma = gamma
        self.clip_epsilon = clip_epsilon
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.entropy_coef = entropy_coef
        self.value_coef = value_coef
        self.device = device
        self.checkpoint_dir = Path(checkpoint_dir) if checkpoint_dir else None

        actor_hidden = actor_hidden or [256, 128]
        critic_hidden = critic_hidden or [256, 128]

        self.actor = AuctionDraftActor(state_dim, actor_hidden).to(device)
        self.critic = AuctionDraftCritic(state_dim, critic_hidden).to(device)

        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr_actor)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr_critic)

        self.buffer = RolloutBuffer(gamma=gamma, gae_lambda=gae_lambda)

        self._total_updates = 0
        self._episode_rewards = []

        if self.checkpoint_dir:
            self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Action selection
    # ------------------------------------------------------------------

    def get_bid_action(
        self,
        state: torch.Tensor,
        budget: float,
        min_bid: float = 1.0,
        deterministic: bool = False,
    ) -> Tuple[float, torch.Tensor, float]:
        """
        Sample a bid amount from the policy.

        Args:
            state: (STATE_DIM,) state tensor
            budget: current remaining budget (scales action from [0,1] → [min_bid, budget])
            min_bid: minimum valid bid
            deterministic: use mean action (evaluation mode)

        Returns:
            bid_amount: float dollar bid
            log_prob: log π(a|s) for PPO ratio
            value: V(s) from critic
        """
        self.actor.eval()
        self.critic.eval()
        with torch.no_grad():
            s = state.unsqueeze(0).to(self.device)
            action_frac, log_prob = self.actor.get_action(s, deterministic=deterministic)
            value = self.critic(s).item()

        # Scale action fraction to valid bid range
        bid_amount = float(action_frac.item()) * (budget - min_bid) + min_bid
        bid_amount = max(min_bid, min(bid_amount, budget))

        return bid_amount, action_frac.squeeze(0), value

    def get_value(self, state: torch.Tensor) -> float:
        """Get V(s) estimate."""
        self.critic.eval()
        with torch.no_grad():
            s = state.unsqueeze(0).to(self.device)
            return self.critic(s).item()

    # ------------------------------------------------------------------
    # Buffer interface
    # ------------------------------------------------------------------

    def store(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
        log_prob: torch.Tensor,
        reward: float,
        value: float,
        done: bool,
    ) -> None:
        """Add a step to the rollout buffer."""
        self.buffer.add(state, action, log_prob, reward, value, done)

    def finish_episode(self, last_value: float = 0.0) -> None:
        """Signal end of episode to compute GAE advantages."""
        self.buffer.finish_episode(last_value)

    # ------------------------------------------------------------------
    # PPO update
    # ------------------------------------------------------------------

    def update(self) -> dict:
        """
        Run PPO update on all collected rollout data.

        Must be called after at least one finish_episode().
        Returns dict of training metrics.
        """
        assert self.buffer.ready, "No data in buffer. Collect episodes before calling update()."

        self.actor.train()
        self.critic.train()

        metrics = {
            'actor_loss': [],
            'critic_loss': [],
            'entropy': [],
            'clip_fraction': [],
        }

        for epoch in range(self.num_epochs):
            for batch in self.buffer.get_batches(self.batch_size):
                states = batch['states'].to(self.device)
                actions = batch['actions'].to(self.device)
                old_log_probs = batch['log_probs'].to(self.device)
                returns = batch['returns'].to(self.device)
                advantages = batch['advantages'].to(self.device)

                # Actor update
                new_log_probs, entropy = self.actor.evaluate_action(states, actions)
                ratio = (new_log_probs - old_log_probs).exp()

                surr1 = ratio * advantages
                surr2 = ratio.clamp(1 - self.clip_epsilon, 1 + self.clip_epsilon) * advantages
                actor_loss = -torch.min(surr1, surr2).mean() - self.entropy_coef * entropy.mean()

                self.actor_optimizer.zero_grad()
                actor_loss.backward()
                nn.utils.clip_grad_norm_(self.actor.parameters(), 0.5)
                self.actor_optimizer.step()

                # Critic update
                values = self.critic(states).squeeze(-1)
                critic_loss = nn.functional.mse_loss(values, returns) * self.value_coef

                self.critic_optimizer.zero_grad()
                critic_loss.backward()
                nn.utils.clip_grad_norm_(self.critic.parameters(), 0.5)
                self.critic_optimizer.step()

                # Track metrics
                clip_frac = ((ratio - 1).abs() > self.clip_epsilon).float().mean().item()
                metrics['actor_loss'].append(actor_loss.item())
                metrics['critic_loss'].append(critic_loss.item())
                metrics['entropy'].append(entropy.mean().item())
                metrics['clip_fraction'].append(clip_frac)

        self._total_updates += 1
        self.buffer.clear()

        return {k: float(np.mean(v)) for k, v in metrics.items()}

    # ------------------------------------------------------------------
    # Checkpointing
    # ------------------------------------------------------------------

    def save(self, path: Optional[Path] = None) -> None:
        if path is None and self.checkpoint_dir:
            path = self.checkpoint_dir / f"ppo_checkpoint_{self._total_updates}.pt"
        if path is None:
            logger.warning("No checkpoint path specified, skipping save.")
            return
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            'actor_state_dict': self.actor.state_dict(),
            'critic_state_dict': self.critic.state_dict(),
            'actor_optimizer': self.actor_optimizer.state_dict(),
            'critic_optimizer': self.critic_optimizer.state_dict(),
            'total_updates': self._total_updates,
        }, path)
        logger.info(f"PPO checkpoint saved to {path}")

    def load(self, path: Path) -> None:
        path = Path(path)
        checkpoint = torch.load(path, map_location=self.device)
        self.actor.load_state_dict(checkpoint['actor_state_dict'])
        self.critic.load_state_dict(checkpoint['critic_state_dict'])
        self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer'])
        self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer'])
        self._total_updates = checkpoint.get('total_updates', 0)
        logger.info(f"PPO checkpoint loaded from {path} (update #{self._total_updates})")

    def load_latest(self) -> bool:
        """Load the most recent checkpoint from checkpoint_dir. Returns True if loaded."""
        if not self.checkpoint_dir or not self.checkpoint_dir.exists():
            return False
        checkpoints = sorted(self.checkpoint_dir.glob("ppo_checkpoint_*.pt"),
                             key=lambda p: int(p.stem.split('_')[-1]))
        if checkpoints:
            self.load(checkpoints[-1])
            return True
        return False
