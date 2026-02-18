"""
GAE Rollout Buffer for PPO training.

Accumulates full episode trajectories (one complete auction draft + season),
then computes Generalized Advantage Estimation (GAE) before PPO updates.

The key fix over the prior implementation: we collect the entire episode
BEFORE any gradient update, enabling proper credit assignment across
multi-step draft decisions.
"""

import numpy as np
import torch
from typing import List, Optional, Tuple, Generator


class RolloutBuffer:
    """
    Stores one or more episode rollouts and computes GAE advantages.

    Usage:
        buffer = RolloutBuffer()
        # During episode:
        buffer.add(state, action, log_prob, reward, value, done)
        # After episode:
        buffer.finish_episode(last_value)
        # After collecting enough episodes:
        for batch in buffer.get_batches(batch_size):
            # PPO update using batch
        buffer.clear()
    """

    def __init__(self, gamma: float = 0.99, gae_lambda: float = 0.95):
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self._clear_buffers()

    def _clear_buffers(self):
        self.states: List[torch.Tensor] = []
        self.actions: List[torch.Tensor] = []          # bid amounts (continuous)
        self.log_probs: List[torch.Tensor] = []        # log π(a|s)
        self.rewards: List[float] = []
        self.values: List[float] = []                  # V(s) from critic
        self.dones: List[bool] = []

        self.advantages: Optional[np.ndarray] = None
        self.returns: Optional[np.ndarray] = None

        self._episode_start_idx = 0

    def add(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
        log_prob: torch.Tensor,
        reward: float,
        value: float,
        done: bool,
    ) -> None:
        """Add one step's data to the buffer."""
        self.states.append(state.detach().cpu())
        self.actions.append(action.detach().cpu())
        self.log_probs.append(log_prob.detach().cpu())
        self.rewards.append(float(reward))
        self.values.append(float(value))
        self.dones.append(bool(done))

    def finish_episode(self, last_value: float = 0.0) -> None:
        """
        Compute GAE advantages for the completed episode.

        Args:
            last_value: V(s_T) for bootstrap. 0 if episode ended naturally.
        """
        episode_slice = slice(self._episode_start_idx, len(self.rewards))
        rewards = np.array(self.rewards[episode_slice])
        values = np.array(self.values[episode_slice])
        dones = np.array(self.dones[episode_slice], dtype=np.float32)

        n = len(rewards)
        advantages = np.zeros(n, dtype=np.float32)
        returns = np.zeros(n, dtype=np.float32)

        gae = 0.0
        next_value = last_value

        for t in reversed(range(n)):
            mask = 1.0 - dones[t]
            delta = rewards[t] + self.gamma * next_value * mask - values[t]
            gae = delta + self.gamma * self.gae_lambda * mask * gae
            advantages[t] = gae
            next_value = values[t]

        returns = advantages + values

        if self.advantages is None:
            self.advantages = advantages
            self.returns = returns
        else:
            self.advantages = np.concatenate([self.advantages, advantages])
            self.returns = np.concatenate([self.returns, returns])

        self._episode_start_idx = len(self.rewards)

    def get_batches(self, batch_size: int) -> Generator[dict, None, None]:
        """
        Yield mini-batches for PPO updates.

        Normalizes advantages before yielding (zero mean, unit variance).
        """
        assert self.advantages is not None, "Call finish_episode() before get_batches()"

        n = len(self.states)
        assert n == len(self.advantages), "Mismatch between states and computed advantages"

        # Normalize advantages
        adv = self.advantages.copy()
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)

        states = torch.stack(self.states)             # (N, state_dim)
        actions = torch.stack(self.actions)           # (N, action_dim)
        log_probs = torch.stack(self.log_probs)       # (N,)
        returns = torch.from_numpy(self.returns).float()
        advantages = torch.from_numpy(adv).float()

        # Shuffle
        indices = np.random.permutation(n)
        for start in range(0, n, batch_size):
            batch_idx = indices[start:start + batch_size]
            yield {
                'states': states[batch_idx],
                'actions': actions[batch_idx],
                'log_probs': log_probs[batch_idx],
                'returns': returns[batch_idx],
                'advantages': advantages[batch_idx],
            }

    def clear(self) -> None:
        """Clear all stored data after a PPO update cycle."""
        self._clear_buffers()

    def __len__(self) -> int:
        return len(self.states)

    @property
    def ready(self) -> bool:
        """True if there's data ready for training."""
        return len(self.states) > 0 and self.advantages is not None
