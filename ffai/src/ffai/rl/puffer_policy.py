"""
AuctionDraftPolicy — Gaussian policy for PufferLib PPO.

Follows the pufferlib.models.Default pattern:
  encode_observations(obs) → hidden
  decode_actions(hidden)   → (Normal(mean, std), value)
  forward_eval(obs, state) → (Normal, value)

PufferLib's pytorch.sample_logits() handles torch.distributions.Normal natively
(pufferlib/pytorch.py lines 189-201), so no custom training loop is needed.

Architecture mirrors AuctionDraftActor + AuctionDraftCritic from ppo_agent.py:
  - LayerNorm after each Linear
  - sigmoid mean head → bid fraction ∈ (0, 1)
  - global learned log_std parameter
  - separate value head
"""

import torch
import torch.nn as nn
from torch.distributions import Normal

from ffai.rl.state_builder import STATE_DIM

LOG_STD_MIN = -4.0
LOG_STD_MAX = 1.0


def _mlp_block(in_dim: int, out_dim: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(in_dim, out_dim),
        nn.LayerNorm(out_dim),
        nn.ReLU(),
    )


class AuctionDraftPolicy(nn.Module):
    """
    Gaussian policy + value function for PufferLib PPO.

    forward_eval() returns (Normal distribution, value tensor) which
    pufferlib.pytorch.sample_logits() can consume directly.
    """

    def __init__(self, env, hidden_size: int = 256, hidden_size2: int = 128):
        super().__init__()
        self.hidden_size = hidden_size  # needed by LSTMWrapper if used

        # Shared encoder
        self.encoder = nn.Sequential(
            _mlp_block(STATE_DIM, hidden_size),
            _mlp_block(hidden_size, hidden_size2),
        )

        # Policy head: outputs mean of bid fraction ∈ (0,1)
        self.mean_head = nn.Linear(hidden_size2, 1)
        # Global log_std (not state-dependent, matches ppo_agent.py design)
        self.log_std = nn.Parameter(torch.zeros(1))

        # Value head
        self.value_head = nn.Linear(hidden_size2, 1)

    def encode_observations(self, observations, state=None):
        """Encode batch of flat observations → hidden representation."""
        x = observations.view(observations.shape[0], -1).float()
        return self.encoder(x)

    def decode_actions(self, hidden):
        """Decode hidden → (Normal distribution, value tensor)."""
        mean = torch.sigmoid(self.mean_head(hidden))  # ∈ (0, 1)
        log_std = self.log_std.clamp(LOG_STD_MIN, LOG_STD_MAX)
        std = log_std.exp().expand_as(mean)
        dist = Normal(mean, std)

        value = self.value_head(hidden)
        return dist, value

    def forward_eval(self, observations, state=None):
        hidden = self.encode_observations(observations, state=state)
        return self.decode_actions(hidden)

    def forward(self, observations, state=None):
        return self.forward_eval(observations, state)
