"""
Behavioral Cloning (BC) reference model for self-play Stage 2.

BCReferenceModel shares the same MLP architecture as AuctionDraftPolicy but
is trained with MSE loss on historical bid fractions from ESPN draft data.
It serves two purposes:

1. Cold-start the opponent pool: Phase 4 can begin with BC-trained opponents
   rather than random/phase-3 checkpoints, providing more human-realistic
   behavior from the start.

2. Auxiliary reward signal: The BC log-probability under the reference policy
   can be added to mid_draft_reward() to keep the PPO policy from drifting
   into degenerate bidding strategies.

Architecture: 72 → 256 → 128 → 1 (sigmoid output = bid fraction ∈ (0, 1))
Loss: MSE(predicted_fraction, historical_bid / budget_at_time)
"""

from __future__ import annotations

import logging
from pathlib import Path

import torch
import torch.nn as nn

from ffai.rl.state_builder import STATE_DIM

logger = logging.getLogger(__name__)


def _mlp_block(in_dim: int, out_dim: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(in_dim, out_dim),
        nn.LayerNorm(out_dim),
        nn.ReLU(),
    )


class BCReferenceModel(nn.Module):
    """
    Behavioral cloning reference model trained on historical ESPN draft picks.

    Predicts a bid fraction ∈ (0, 1) given a 72-dim state observation.
    The fraction is calibrated to historical bid_amount / budget_at_time.

    Args:
        hidden_size: width of the first hidden layer (matches AuctionDraftPolicy)
        hidden_size2: width of the second hidden layer
    """

    def __init__(self, hidden_size: int = 256, hidden_size2: int = 128):
        super().__init__()

        self.encoder = nn.Sequential(
            _mlp_block(STATE_DIM, hidden_size),
            _mlp_block(hidden_size, hidden_size2),
        )
        self.head = nn.Linear(hidden_size2, 1)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """
        Args:
            obs: (batch, STATE_DIM) float tensor

        Returns:
            bid_fraction: (batch, 1) tensor ∈ (0, 1)
        """
        x = obs.view(obs.shape[0], -1).float()
        hidden = self.encoder(x)
        return torch.sigmoid(self.head(hidden))

    def log_prob(self, obs: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """
        Approximate log-probability of an action under a Gaussian centered at
        the BC prediction. Used for the auxiliary PPO reward signal.

        Args:
            obs: (batch, STATE_DIM) float tensor
            action: (batch, 1) bid fraction actually taken by the PPO policy

        Returns:
            log_prob: (batch,) tensor
        """
        mean = self.forward(obs)  # (batch, 1)
        # Fixed std for BC reference (not learned — just for reward scaling)
        std = torch.tensor(0.1, device=obs.device)
        dist = torch.distributions.Normal(mean, std)
        return dist.log_prob(action).squeeze(-1)

    def save(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.state_dict(), path)
        logger.info(f"BCReferenceModel saved to {path}")

    @classmethod
    def load(cls, path: Path, device: str = "cpu") -> "BCReferenceModel":
        model = cls()
        state_dict = torch.load(path, map_location=device, weights_only=True)
        model.load_state_dict(state_dict)
        model.eval()
        logger.info(f"BCReferenceModel loaded from {path}")
        return model

    def as_checkpoint_policy(self, path: Path, device: str = "cpu"):
        """
        Export this BC model as a LoadedCheckpointPolicy for use in OpponentPool.

        Saves the model state dict in the same format as AuctionDraftPolicy
        checkpoints so it can be loaded by LoadedCheckpointPolicy directly.
        The BC model's head weights are copied to AuctionDraftPolicy's mean_head
        and a zeroed value_head is added.

        Args:
            path: where to save the checkpoint
            device: torch device

        Returns:
            LoadedCheckpointPolicy wrapping the converted checkpoint
        """
        from ffai.rl.puffer_policy import AuctionDraftPolicy
        from ffai.rl.opponent_pool import LoadedCheckpointPolicy

        policy = AuctionDraftPolicy(env=None)

        # Copy encoder weights
        policy.encoder.load_state_dict(self.encoder.state_dict())

        # Copy head weights (BC head → policy mean_head)
        with torch.no_grad():
            policy.mean_head.weight.copy_(self.head.weight)
            policy.mean_head.bias.copy_(self.head.bias)

        # Save in AuctionDraftPolicy format
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(policy.state_dict(), path)
        logger.info(f"BCReferenceModel exported as checkpoint to {path}")

        return LoadedCheckpointPolicy(path, device=device)
