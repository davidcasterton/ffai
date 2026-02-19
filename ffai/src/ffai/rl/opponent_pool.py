"""
Checkpoint pool for self-play training (Phase 4).

OpponentPool manages a collection of past RL policy checkpoints and samples
them to provide learned opponent behavior in place of the heuristic bidder.

LoadedCheckpointPolicy wraps a saved AuctionDraftPolicy and provides
get_bid(sim, team_name, player, current_bid) → float for simulator integration.

Usage:
    pool = OpponentPool(heuristic_fraction=0.3, max_pool_size=10)
    pool.add_checkpoint(Path("checkpoints/puffer/phase3_final.pt"))

    # At episode start, sample one policy per opponent slot
    opponent_policies = pool.sample_opponents(n=11)
    sim._opponent_policies = opponent_policies
"""

from __future__ import annotations

import logging
import random
from pathlib import Path
from typing import Optional

import torch

logger = logging.getLogger(__name__)


class LoadedCheckpointPolicy:
    """Wraps a loaded AuctionDraftPolicy checkpoint for use as a sim opponent."""

    def __init__(self, path: Path, device: str = "cpu"):
        from ffai.rl.puffer_policy import AuctionDraftPolicy

        self.path = path
        self.device = device
        # AuctionDraftPolicy only uses `env` for LSTMWrapper metadata; None is safe.
        self.policy = AuctionDraftPolicy(env=None)
        state_dict = torch.load(path, map_location=device, weights_only=True)
        self.policy.load_state_dict(state_dict)
        self.policy.eval()
        logger.info(f"Loaded checkpoint policy from {path}")

    def get_bid(self, sim, team_name: str, player: dict, current_bid: float) -> float:
        """
        Compute max bid for team_name given current draft state.

        Calls sim.get_state_for(team_name) to build a perspective-correct
        observation, runs it through the policy, and returns the resulting
        max_bid in dollars.

        Returns:
            max_bid (float): always >= 1.0, <= team's remaining budget.
        """
        from ffai.rl.state_builder import build_state

        state_dict = sim.get_state_for(team_name)
        remaining_budget = float(sim.teams[team_name]["current_budget"])
        feature_store = getattr(sim, "feature_store", None)

        obs = build_state(
            state_dict,
            team_name=team_name,
            current_player=player,
            current_bid=current_bid,
            feature_store=feature_store,
            year=sim.year,
        )

        with torch.no_grad():
            obs_t = obs.unsqueeze(0).float().to(self.device)
            hidden = self.policy.encode_observations(obs_t)
            mean = torch.sigmoid(self.policy.mean_head(hidden))
            bid_fraction = float(mean.squeeze().cpu())

        return max(1.0, bid_fraction * remaining_budget)


class OpponentPool:
    """
    Pool of past RL policy checkpoints for self-play.

    sample_policy() returns either a LoadedCheckpointPolicy (drawn uniformly
    from the pool) or None (meaning the heuristic bidder should be used).
    The heuristic fraction controls the probability of returning None.

    Attributes:
        heuristic_fraction: probability of using the heuristic for any given
            opponent slot. Default 0.3 (70% learned, 30% heuristic).
        max_pool_size: max number of checkpoints retained (FIFO eviction).
        device: torch device for policy inference.
    """

    def __init__(
        self,
        heuristic_fraction: float = 0.3,
        max_pool_size: int = 10,
        device: str = "cpu",
    ):
        if not 0.0 <= heuristic_fraction <= 1.0:
            raise ValueError(
                f"heuristic_fraction must be in [0, 1], got {heuristic_fraction}"
            )
        self.heuristic_fraction = heuristic_fraction
        self.max_pool_size = max_pool_size
        self.device = device
        self._pool: list[LoadedCheckpointPolicy] = []

    def add_checkpoint(self, path: Path) -> None:
        """
        Add a checkpoint to the pool. Evicts the oldest entry when full.

        Args:
            path: path to a .pt file containing AuctionDraftPolicy state_dict.
        """
        path = Path(path)
        if not path.exists():
            logger.warning(f"OpponentPool.add_checkpoint: {path} not found, skipping")
            return
        policy = LoadedCheckpointPolicy(path, device=self.device)
        self._pool.append(policy)
        if len(self._pool) > self.max_pool_size:
            evicted = self._pool.pop(0)
            logger.debug(f"OpponentPool: evicted {evicted.path}")
        logger.info(f"OpponentPool: added {path} (pool size={len(self._pool)})")

    def sample_policy(self) -> Optional[LoadedCheckpointPolicy]:
        """
        Sample one policy for a single opponent slot.

        Returns:
            LoadedCheckpointPolicy if pool is non-empty and random draw exceeds
            heuristic_fraction. None → caller should use heuristic bidding.
        """
        if not self._pool:
            return None
        if random.random() < self.heuristic_fraction:
            return None
        return random.choice(self._pool)

    def sample_opponents(self, n: int = 11) -> dict:
        """
        Sample n opponent policies for a full draft episode.

        Returns a dict mapping simulator slot names ("Team 2".."Team {n+1}")
        to either a LoadedCheckpointPolicy or None (→ heuristic).
        """
        return {f"Team {i + 2}": self.sample_policy() for i in range(n)}

    @property
    def size(self) -> int:
        return len(self._pool)

    @property
    def is_empty(self) -> bool:
        return len(self._pool) == 0
