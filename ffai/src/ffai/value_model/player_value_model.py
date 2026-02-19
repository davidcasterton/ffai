"""
PlayerValueModel — two-head supervised model for player value estimation.

Outputs:
  - expected_season_points: projected total fantasy points for the season
  - fair_auction_dollar: fair auction dollar value (PAR/VORP-scaled)

Architecture:
  player_embedding:    Embedding(num_players+1, 256)
  position_embedding:  Embedding(num_positions+1, 64)
  numerical_encoder:   MLP([14] -> [128])  # see preprocessor.py for feature layout
  combined:            concat(256 + 64 + 128) = 448 dims
  points_head:         Linear(448, 1) -> expected season pts
  value_head:          Linear(448, 1) -> fair auction $
"""

import torch
import torch.nn as nn
from pathlib import Path
from typing import Optional, Tuple
import logging

logger = logging.getLogger(__name__)

NUM_NUMERICAL_FEATURES = 14  # matches preprocessor.py 14-dim feature vector


class PlayerValueModel(nn.Module):
    def __init__(
        self,
        num_players: int,
        num_positions: int = 7,  # QB, RB, WR, TE, K, D/ST, UNKNOWN
        player_embedding_dim: int = 256,
        position_embedding_dim: int = 64,
        numerical_hidden_dim: int = 128,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.num_players = num_players
        self.num_positions = num_positions

        # Embeddings
        self.player_embedding = nn.Embedding(num_players + 1, player_embedding_dim, padding_idx=0)
        self.position_embedding = nn.Embedding(num_positions + 1, position_embedding_dim, padding_idx=0)

        # Numerical feature encoder
        self.numerical_encoder = nn.Sequential(
            nn.Linear(NUM_NUMERICAL_FEATURES, numerical_hidden_dim),
            nn.LayerNorm(numerical_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(numerical_hidden_dim, numerical_hidden_dim),
            nn.ReLU(),
        )

        combined_dim = player_embedding_dim + position_embedding_dim + numerical_hidden_dim

        # Shared trunk
        self.trunk = nn.Sequential(
            nn.Linear(combined_dim, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.ReLU(),
        )

        # Two output heads
        self.points_head = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )
        self.value_head = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Softplus(),  # Ensure positive dollar values
        )

        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0, std=0.01)

    def forward(
        self,
        player_ids: torch.Tensor,
        position_ids: torch.Tensor,
        numerical_features: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            player_ids: (batch,) long tensor
            position_ids: (batch,) long tensor
            numerical_features: (batch, NUM_NUMERICAL_FEATURES) float tensor

        Returns:
            points_hat: (batch, 1) — predicted season points
            dollar_hat: (batch, 1) — predicted fair auction value ($)
        """
        player_emb = self.player_embedding(player_ids)       # (B, 256)
        position_emb = self.position_embedding(position_ids)  # (B, 64)
        numerical_out = self.numerical_encoder(numerical_features)  # (B, 128)

        combined = torch.cat([player_emb, position_emb, numerical_out], dim=-1)  # (B, 448)
        trunk_out = self.trunk(combined)  # (B, 128)

        points_hat = self.points_head(trunk_out)  # (B, 1)
        dollar_hat = self.value_head(trunk_out)   # (B, 1)

        return points_hat, dollar_hat

    def predict_player(
        self,
        player_id: int,
        position_id: int,
        numerical_features: torch.Tensor,
        device: str = 'cpu',
    ) -> Tuple[float, float]:
        """
        Single-player inference. Returns (predicted_points, predicted_dollar_value).
        """
        self.eval()
        with torch.no_grad():
            pid = torch.tensor([player_id], dtype=torch.long, device=device)
            posid = torch.tensor([position_id], dtype=torch.long, device=device)
            num_feat = numerical_features.unsqueeze(0).to(device)
            pts, dollar = self(pid, posid, num_feat)
        return pts.item(), dollar.item()

    def save(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            'state_dict': self.state_dict(),
            'num_players': self.num_players,
            'num_positions': self.num_positions,
        }, path)
        logger.info(f"PlayerValueModel saved to {path}")

    @classmethod
    def load(cls, path: Path, device: str = 'cpu') -> 'PlayerValueModel':
        path = Path(path)
        checkpoint = torch.load(path, map_location=device)
        model = cls(
            num_players=checkpoint['num_players'],
            num_positions=checkpoint['num_positions'],
        )
        model.load_state_dict(checkpoint['state_dict'])
        model.to(device)
        model.eval()
        logger.info(f"PlayerValueModel loaded from {path}")
        return model
