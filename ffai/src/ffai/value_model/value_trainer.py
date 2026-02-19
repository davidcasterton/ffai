"""
Training loop for PlayerValueModel (supervised, two-head regression).

Trains on historical ESPN draft data:
  - Input: player embeddings + position embeddings + numerical features
  - Target 1: actual season total_points (points_head)
  - Target 2: VORP-derived fair auction dollar value (value_head)
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from pathlib import Path
from typing import List, Optional, Tuple
import logging

from ffai.value_model.player_value_model import PlayerValueModel

logger = logging.getLogger(__name__)


class PlayerDataset(Dataset):
    def __init__(
        self,
        player_ids: np.ndarray,
        position_ids: np.ndarray,
        numerical_features: np.ndarray,
        target_points: np.ndarray,
        target_dollar: np.ndarray,
    ):
        self.player_ids = torch.from_numpy(player_ids).long()
        self.position_ids = torch.from_numpy(position_ids).long()
        self.numerical_features = torch.from_numpy(numerical_features).float()
        self.target_points = torch.from_numpy(target_points).float()
        self.target_dollar = torch.from_numpy(target_dollar).float()

    def __len__(self):
        return len(self.player_ids)

    def __getitem__(self, idx):
        return {
            'player_id': self.player_ids[idx],
            'position_id': self.position_ids[idx],
            'numerical': self.numerical_features[idx],
            'target_points': self.target_points[idx],
            'target_dollar': self.target_dollar[idx],
        }


class ValueModelTrainer:
    def __init__(
        self,
        model: PlayerValueModel,
        learning_rate: float = 1e-3,
        batch_size: int = 64,
        num_epochs: int = 100,
        early_stopping_patience: int = 10,
        checkpoint_dir: Optional[Path] = None,
        device: str = 'cpu',
    ):
        self.model = model.to(device)
        self.device = device
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.patience = early_stopping_patience
        self.checkpoint_dir = Path(checkpoint_dir) if checkpoint_dir else None

        self.optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, patience=5, factor=0.5
        )
        self.points_loss_fn = nn.MSELoss()
        self.dollar_loss_fn = nn.MSELoss()

        if self.checkpoint_dir:
            self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    def _run_epoch(self, loader: DataLoader, train: bool = True) -> Tuple[float, float, float]:
        """Run one epoch. Returns (total_loss, points_loss, dollar_loss)."""
        self.model.train(train)
        total_loss = 0.0
        pts_loss_total = 0.0
        dollar_loss_total = 0.0

        with torch.set_grad_enabled(train):
            for batch in loader:
                player_ids = batch['player_id'].to(self.device)
                position_ids = batch['position_id'].to(self.device)
                numerical = batch['numerical'].to(self.device)
                target_pts = batch['target_points'].to(self.device).unsqueeze(-1)
                target_dollar = batch['target_dollar'].to(self.device).unsqueeze(-1)

                pts_hat, dollar_hat = self.model(player_ids, position_ids, numerical)

                pts_loss = self.points_loss_fn(pts_hat, target_pts)
                dollar_loss = self.dollar_loss_fn(dollar_hat, target_dollar)
                loss = pts_loss + dollar_loss

                if train:
                    self.optimizer.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                    self.optimizer.step()

                total_loss += loss.item()
                pts_loss_total += pts_loss.item()
                dollar_loss_total += dollar_loss.item()

        n = len(loader)
        return total_loss / n, pts_loss_total / n, dollar_loss_total / n

    def train(
        self,
        train_data: Tuple[np.ndarray, ...],
        val_data: Tuple[np.ndarray, ...],
    ) -> dict:
        """
        Train the model.

        Args:
            train_data: (player_ids, position_ids, numerical, target_points, target_dollar)
            val_data: same format

        Returns:
            metrics dict with training history
        """
        train_dataset = PlayerDataset(*train_data)
        val_dataset = PlayerDataset(*val_data)

        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False)

        best_val_loss = float('inf')
        patience_counter = 0
        history = {'train_loss': [], 'val_loss': [], 'val_pts_rmse': [], 'val_dollar_mae': []}

        for epoch in range(self.num_epochs):
            train_loss, _, _ = self._run_epoch(train_loader, train=True)
            val_loss, val_pts_loss, val_dollar_loss = self._run_epoch(val_loader, train=False)

            self.scheduler.step(val_loss)

            val_pts_rmse = val_pts_loss ** 0.5
            val_dollar_mae = val_dollar_loss ** 0.5

            history['train_loss'].append(train_loss)
            history['val_loss'].append(val_loss)
            history['val_pts_rmse'].append(val_pts_rmse)
            history['val_dollar_mae'].append(val_dollar_mae)

            logger.info(
                f"Epoch {epoch+1:3d}/{self.num_epochs} | "
                f"train_loss={train_loss:.4f} | "
                f"val_loss={val_loss:.4f} | "
                f"pts_RMSE={val_pts_rmse:.2f} | "
                f"dollar_MAE=${val_dollar_mae:.2f}"
            )

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                if self.checkpoint_dir:
                    self.model.save(self.checkpoint_dir / 'best_model.pt')
                    logger.info(f"  Saved best model (val_loss={val_loss:.4f})")
            else:
                patience_counter += 1
                if patience_counter >= self.patience:
                    logger.info(f"Early stopping at epoch {epoch+1} (patience={self.patience})")
                    break

        # Load best weights
        if self.checkpoint_dir and (self.checkpoint_dir / 'best_model.pt').exists():
            self.model = PlayerValueModel.load(self.checkpoint_dir / 'best_model.pt', device=self.device)

        return history

    def evaluate(self, test_data: Tuple[np.ndarray, ...]) -> dict:
        """Evaluate on test set. Returns RMSE for points and MAE for dollar value."""
        test_dataset = PlayerDataset(*test_data)
        test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False)

        all_pts_hat, all_pts_true = [], []
        all_dollar_hat, all_dollar_true = [], []

        self.model.eval()
        with torch.no_grad():
            for batch in test_loader:
                player_ids = batch['player_id'].to(self.device)
                position_ids = batch['position_id'].to(self.device)
                numerical = batch['numerical'].to(self.device)

                pts_hat, dollar_hat = self.model(player_ids, position_ids, numerical)
                all_pts_hat.extend(pts_hat.squeeze().cpu().tolist())
                all_pts_true.extend(batch['target_points'].tolist())
                all_dollar_hat.extend(dollar_hat.squeeze().cpu().tolist())
                all_dollar_true.extend(batch['target_dollar'].tolist())

        pts_rmse = np.sqrt(np.mean((np.array(all_pts_hat) - np.array(all_pts_true)) ** 2))
        dollar_mae = np.mean(np.abs(np.array(all_dollar_hat) - np.array(all_dollar_true)))

        metrics = {
            'points_rmse': pts_rmse,
            'dollar_mae': dollar_mae,
        }
        logger.info(f"Test set: points_RMSE={pts_rmse:.2f}, dollar_MAE=${dollar_mae:.2f}")
        return metrics
