"""
Data preprocessor for fantasy football auction draft training.

Cherry-picks encoding/scaling from favrefignewton and adds PAR/VORP calculations
for use as targets in the supervised PlayerValueModel.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from typing import Dict, List, Optional, Tuple
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

# Positions and their replacement-level thresholds (roster spots across N teams)
# Standard 12-team league: 12 QBs, ~24 RBs, ~24 WRs, 12 TEs, 12 Ks, 12 D/ST
REPLACEMENT_THRESHOLDS = {
    'QB': 12,
    'RB': 24,
    'WR': 24,
    'TE': 12,
    'K': 12,
    'D/ST': 12,
}

POSITIONS = ['QB', 'RB', 'WR', 'TE', 'K', 'D/ST']

# Total auction budget across all teams
TOTAL_BUDGET = 200 * 12  # $2400 for 12 teams


class FantasyDataPreprocessor:
    """
    Preprocesses historical ESPN draft data for training the PlayerValueModel.

    Key outputs per player:
    - player_idx: integer index (for embedding lookup)
    - position_idx: integer index (for embedding lookup)
    - numerical_features: [projected_pts, adp, year_norm, pos_scarcity_rank, points_per_dollar]
    - target_points: actual season total_points (supervised label)
    - target_dollar: fair auction dollar value derived from PAR/VORP (supervised label)
    """

    def __init__(self):
        self.player_encoder = LabelEncoder()
        self.position_encoder = LabelEncoder()
        self.scaler = StandardScaler()

        self.player_id_to_idx: Dict[str, int] = {}
        self.idx_to_player_id: Dict[int, str] = {}
        self.player_names: Dict[str, str] = {}
        self.player_positions: Dict[str, str] = {}
        self.player_points: Dict[str, float] = {}
        self.player_par: Dict[str, float] = {}
        self.player_vorp_dollar: Dict[str, float] = {}

        self.position_map = {pos: pos for pos in POSITIONS}
        self.position_map['Unknown'] = 'UNKNOWN'

        self._fitted = False

    # ------------------------------------------------------------------
    # PAR / VORP calculations
    # ------------------------------------------------------------------

    def calculate_par(self, stats_df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate Points Above Replacement (PAR) for each player.

        Replacement level = the Nth-ranked player at each position,
        where N = REPLACEMENT_THRESHOLDS[position].
        """
        df = stats_df.copy()
        df['PAR'] = 0.0
        df['replacement_points'] = 0.0

        for pos in POSITIONS:
            mask = df['position'] == pos
            pos_players = df[mask].sort_values('total_points', ascending=False)
            threshold = REPLACEMENT_THRESHOLDS.get(pos, 12)

            if len(pos_players) >= threshold:
                replacement_pts = pos_players.iloc[threshold - 1]['total_points']
            elif len(pos_players) > 0:
                replacement_pts = pos_players.iloc[-1]['total_points']
            else:
                replacement_pts = 0.0

            df.loc[mask, 'replacement_points'] = replacement_pts
            df.loc[mask, 'PAR'] = df.loc[mask, 'total_points'] - replacement_pts

        return df

    def calculate_vorp_dollar(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Convert PAR to dollar value using Value Over Replacement Player (VORP).

        Formula: Each team has $200 budget. ~$100 is typically spent on "value"
        players above replacement. We scale PAR proportionally to that $100.

        dollar_value = max(1, (PAR / total_positive_PAR) * total_value_budget)

        where total_value_budget ≈ $1200 (12 teams × ~$100 in value spending).
        """
        df = df.copy()
        total_value_budget = 12 * 100  # conservative estimate

        # Only starters (positive PAR) share the value budget
        positive_par_mask = df['PAR'] > 0
        total_positive_par = df.loc[positive_par_mask, 'PAR'].sum()

        if total_positive_par > 0:
            df['VORP_dollar'] = np.where(
                positive_par_mask,
                (df['PAR'] / total_positive_par) * total_value_budget,
                1.0  # replacement players worth $1
            )
        else:
            df['VORP_dollar'] = 1.0

        # Cap at $80 (no player should consume >40% of budget in practice)
        df['VORP_dollar'] = df['VORP_dollar'].clip(1.0, 80.0)

        return df

    # ------------------------------------------------------------------
    # Core data processing
    # ------------------------------------------------------------------

    def process_draft_data(
        self,
        draft_df: pd.DataFrame,
        stats_df: pd.DataFrame,
        year: Optional[int] = None,
    ) -> Tuple[np.ndarray, ...]:
        """
        Process raw draft + stats data into arrays for model training.

        Returns:
            player_ids: (N,) int array of player embedding indices
            position_ids: (N,) int array of position embedding indices
            numerical_features: (N, 5) float array
            target_points: (N,) float array — actual season points
            target_dollar: (N,) float array — VORP-derived fair auction value
        """
        draft_df = draft_df.copy()
        stats_df = stats_df.copy()
        draft_df['player_id'] = draft_df['player_id'].astype(str)
        stats_df['player_id'] = stats_df['player_id'].astype(str)

        # Merge
        merged = pd.merge(draft_df, stats_df, on='player_id', how='left')
        merged['position'] = merged['position'].fillna('UNKNOWN')
        merged['total_points'] = merged['total_points'].fillna(0.0)
        merged['bid_amount'] = merged['bid_amount'].fillna(1.0)

        # Cache player metadata
        self.player_names.update(dict(zip(merged['player_id'], merged.get('player_name', merged.get('name', '')))))
        self.player_positions.update(dict(zip(merged['player_id'], merged['position'])))
        self.player_points.update(dict(zip(merged['player_id'], merged['total_points'].astype(float))))

        # PAR / VORP
        merged = self.calculate_par(merged)
        merged = self.calculate_vorp_dollar(merged)
        self.player_par.update(dict(zip(merged['player_id'], merged['PAR'].astype(float))))
        self.player_vorp_dollar.update(dict(zip(merged['player_id'], merged['VORP_dollar'].astype(float))))

        # Encode players and positions (fit if first call)
        if not self._fitted:
            self.player_encoder.fit(merged['player_id'])
            self.position_encoder.fit(POSITIONS + ['UNKNOWN'])
            self._fitted = True
        else:
            # Handle unseen players by adding them
            known = set(self.player_encoder.classes_)
            new_players = [p for p in merged['player_id'].unique() if p not in known]
            if new_players:
                self.player_encoder.classes_ = np.concatenate([self.player_encoder.classes_, new_players])

        merged['player_idx'] = self._safe_encode_player(merged['player_id'])
        merged['position_idx'] = self._safe_encode_position(merged['position'])

        # Update index mappings
        for pid, idx in zip(merged['player_id'], merged['player_idx']):
            self.player_id_to_idx[pid] = int(idx)
            self.idx_to_player_id[int(idx)] = pid

        # Position scarcity rank within this dataset (lower = more scarce)
        merged['pos_scarcity_rank'] = merged.groupby('position')['total_points'].rank(ascending=False)

        # Points per dollar (with bid_amount from actual draft)
        merged['points_per_dollar'] = merged['total_points'] / (merged['bid_amount'].clip(lower=1))

        year_norm = float(year - 2009) / 15.0 if year else 0.0

        numerical = np.column_stack([
            merged['bid_amount'].fillna(1).values.astype(np.float32),          # historical bid
            merged.get('adp', pd.Series(0, index=merged.index)).fillna(0).values.astype(np.float32),
            np.full(len(merged), year_norm, dtype=np.float32),
            merged['pos_scarcity_rank'].values.astype(np.float32),
            merged['points_per_dollar'].fillna(0).values.astype(np.float32),
        ])

        # Fit scaler on first call
        if not hasattr(self.scaler, 'mean_') or self.scaler.mean_ is None:
            self.scaler.fit(numerical)
        numerical_scaled = self.scaler.transform(numerical)

        return (
            merged['player_idx'].values.astype(np.int64),
            merged['position_idx'].values.astype(np.int64),
            numerical_scaled.astype(np.float32),
            merged['total_points'].values.astype(np.float32),
            merged['VORP_dollar'].values.astype(np.float32),
        )

    def process_multi_year(
        self,
        year_data: List[Tuple[int, pd.DataFrame, pd.DataFrame]],
    ) -> Tuple[np.ndarray, ...]:
        """
        Process multiple years of data. Fits encoders on the full corpus first.

        Args:
            year_data: list of (year, draft_df, stats_df)

        Returns: same tuple as process_draft_data but concatenated across years
        """
        # First pass: collect all player IDs and fit encoders
        all_player_ids = []
        for year, draft_df, stats_df in year_data:
            all_player_ids.extend(draft_df['player_id'].astype(str).tolist())

        all_player_ids = list(set(all_player_ids))
        self.player_encoder.fit(all_player_ids)
        self.position_encoder.fit(POSITIONS + ['UNKNOWN'])
        self._fitted = True

        # Fit scaler on first year's numerical features
        first_year, first_draft, first_stats = year_data[0]
        first_draft = first_draft.copy()
        first_stats = first_stats.copy()
        first_merged = pd.merge(first_draft.assign(player_id=first_draft['player_id'].astype(str)),
                                first_stats.assign(player_id=first_stats['player_id'].astype(str)),
                                on='player_id', how='left')
        first_merged['total_points'] = first_merged['total_points'].fillna(0)
        first_merged['bid_amount'] = first_merged['bid_amount'].fillna(1)
        first_merged['pos_scarcity_rank'] = first_merged.groupby('position')['total_points'].rank(ascending=False)
        first_merged['points_per_dollar'] = first_merged['total_points'] / first_merged['bid_amount'].clip(lower=1)
        year_norm = float(first_year - 2009) / 15.0
        numerical_sample = np.column_stack([
            first_merged['bid_amount'].values.astype(np.float32),
            first_merged.get('adp', pd.Series(0, index=first_merged.index)).fillna(0).values.astype(np.float32),
            np.full(len(first_merged), year_norm, dtype=np.float32),
            first_merged['pos_scarcity_rank'].values.astype(np.float32),
            first_merged['points_per_dollar'].fillna(0).values.astype(np.float32),
        ])
        self.scaler.fit(numerical_sample)

        # Second pass: process each year
        all_results = [[], [], [], [], []]
        for year, draft_df, stats_df in year_data:
            results = self.process_draft_data(draft_df, stats_df, year=year)
            for i, arr in enumerate(results):
                all_results[i].append(arr)

        return tuple(np.concatenate(arrays) for arrays in all_results)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _safe_encode_player(self, player_ids: pd.Series) -> pd.Series:
        """Encode player IDs, mapping unknown players to 0."""
        known = set(self.player_encoder.classes_)
        result = []
        for pid in player_ids:
            if pid in known:
                result.append(self.player_encoder.transform([pid])[0])
            else:
                result.append(0)
        return pd.Series(result, index=player_ids.index)

    def _safe_encode_position(self, positions: pd.Series) -> pd.Series:
        """Encode positions, mapping unknowns to 'UNKNOWN' index."""
        known = set(self.position_encoder.classes_)
        result = []
        for pos in positions:
            p = pos if pos in known else 'UNKNOWN'
            result.append(self.position_encoder.transform([p])[0])
        return pd.Series(result, index=positions.index)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, save_dir: Path) -> None:
        """Save all encoder state to disk."""
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

        np.save(save_dir / 'player_classes.npy', self.player_encoder.classes_)
        np.save(save_dir / 'position_classes.npy', self.position_encoder.classes_)
        np.save(save_dir / 'player_id_to_idx.npy', np.array(list(self.player_id_to_idx.items())))
        np.save(save_dir / 'player_names.npy', np.array(list(self.player_names.items())))
        np.save(save_dir / 'player_positions.npy', np.array(list(self.player_positions.items())))
        np.save(save_dir / 'player_points.npy', np.array(list(self.player_points.items())))
        np.save(save_dir / 'player_par.npy', np.array(list(self.player_par.items())))
        np.save(save_dir / 'player_vorp_dollar.npy', np.array(list(self.player_vorp_dollar.items())))
        np.save(save_dir / 'scaler_mean.npy', self.scaler.mean_)
        np.save(save_dir / 'scaler_scale.npy', self.scaler.scale_)
        logger.info(f"Preprocessor state saved to {save_dir}")

    def load(self, save_dir: Path) -> None:
        """Load encoder state from disk."""
        save_dir = Path(save_dir)

        self.player_encoder.classes_ = np.load(save_dir / 'player_classes.npy', allow_pickle=True)
        self.position_encoder.classes_ = np.load(save_dir / 'position_classes.npy', allow_pickle=True)

        id_to_idx = np.load(save_dir / 'player_id_to_idx.npy', allow_pickle=True)
        self.player_id_to_idx = {str(k): int(v) for k, v in id_to_idx}
        self.idx_to_player_id = {int(v): str(k) for k, v in id_to_idx}

        self.player_names = dict(np.load(save_dir / 'player_names.npy', allow_pickle=True))
        self.player_positions = dict(np.load(save_dir / 'player_positions.npy', allow_pickle=True))
        self.player_points = {k: float(v) for k, v in np.load(save_dir / 'player_points.npy', allow_pickle=True)}
        self.player_par = {k: float(v) for k, v in np.load(save_dir / 'player_par.npy', allow_pickle=True)}
        self.player_vorp_dollar = {k: float(v) for k, v in np.load(save_dir / 'player_vorp_dollar.npy', allow_pickle=True)}

        self.scaler.mean_ = np.load(save_dir / 'scaler_mean.npy')
        self.scaler.scale_ = np.load(save_dir / 'scaler_scale.npy')
        self._fitted = True
        logger.info(f"Preprocessor state loaded from {save_dir}")

    # ------------------------------------------------------------------
    # Lookup utilities
    # ------------------------------------------------------------------

    def get_player_name(self, player_idx: int) -> str:
        player_id = self.idx_to_player_id.get(player_idx)
        return str(self.player_names.get(player_id, "Unknown Player")) if player_id else "Unknown Player"

    def get_player_position(self, player_idx: int) -> str:
        player_id = self.idx_to_player_id.get(player_idx)
        return str(self.player_positions.get(player_id, "UNKNOWN")) if player_id else "UNKNOWN"

    def get_player_par(self, player_id: str) -> float:
        return self.player_par.get(str(player_id), 0.0)

    def get_player_vorp_dollar(self, player_id: str) -> float:
        return self.player_vorp_dollar.get(str(player_id), 1.0)

    @property
    def num_players(self) -> int:
        return len(self.player_encoder.classes_)

    @property
    def num_positions(self) -> int:
        return len(self.position_encoder.classes_)
