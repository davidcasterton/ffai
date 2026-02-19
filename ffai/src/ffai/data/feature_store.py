"""
FeatureStore — unified loader for all processed feature CSVs.

Provides per-lookup access with position-year mean imputation for missing players.
All CSVs are written by scripts/build_features.py and live in data/processed/.
"""

import logging
from pathlib import Path

import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)

PROCESSED_DIR = Path(__file__).parent / "processed"
LEAGUE_ID = "770280"

PLAYER_HISTORY_COLS = [
    "pts_3yr_avg", "pts_3yr_std", "pts_1yr_val", "yoy_pct_change",
    "years_in_league", "proj_ratio_3yr_avg", "proj_bias_1yr",
]
NFL_FEATURE_COLS = [
    "targets_per_game", "carries_per_game", "target_share_3yr",
    "snap_pct_1yr", "draft_round", "years_nfl_exp",
]
MANAGER_COLS = [
    "rb_budget_share", "wr_budget_share", "qb_budget_share", "te_budget_share",
    "bid_per_proj_pt_rb", "bid_per_proj_pt_wr", "bid_per_proj_pt_qb", "bid_per_proj_pt_te",
    "high_bid_rate", "dollar_one_rate", "seasons_active",
    "recent_wr_share", "recent_rb_share",
]
POSITION_STRATEGY_COLS = [
    "avg_bid", "median_bid", "top5_avg_bid",
    "roi_mean", "proj_accuracy_ratio", "budget_share_pct", "winning_budget_share",
]


class FeatureStore:
    """
    Thin wrapper around processed CSVs. Loads once at init; serves feature dicts
    via get_player_features(), get_manager_tendencies(), get_position_strategy().

    Missing values are imputed with position-year means (for numeric features)
    or 0 (categorical/count features).
    """

    def __init__(
        self,
        processed_dir: Path = PROCESSED_DIR,
        league_id: str = LEAGUE_ID,
    ):
        self.processed_dir = Path(processed_dir)
        self.league_id = league_id

        self._player_history: pd.DataFrame | None = None
        self._manager_tendencies: pd.DataFrame | None = None
        self._position_strategy: pd.DataFrame | None = None

        # Pre-computed position-year means for imputation
        self._ph_means: pd.DataFrame | None = None
        self._nfl_means: pd.DataFrame | None = None
        self._ps_means: pd.DataFrame | None = None

        self._load()

    def _load(self) -> None:
        ph_path = self.processed_dir / f"player_history_{self.league_id}.csv"
        mt_path = self.processed_dir / f"manager_tendencies_{self.league_id}.csv"
        ps_path = self.processed_dir / f"position_strategy_{self.league_id}.csv"

        if ph_path.exists():
            self._player_history = pd.read_csv(ph_path)
            self._player_history["player_id"] = self._player_history["player_id"].astype(str)
            self._player_history = self._player_history.set_index(["player_id", "year"])
            # Compute position-year means (requires 'position' column in player_history)
            if "position" in self._player_history.columns:
                self._ph_means = (
                    self._player_history
                    .groupby(["position", "year"])[PLAYER_HISTORY_COLS + NFL_FEATURE_COLS]
                    .mean()
                )
        else:
            logger.warning(f"player_history not found at {ph_path} — run build_features.py first")

        if mt_path.exists():
            self._manager_tendencies = pd.read_csv(mt_path)
            self._manager_tendencies = self._manager_tendencies.set_index("manager_id")
        else:
            logger.warning(f"manager_tendencies not found at {mt_path}")

        if ps_path.exists():
            self._position_strategy = pd.read_csv(ps_path)
            self._position_strategy = self._position_strategy.set_index(["position", "year"])
        else:
            logger.warning(f"position_strategy not found at {ps_path}")

    @property
    def loaded(self) -> bool:
        return self._player_history is not None

    def get_player_features(self, player_id: str, year: int, position: str = "") -> dict:
        """
        Return a dict of all player history + nfl features for (player_id, year).
        Missing players are imputed with position-year means.
        """
        all_cols = PLAYER_HISTORY_COLS + NFL_FEATURE_COLS
        result = {col: np.nan for col in all_cols}

        if self._player_history is not None:
            try:
                row = self._player_history.loc[(str(player_id), year)]
                for col in all_cols:
                    if col in row.index:
                        result[col] = float(row[col]) if not pd.isna(row[col]) else np.nan
            except KeyError:
                pass  # player not in history; will use imputed values

        # Impute NaNs with position-year means
        if self._ph_means is not None and position:
            try:
                means_row = self._ph_means.loc[(position, year)]
                for col in all_cols:
                    if np.isnan(result.get(col, np.nan)) and col in means_row.index:
                        result[col] = float(means_row[col]) if not pd.isna(means_row[col]) else 0.0
            except KeyError:
                pass

        # Final fallback: 0
        for col in all_cols:
            if col not in result or (isinstance(result[col], float) and np.isnan(result[col])):
                result[col] = 0.0

        return result

    def get_manager_tendencies(self, manager_id: str) -> dict:
        """Return bidding tendency dict for a manager. Returns league-average defaults if unknown."""
        defaults = {col: np.nan for col in MANAGER_COLS}

        if self._manager_tendencies is not None:
            try:
                row = self._manager_tendencies.loc[manager_id]
                for col in MANAGER_COLS:
                    if col in row.index:
                        defaults[col] = float(row[col]) if not pd.isna(row[col]) else np.nan
            except KeyError:
                pass

        # League-average fallback for NaN
        if self._manager_tendencies is not None:
            means = self._manager_tendencies[MANAGER_COLS].mean()
            for col in MANAGER_COLS:
                if np.isnan(defaults.get(col, np.nan)):
                    defaults[col] = float(means.get(col, 0.0))

        return defaults

    def get_position_strategy(self, position: str, year: int) -> dict:
        """Return strategic signals for (position, year). Falls back to prior year if missing."""
        defaults = {col: np.nan for col in POSITION_STRATEGY_COLS}

        if self._position_strategy is not None:
            # Try exact year first, then walk back up to 3 years
            for y in [year, year - 1, year - 2, year - 3]:
                try:
                    row = self._position_strategy.loc[(position, y)]
                    for col in POSITION_STRATEGY_COLS:
                        if col in row.index:
                            defaults[col] = float(row[col]) if not pd.isna(row[col]) else np.nan
                    break
                except KeyError:
                    continue

        # Final fallback: 0
        for col in POSITION_STRATEGY_COLS:
            if np.isnan(defaults.get(col, np.nan)):
                defaults[col] = 0.0

        return defaults
