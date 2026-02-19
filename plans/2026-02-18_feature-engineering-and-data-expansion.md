# Feature Engineering and Data Expansion

**Date**: 2026-02-18

## Context

The current training pipeline has a sparse feature set (only 5 numerical features per player) and relies exclusively on ESPN data. With 16 years of historical league data plus access to comprehensive public NFL stats (nflverse/nflreadpy), we can dramatically improve point projection quality and enable strategic bidding intelligence. This plan covers:

1. Rename `data/raw/` → `data/favrefignewton/` (ESPN-sourced data directory)
2. Acquire external NFL data (nflverse) into a new `data/nflverse/` directory
3. Build a feature engineering pipeline producing enriched processed CSVs
4. Enhance the value model (5 → 14 numerical features)
5. Add strategic analysis script to answer league-specific questions
6. Extend RL state vector (56 → 72 dims)

---

## Key Findings from Data Exploration

- `predraft_values.auction_value` == `draft_results.bid_amount` exactly — it is NOT an independent pre-draft projection
- `predraft_values.projected_points` is the pre-season ESPN projection; actual `total_points` consistently exceeds it (QB: +194pts, TE: +90pts, WR: +89pts, RB: +89pts bias)
- WR budget share correlates +0.253 with team total points; top-4 teams spend more on WRs and less on RBs than bottom teams
- 290 players have 4+ seasons of historical data — sufficient for multi-year trend features
- Weekly stats available for 2019–2024; JSON `stats` field contains `rushingAttempts`, `receivingTargets`, etc.
- Current opponent heuristic (`auction_value * (1 ± 0.10)`) is unrealistic — managers have measurable position preferences

---

## Part 0: Directory Rename

### `data/raw/` → `data/favrefignewton/`

**Files to update** (5 Python + 3 config):

| File | Line | Change |
|------|------|--------|
| `ffai/src/ffai/simulation/auction_draft_simulator.py` | 40 | `"data/raw"` → `"data/favrefignewton"` |
| `ffai/src/ffai/simulation/season_simulator.py` | 15 | `"data/raw"` → `"data/favrefignewton"` |
| `ffai/src/ffai/rl/puffer_env.py` | 112 | `"data/raw"` → `"data/favrefignewton"` |
| `ffai/src/ffai/auction_draft_simulator.py` | 29 | `"data/raw"` → `"data/favrefignewton"` |
| `ffai/src/ffai/season_simulator.py` | 15 | `"data/raw"` → `"data/favrefignewton"` |
| `.gitignore` | 46 | `ffai/data/raw/` → `ffai/data/favrefignewton/` |
| `CLAUDE.md` | 18, 21 | update `raw/` references |
| `README.md` | 80 | update `raw/` reference |

Also physically rename the directory:
```bash
mv ffai/src/ffai/data/raw ffai/src/ffai/data/favrefignewton
```

---

## Part 1: External NFL Data Acquisition

### New Directory: `data/nflverse/`

Use the `nflreadpy` Python package (modern replacement for `nfl_data_py`) to download comprehensive NFL data. This is gitignored (local only) like `favrefignewton/`.

**Package**: `pip install nflreadpy`
**GitHub**: https://github.com/nflverse/nflreadpy

### New Script: `ffai/scripts/fetch_nfl_data.py`

Downloads and caches data into `ffai/src/ffai/data/nflverse/`:

```
data/nflverse/
  player_stats_{year}.csv       # per-player season rushing/receiving/passing totals
  weekly_player_stats_{year}.csv # per-player per-week stats with snap counts
  rosters_{year}.csv            # player demographics (age, position, team, draft round)
  draft_picks.csv               # all-time draft picks with pick number and round
  snap_counts_{year}.csv        # weekly snap count % by player (2012+)
```

**Data loaded via nflreadpy:**

```python
import nflreadpy as nfl

# Season-level player stats (1999+)
nfl.load_player_stats(seasons=range(2009, 2025))

# Weekly player stats with snap counts (2012+)
nfl.load_player_stats(seasons=range(2012, 2025), stat_type="weekly")

# Roster/demographics: age, draft_round, experience, height, weight
nfl.load_rosters(seasons=range(2009, 2025))

# Historical draft picks: player, round, pick, team
nfl.load_draft_picks()
```

**Key columns from nflreadpy player stats:**
- `player_id`, `player_name`, `position`, `season`
- `rushing_att`, `rushing_yards`, `rushing_tds`
- `targets`, `receptions`, `receiving_yards`, `receiving_tds`
- `target_share`, `air_yards_share`, `wopr` (weighted opportunity rating)
- `snap_pct` (weekly only: percentage of team offensive snaps)
- `fantasy_points_ppr`

**Key columns from nflreadpy rosters:**
- `gsis_id` (maps to ESPN `player_id` via cross-reference)
- `age`, `draft_number`, `draft_round`, `entry_year`, `years_exp`
- `weight`, `height`

**ID cross-reference**: nflverse uses `gsis_id`; ESPN uses its own IDs. Use nflreadpy's `load_ff_playerids()` which provides a mapping table including ESPN IDs.

### Gitignore addition:
```
ffai/data/nflverse/
```

---

## Part 2: Feature Engineering Pipeline

### New Module: `ffai/src/ffai/data/feature_engineering/`

```
feature_engineering/
  __init__.py
  weekly_stats_parser.py    # parse ESPN weekly JSON stats (2019+)
  player_history.py         # multi-year lookback features per (player, year)
  manager_tendencies.py     # per-manager bidding profile
  position_strategy.py      # position-level strategic signals per year
  nfl_feature_builder.py    # join nflverse data to produce enriched player features
```

### New File: `ffai/src/ffai/data/feature_store.py`

Unified loader for all processed CSVs. Provides:
```python
class FeatureStore:
    def get_player_features(self, player_id: str, year: int) -> dict
    def get_manager_tendencies(self, manager_id: str) -> dict
    def get_position_strategy(self, position: str, year: int) -> dict
```
Handles missing players with mean imputation by position.

### New Script: `ffai/scripts/build_features.py`

CLI orchestrator that runs all feature modules in order:
```bash
python scripts/build_features.py [--league-id 770280] [--years 2009-2024]
```

### Output CSVs (all go to `data/processed/`):

#### `player_history_770280.csv`
One row per (player_id, year). All features are **lookback-safe** (only use data from years < current year — no future leakage). Rookies get NaN/0 for lookback features.

**Feature groups:**

*Multi-year rolling averages (lookback from prior years):*
```
pts_3yr_avg        = mean(total_points, Y-1 to Y-3)
pts_3yr_std        = std(total_points, Y-1 to Y-3)     # consistency
pts_1yr_val        = total_points[Y-1]                  # most recent season
yoy_pct_change     = (pts[Y-1] - pts[Y-2]) / pts[Y-2]  # momentum, clipped [-1, 1]
years_in_league    = count of prior seasons with data
```

*Projection bias correction (most impactful new feature):*
```
proj_ratio_3yr_avg = mean(total_points[Y-k] / projected_points[Y-k], k=1..3)
proj_bias_1yr      = total_points[Y-1] - projected_points[Y-1]
```
Per-position baseline ratios: QB ~1.09, RB ~1.03, TE ~1.10, WR ~1.04. Players who consistently outperform projections should get a premium.

*Weekly consistency (ESPN weekly data, 2019+; else NaN):*
```
weekly_pts_cv      = std(weekly_points) / mean(weekly_points)  # boom/bust
floor_pts          = percentile_25(weekly_points)
ceiling_pts        = percentile_75(weekly_points)
weeks_active       = count(weeks with points > 0)              # durability
```

*Usage from nflverse (2009+):*
```
targets_per_game   = mean(targets per week)      # WR/TE/RB
carries_per_game   = mean(rushing_att per week)  # RB
target_share_3yr   = mean(target_share, Y-1 to Y-3)  # nflverse metric
snap_pct_1yr       = snap_pct[Y-1]              # from nflverse (2012+)
```

*Player demographics (from nflverse rosters):*
```
age_at_draft       = player age in September of draft year
draft_round        = NFL draft round (0 = undrafted, UDFA)
years_nfl_exp      = years in NFL as of draft year
```

#### `manager_tendencies_770280.csv`
One row per manager_id. Compute relative to position-year averages (not to `auction_value` since `auction_value == bid_amount`):

```
rb_budget_share, wr_budget_share, qb_budget_share, te_budget_share
bid_per_proj_pt_rb, bid_per_proj_pt_wr, bid_per_proj_pt_qb, bid_per_proj_pt_te
high_bid_rate       = fraction of picks with bid > $30
dollar_one_rate     = fraction of picks costing $1
seasons_active      = total seasons participated
recent_wr_share     = wr_budget_share over last 3 seasons
recent_rb_share     = rb_budget_share over last 3 seasons
```

#### `position_strategy_770280.csv`
One row per (position, year):
```
avg_bid, median_bid, top5_avg_bid
roi_mean             = mean(total_points / bid_amount)
proj_accuracy_ratio  = mean(total_points / projected_points)
budget_share_pct     = position_total_bid / league_total_bid
winning_budget_share = avg budget share for top-4 teams at this position
```

---

## Part 3: Enhanced Value Model

### `ffai/src/ffai/data/preprocessor.py`

Add `feature_store: FeatureStore` parameter to `process_draft_data()`. Join player history features during training data construction.

Remove `bid_amount` and `points_per_dollar` from numerical features (circular at inference time — the bid hasn't happened yet). Replace with `projected_pts_norm`.

### `ffai/src/ffai/value_model/player_value_model.py`

Change `NUM_NUMERICAL_FEATURES = 5` → `NUM_NUMERICAL_FEATURES = 14`. The `numerical_encoder` layer (`nn.Linear(5, 128)` → `nn.Linear(14, 128)`) is the only architectural change.

**New 14-dim numerical feature vector:**
```
0: projected_pts_norm        = projected_points / 400.0
1: adp_round                 = draft round (renamed from 'adp')
2: year_norm                 = (year - 2009) / 15.0
3: pos_scarcity_rank         = within-year rank by total_points (normalized)
4: proj_ratio_3yr_avg        # from player_history (NEW)
5: proj_bias_1yr_norm        # from player_history (NEW)
6: pts_3yr_avg_norm          = pts_3yr_avg / 400.0 (NEW)
7: pts_1yr_val_norm          = pts_1yr_val / 400.0 (NEW)
8: yoy_pct_change            # clipped [-1, 1] (NEW)
9: years_in_league           # clipped [0, 1] with max=10 (NEW)
10: weekly_pts_cv            # 0 if pre-2019 or no data (NEW)
11: floor_pts_norm           = floor_pts / 40.0 (NEW)
12: targets_per_game_norm    = targets_per_game / 10.0 (NEW)
13: snap_pct_1yr             # 0 if pre-2012 or no data (NEW)
```

Missing values imputed with position-year mean before standardization.

**Retrain value model after building features:**
```bash
python scripts/build_features.py
python scripts/train_value_model.py --train-years 2009-2022 --val-year 2023 --test-year 2024
```

---

## Part 4: Strategic Analysis Script

### `ffai/scripts/analyze_strategy.py`

Standalone script that prints analysis tables. Run once to inform RL design. No file output required.

```bash
python scripts/analyze_strategy.py
```

**Answers to produce:**
1. **Position ROI over 16 years**: avg bid, avg points, points-per-dollar by position and year
2. **Manager tendencies**: which managers overbid/underbid by position; rank by budget efficiency
3. **Projection bias**: systematic ESPN bias by position; how much to scale up
4. **Winning patterns**: WR vs RB budget allocation for top-4 vs bottom-8 teams; Spearman correlation of position budget share with final standing

---

## Part 5: RL State Enhancement

### `ffai/src/ffai/rl/state_builder.py`

Change `STATE_DIM = 56` → `STATE_DIM = 72`. Add `feature_store: FeatureStore` parameter to `build_state()`.

**New state layout:**
```
[0:56]  Existing dims (unchanged)
[56:62] Current player history (6 dims):
          pts_3yr_avg_norm, proj_ratio_3yr_avg, yoy_pct_change,
          weekly_pts_cv, floor_pts_norm, years_in_league
[62:66] Position strategy (4 dims):
          position_winning_budget_share, position_roi_3yr_avg,
          position_proj_accuracy_ratio, position_scarcity_historical
[66:72] Opponent tendencies (6 dims, aggregated over top-3 budget opponents):
          avg_rb_budget_share, avg_wr_budget_share,
          avg_high_bid_rate, avg_dollar_one_rate,
          avg_bid_per_proj_pt_rb, avg_bid_per_proj_pt_wr
```

### `ffai/src/ffai/simulation/auction_draft_simulator.py`

Load `FeatureStore` at `__init__` time. Inject `current_player_history` and `opponent_tendencies` into `get_state()` return dict.

Also upgrade opponent bidding heuristic using manager tendencies:
```python
# Replace: max_bid = auction_value * (1 ± 0.10)
# With: max_bid = projected_points * mgr_tendency['bid_per_proj_pt_{position}']
```

### `ffai/src/ffai/rl/puffer_env.py`

Update observation space shape: `shape=(STATE_DIM,)` (imports `STATE_DIM` from `state_builder`; no hardcoded value).

---

## Implementation Order

| Step | Task | Impact |
|------|------|--------|
| 1 | Rename `raw/` → `favrefignewton/` (rename dir + update 8 files) | Unblocks clean directory structure |
| 2 | `fetch_nfl_data.py` — download nflverse data | Provides age, snap %, target share, WOPR |
| 3 | `feature_engineering/weekly_stats_parser.py` — parse ESPN JSON stats | Foundation for consistency features |
| 4 | `feature_engineering/player_history.py` + `nfl_feature_builder.py` | Core feature set |
| 5 | `feature_engineering/manager_tendencies.py` + `position_strategy.py` | Strategic signals |
| 6 | `feature_store.py` + `build_features.py` CLI | Wires pipeline together |
| 7 | `analyze_strategy.py` — run and read output | Validates features, informs design |
| 8 | Update `preprocessor.py` (5 → 14 features) + retrain value model | Improves projection quality |
| 9 | Update `state_builder.py` (56 → 72 dims) + upgrade opponent model | Enables strategic bidding |
| 10 | Retrain RL agent from Phase 1 with new state | Full training run |

---

## Verification

```bash
# 1. Confirm rename is clean
python -c "from ffai.simulation.auction_draft_simulator import AuctionDraftSimulator; print('import ok')"

# 2. Confirm nflverse data downloaded
ls ffai/src/ffai/data/nflverse/

# 3. Build features and sanity check
python scripts/build_features.py
python -c "
import pandas as pd
h = pd.read_csv('ffai/src/ffai/data/processed/player_history_770280.csv')
print('rows:', len(h))
print('no future leakage:', (h.groupby('year').apply(lambda g: g['pts_3yr_avg'].isna().any())).all())
print(h.head())
"

# 4. Run strategic analysis
python scripts/analyze_strategy.py

# 5. Retrain value model, compare RMSE before/after
python scripts/train_value_model.py --train-years 2009-2022 --val-year 2023 --test-year 2024

# 6. Smoke test RL env with new state dim
python -c "
from ffai.rl.puffer_env import AuctionDraftEnv
env = AuctionDraftEnv(year=2024, enable_season_sim=False)
obs, info = env.reset()
print('obs shape:', obs.shape)  # expect (72,)
"
```

---

## Files to Create

```
ffai/src/ffai/data/feature_engineering/__init__.py
ffai/src/ffai/data/feature_engineering/weekly_stats_parser.py
ffai/src/ffai/data/feature_engineering/player_history.py
ffai/src/ffai/data/feature_engineering/manager_tendencies.py
ffai/src/ffai/data/feature_engineering/position_strategy.py
ffai/src/ffai/data/feature_engineering/nfl_feature_builder.py
ffai/src/ffai/data/feature_store.py
ffai/scripts/fetch_nfl_data.py
ffai/scripts/build_features.py
ffai/scripts/analyze_strategy.py
```

## Files to Modify

```
ffai/src/ffai/data/preprocessor.py                    # 5 → 14 numerical features
ffai/src/ffai/value_model/player_value_model.py       # NUM_NUMERICAL_FEATURES = 14
ffai/src/ffai/rl/state_builder.py                     # STATE_DIM = 72
ffai/src/ffai/simulation/auction_draft_simulator.py   # data dir + opponent model
ffai/src/ffai/rl/puffer_env.py                        # obs space shape (via STATE_DIM import)
ffai/src/ffai/auction_draft_simulator.py              # data dir rename
ffai/src/ffai/season_simulator.py                     # data dir rename
ffai/src/ffai/simulation/season_simulator.py          # data dir rename
.gitignore                                            # add nflverse/, update favrefignewton/
CLAUDE.md                                             # update directory convention
README.md                                             # update raw/ reference
```
