# ffai Model Design Document

This document describes the machine learning system used to optimize bidding strategy in ESPN fantasy football auction drafts. The system combines a supervised **Player Value Model** with a **Proximal Policy Optimization (PPO)** reinforcement learning agent trained via [PufferLib](https://github.com/PufferAI/PufferLib).

---

## Table of Contents

1. [What the Model Learns](#1-what-the-model-learns)
2. [System Architecture](#2-system-architecture)
3. [Component 1 — Player Value Model (Supervised)](#3-component-1--player-value-model-supervised)
4. [Component 2 — Feature Engineering Pipeline](#4-component-2--feature-engineering-pipeline)
5. [Component 3 — Auction Draft Environment](#5-component-3--auction-draft-environment)
6. [Component 4 — State Vector (72 Dimensions)](#6-component-4--state-vector-72-dimensions)
7. [Component 5 — PPO Policy Network](#7-component-5--ppo-policy-network)
8. [Component 6 — Reward Function](#8-component-6--reward-function)
9. [Component 7 — Opponent Modeling](#9-component-7--opponent-modeling)
10. [Training — Three-Phase Curriculum](#10-training--three-phase-curriculum)
11. [Training Hyperparameters](#11-training-hyperparameters)
12. [Episode Lifecycle](#12-episode-lifecycle)
13. [Using the Trained Model](#13-using-the-trained-model)
14. [Key Design Decisions](#14-key-design-decisions)
15. [Extending the System](#15-extending-the-system)

---

## 1. What the Model Learns

An **auction draft** is the most strategically complex format in fantasy football. All managers start with the same fixed budget (e.g., $200) and take turns nominating players. Any manager can bid on any nominated player. The draft ends when every manager fills every roster slot.

Good auction draft strategy requires simultaneously reasoning about:

- **Absolute value**: Is this player worth the current bid given their projected output?
- **Relative value**: Are there comparable players still available who will go for less?
- **Budget management**: If I spend heavily now, will I have enough to fill the rest of my roster?
- **Scarcity**: Are elite players at this position running out? Do I need to pay up now or risk going without?
- **Opponent pressure**: Which opponents have money left to push up prices? Who is exhausted?
- **Timing**: When is the right phase of the draft to load up on one position vs. spread budget?

The RL agent learns a **bidding policy**: given the current state of the draft, what fraction of my remaining budget am I willing to pay for this player? It does not learn a static valuation table — it learns a context-sensitive strategy that adapts to how the draft is unfolding.

---

## 2. System Architecture

The system has two major stages:

**Stage 1: Supervised Learning**

A `PlayerValueModel` is trained on 15 years of historical ESPN draft and performance data. It learns to predict:
- A player's expected season fantasy points for a given year
- A player's fair auction dollar value (calibrated against historical prices)

This model's outputs are frozen and used as **inputs** to the RL stage.

**Stage 2: Reinforcement Learning**

A `AuctionDraftPolicy` is trained via PPO in a simulated auction draft environment (`AuctionDraftEnv`). The simulator runs full 12-team auctions using opponent behavioral models derived from historical draft data. The RL agent makes one decision per bidding round it participates in: the maximum price it is willing to pay for the current player.

```
Historical ESPN + nflverse Data
           |
           v
  [Feature Engineering]
    player_history.csv          → 3yr avg points, consistency, YoY trends
    manager_tendencies.csv      → per-manager bidding profiles
    position_strategy.csv       → ROI, winning team budget shares
           |
           v
  [PlayerValueModel Training]   (supervised, train_value_model.py)
    Input:  14 features per (player, year)
    Output: expected_season_points, fair_auction_dollar
    Saved:  checkpoints/value_model/best_model.pt
           |
           v
  [PPO Training]                (train_puffer.py, 3-phase curriculum)
    Environment: AuctionDraftEnv  — 12-team auction draft simulator
    Observation: 72-dim float32   — state of the draft at each decision
    Action:      bid_fraction ∈ [0,1]  →  max_bid = fraction × budget
    Reward:      value_efficiency (per pick) + season standing (terminal)
    Policy:      AuctionDraftPolicy — 2-layer MLP, Gaussian head
    Saved:  checkpoints/puffer/phase{1,2,3}_final.pt
           |
           v
  [Deployment]
    simulate_draft.py   — offline evaluation, play-by-play transcripts
    advisory_draft.py   — live draft assistant, terminal UI recommendations
    autonomous_draft.py — browser automation, fully hands-off drafting
```

---

## 3. Component 1 — Player Value Model (Supervised)

**Source**: `ffai/src/ffai/value_model/player_value_model.py`

### Purpose

The value model translates raw player features into two actionable numbers: expected season points and fair auction dollar value. These numbers are used both as inputs to the RL state vector and as the baseline for computing mid-draft rewards.

### Architecture

```
Input: (player_id, position_id, numerical_features[14])
            |
player_embedding:    Embedding(num_players+1, 256)  →  256 dims
position_embedding:  Embedding(num_positions+1, 64) →   64 dims
numerical_encoder:   Linear(14→128) + LayerNorm + ReLU + Dropout
                     Linear(128→128) + ReLU            → 128 dims
            |
combined = concat([256, 64, 128])                   → 448 dims
            |
shared trunk:
  Linear(448→256) + LayerNorm + ReLU + Dropout
  Linear(256→128) + ReLU                            → 128 dims
            |                    |
     points_head:          value_head:
  Linear(128→64)        Linear(128→64)
  ReLU                  ReLU
  Linear(64→1)          Linear(64→1)
       |                     |
  season_points          fair_dollar (Softplus, always > 0)
```

**Key choices:**
- **Player embeddings** (256-dim): Let the model learn a latent representation per player that captures factors not in the numerical features (playing style, coaching scheme, etc.)
- **Position embeddings** (64-dim): Let position-specific patterns be learned distinctly from player-specific patterns
- **Softplus on value head**: Ensures predicted auction values are always positive
- **Shared trunk**: Forces the model to learn a unified representation before branching into the two output heads

### 14 Numerical Input Features

| # | Feature | Description |
|---|---------|-------------|
| 1 | `targets_per_game` | Receiving targets per game (prior year) |
| 2 | `carries_per_game` | Rushing attempts per game (prior year) |
| 3 | `target_share_3yr` | 3-year rolling average of target share |
| 4 | `snap_pct_1yr` | Snap count percentage (prior year) |
| 5 | `draft_round` | NFL round drafted (0 = undrafted) |
| 6 | `years_nfl_exp` | Years of NFL experience |
| 7 | `pts_3yr_avg` | Average season points over prior 3 years |
| 8 | `pts_3yr_std` | Standard deviation of season points (consistency) |
| 9 | `pts_1yr_val` | Prior year's total season points |
| 10 | `yoy_pct_change` | Year-over-year point change, clipped to [-1, 1] |
| 11 | `years_in_league` | Years active in this ESPN league |
| 12 | `proj_ratio_3yr_avg` | Mean(actual / ESPN projected) over prior 3 years |
| 13 | `proj_bias_1yr` | Prior year's (actual − projected) in points |
| 14 | `weekly_pts_cv` | Coefficient of variation of weekly points (lower = more consistent) |

### Training

- **Loss**: MSE on both heads simultaneously
- **Optimizer**: Adam (lr=0.001)
- **Batch size**: 64
- **Epochs**: 100 with early stopping (patience=10)
- **Dropout**: 0.1 in encoder and trunk
- **Train/val/test split**: 2009–2022 train, 2023 val, 2024 test (time-series safe)

---

## 4. Component 2 — Feature Engineering Pipeline

**Source**: `ffai/src/ffai/data/feature_engineering/`

All feature engineering is **lookback-safe**: when computing features for draft year Y, only data from years Y-1 and earlier is used. This prevents data leakage.

### player_history.py

Produces one row per `(player_id, year)` from ESPN historical draft and scoring data.

| Column | Description |
|--------|-------------|
| `pts_3yr_avg` | Mean season points over prior 3 years |
| `pts_3yr_std` | Std dev of season points over prior 3 years |
| `pts_1yr_val` | Prior year's total season points |
| `yoy_pct_change` | (pts_Y-1 − pts_Y-2) / pts_Y-2, clipped to [-1, 1] |
| `years_in_league` | Years appearing in this ESPN league's data |
| `proj_ratio_3yr_avg` | Mean(actual / ESPN projected) over prior 3 years |
| `proj_bias_1yr` | Prior year's (actual − ESPN projected) |

### nfl_feature_builder.py

Augments player records with NFL usage and career data from nflverse.

| Column | Description |
|--------|-------------|
| `targets_per_game` | Receiving targets per game (prior year) |
| `carries_per_game` | Rushing attempts per game (prior year) |
| `target_share_3yr` | 3-year rolling average of target share |
| `snap_pct_1yr` | Snap count percentage (prior year) |
| `draft_round` | NFL round (0 = undrafted) |
| `years_nfl_exp` | NFL experience years |

### weekly_stats_parser.py

Parses ESPN weekly scoring logs (2019+) to compute consistency metrics.

| Column | Description |
|--------|-------------|
| `weekly_pts_cv` | Coefficient of variation of weekly points |
| `floor_pts` | 25th percentile weekly points |
| `ceiling_pts` | 75th percentile weekly points |
| `weeks_active` | Weeks with >0 fantasy points |

### manager_tendencies.py

Produces one row per `manager_id` by aggregating all their historical draft decisions.

| Column | Description |
|--------|-------------|
| `rb_budget_share` | Fraction of budget spent on RBs historically |
| `wr_budget_share` | Fraction of budget spent on WRs historically |
| `bid_per_proj_pt_rb` | Dollars per projected point paid for RBs |
| `bid_per_proj_pt_wr` | Dollars per projected point paid for WRs |
| `high_bid_rate` | Fraction of picks costing >$30 |
| `dollar_one_rate` | Fraction of picks costing $1 (sleeper rate) |
| `seasons_active` | Number of seasons in the data |
| `recent_rb_share` | RB budget share in the last 3 seasons |
| `recent_wr_share` | WR budget share in the last 3 seasons |

These profiles are used by the simulator to drive realistic opponent bidding behavior and are also encoded directly in the RL state vector.

### position_strategy.py

Produces one row per `(position, year)` by aggregating across all teams in each season.

| Column | Description |
|--------|-------------|
| `avg_bid` | Average auction price for this position |
| `median_bid` | Median auction price |
| `top5_avg_bid` | Average price of the top 5 most expensive players |
| `roi_mean` | Points per dollar (excluding $1 buys) |
| `proj_accuracy_ratio` | Mean(actual / projected) for this position |
| `budget_share_pct` | % of total league budget spent on this position |
| `winning_budget_share` | Avg budget % for top-4 finishing teams |

### FeatureStore

**Source**: `ffai/src/ffai/data/feature_store.py`

A unified loader that reads all three processed CSVs at startup. Provides lookup methods used by the state builder during training:
- `get_player_history(player_id, year)` → 6 features
- `get_position_strategy(position, year)` → 4 features
- `get_manager_tendencies(manager_id)` → per-manager profile

Missing values are imputed with position-year means. Final fallback is 0.0.

---

## 5. Component 3 — Auction Draft Environment

**Source**: `ffai/src/ffai/rl/puffer_env.py`, `ffai/src/ffai/simulation/auction_draft_simulator.py`

### Environment Overview

`AuctionDraftEnv` wraps the `AuctionDraftSimulator` as a PufferLib-native environment. One RL **step** corresponds to one bidding decision by the RL-controlled team.

```
Observation space: Box(shape=(72,), dtype=float32, low=-inf, high=inf)
Action space:      Box(shape=(1,),  dtype=float32, low=0.0, high=1.0)
```

### Auction Draft Simulator

The simulator runs a complete 12-team auction draft:

1. Teams take turns **nominating** players (round-robin)
2. Bidding starts at $1 and increments by $1 each round
3. Any team that hasn't exhausted its budget or needed roster slots can bid
4. The team left standing wins the player at the final price
5. The process repeats until all rosters are full (~168 nominations in a standard league)

**RL team integration**: When the RL team is the current bidder, the simulator yields control to the environment via the `draft_steps()` generator. The environment sends back the agent's `max_bid`, and the simulator uses it to decide whether to keep bidding as the price climbs.

### Data Caching

ESPN historical data is cached at the class level (`_SimDataCache`) so each parallel worker process only loads it once, avoiding redundant I/O during multi-environment training.

### Optional Season Simulation

After the draft completes, the environment optionally runs `SeasonSimulator` to compute final standings and wins. This terminal reward is added to the last step's reward signal. The season sim is introduced progressively in Phase 2 and 3 of curriculum training.

---

## 6. Component 4 — State Vector (72 Dimensions)

**Source**: `ffai/src/ffai/rl/state_builder.py`

The state is a 72-dimensional float32 vector passed to the policy at every bidding decision. It is structured in six semantically distinct segments:

```
dims  [0: 4]  Budget context       — financial and temporal position
dims  [4:15]  Opponent pressure    — how much budget each opponent has left
dims [15:27]  My roster state      — where the roster is complete vs. incomplete
dims [27:45]  Market state         — what's left on the board by position
dims [45:53]  Current player       — who we're bidding on right now
dims [53:56]  Bid context          — how aggressive is the current bidding?
dims [56:62]  Player history       — the player's historical performance track record
dims [62:66]  Position strategy    — position-level ROI and winning budget patterns
dims [66:72]  Opponent tendencies  — behavioral profiles of the most dangerous opponents
```

### [0:4] Budget Context

| Dim | Formula | Meaning |
|-----|---------|---------|
| 0 | `rl_budget / 200` | Fraction of budget remaining |
| 1 | `budget_per_slot / 200` | Budget ÷ remaining roster slots |
| 2 | `draft_progress` | Fraction of total nominations complete |
| 3 | `remaining_slots / 14` | Fraction of roster slots still to fill |

These four numbers tell the agent where it stands financially and how much time is left.

### [4:15] Opponent Pressure (11 dims)

One value per opponent team: `opponent_budget / 200`

When opponents are low on budget, they cannot push prices high — this is a signal to be more aggressive. When multiple opponents are flush, expect competitive bidding.

### [15:27] My Roster State (6 positions × 2 = 12 dims)

For each of the 6 standard positions (QB, RB, WR, TE, D/ST, K):

| Offset | Formula | Meaning |
|--------|---------|---------|
| 0 | `slots_filled / slots_required` | Fraction of starter slots filled (e.g., 1/2 RBs = 0.5) |
| 1 | `points_accumulated / 500` | Projected points accumulated at this position |

Tells the agent which positional needs are urgent vs. already satisfied.

### [27:45] Market State (6 positions × 3 = 18 dims)

For each position:

| Offset | Formula | Meaning |
|--------|---------|---------|
| 0 | `avg_remaining_value / 80` | Average value of remaining players at this position |
| 1 | `top_available_pts / 400` | Projected points of the best remaining player |
| 2 | `scarcity_ratio` | (total_needed − available) / 24, clipped [0, 1] |

High scarcity means if you don't pay up now, you may not find a replacement later. Low scarcity means you can wait.

### [45:53] Current Player (8 dims)

Information about the player currently being auctioned:

| Dim | Formula | Meaning |
|-----|---------|---------|
| 0 | `value_model_points / 400` | Expected season points (from PlayerValueModel) |
| 1 | `value_model_dollar / 80` | Fair auction value in dollars (from PlayerValueModel) |
| 2 | `PAR / 200` | Points Above Replacement (marginal value vs. waiver wire) |
| 3 | `VORP_dollar / 80` | Value Over Replacement Player in dollars |
| 4–7 | One-hot | Position encoding: [RB, WR, QB, other] |

### [53:56] Bid Context (3 dims)

| Dim | Formula | Meaning |
|-----|---------|---------|
| 0 | `current_bid / 200` | Current highest bid as fraction of max budget |
| 1 | `current_bid / fair_value` | Bid vs. fair value (>1.0 = market is overpaying) |
| 2 | `min_needed_budget / 200` | Minimum needed to fill remaining slots at $1 each |

The ratio in dim 1 is particularly powerful: when it exceeds 1.0, the market is running hot and the agent should generally back off. When it's well below 1.0, this is a potential bargain.

### [56:62] Player History (6 dims)

Historical performance data for the current player from `FeatureStore` (lookback-safe):

| Dim | Formula | Meaning |
|-----|---------|---------|
| 0 | `pts_3yr_avg / 400` | Mean season points over prior 3 years |
| 1 | `proj_ratio_3yr_avg / 3.0` | How well ESPN projects this player (>1 = consistently outperforms) |
| 2 | `yoy_pct_change` | Year-over-year trajectory, clipped to [-1, 1] |
| 3 | `weekly_pts_cv / 3.0` | Consistency (lower = more reliable week-to-week) |
| 4 | `floor_pts / 40` | 25th-percentile weekly points (downside risk) |
| 5 | `years_in_league_norm` | Experience in the league, normalized to [0, 1] over 10 years |

These features let the agent distinguish between a player projecting 200 points with a proven track record vs. one projecting 200 points as a sophomore with high variance.

### [62:66] Position Strategy (4 dims)

League-level signals about the current player's position, from `FeatureStore`:

| Dim | Formula | Meaning |
|-----|---------|---------|
| 0 | `winning_budget_share` | % of budget winning teams historically spent on this position |
| 1 | `roi_mean / 20` | Points per dollar for this position (clipped to [0, 1]) |
| 2 | `proj_accuracy_ratio / 3.0` | How well ESPN projections match reality at this position |
| 3 | `budget_share_pct` | % of total league budget flowing to this position |

The `winning_budget_share` is especially informative: if winning teams historically spent 35% of their budget on RBs but you've only spent 10%, you're falling behind on a proven winning pattern.

### [66:72] Opponent Tendencies (6 dims)

Aggregated behavioral profile of the **top 3 opponents by remaining budget** (the most dangerous bidders):

| Dim | Formula | Meaning |
|-----|---------|---------|
| 0 | `avg_rb_budget_share` | How much these opponents typically spend on RBs |
| 1 | `avg_wr_budget_share` | How much these opponents typically spend on WRs |
| 2 | `avg_high_bid_rate` | Fraction of their picks that cost >$30 (aggression metric) |
| 3 | `avg_dollar_one_rate` | Fraction of their picks costing $1 (sleeper-hunting rate) |
| 4 | `avg_bid_per_proj_pt_rb` | How much they pay per projected RB point |
| 5 | `avg_bid_per_proj_pt_wr` | How much they pay per projected WR point |

By focusing on the top-3 budget opponents, the agent gets a concentrated signal about who can realistically push prices up, rather than averaging over managers who are nearly out of money.

---

## 7. Component 5 — PPO Policy Network

**Source**: `ffai/src/ffai/rl/puffer_policy.py`

### Architecture

```
Input: (batch, 72)
    |
encoder (shared trunk):
  Linear(72  → 256) + LayerNorm + ReLU
  Linear(256 → 128) + LayerNorm + ReLU
    |                    |
policy head:         value head:
  Linear(128 → 1)    Linear(128 → 1)
  sigmoid()          → scalar V(s)
  → mean ∈ (0, 1)
  global log_std (learnable, clamped [-4.0, 1.0])
  → Normal(mean, exp(log_std))
```

### Design Choices

**Sigmoid output mean**: The bid fraction must be in [0, 1]. Using sigmoid on the policy head mean guarantees the center of the distribution stays in-bounds, while the Gaussian still allows the sampled value to be clipped at the environment level.

**Global log_std**: The exploration spread is a single learnable parameter, not state-dependent. This is standard for PPO and keeps the action distribution simple. The clamp range [-4.0, 1.0] allows std ∈ [0.018, 2.72], covering both precise exploitation and broad exploration.

**LayerNorm after each hidden layer**: Stabilizes gradients with the heterogeneous 72-dim input (mixing normalized fractions, one-hot encodings, and raw counts). Particularly important during curriculum transitions when the reward scale changes.

**PufferLib integration**: The policy's `forward_eval()` method returns `(Normal distribution, value tensor)` consumed directly by PufferLib's PPO update. Log-probability and entropy are computed automatically by PufferLib from the distribution object.

---

## 8. Component 6 — Reward Function

**Source**: `ffai/src/ffai/rl/reward.py`

### Mid-Draft Reward (per winning pick)

Issued each time the RL team wins a player at auction:

```
base = (fair_dollar_value - bid_amount) / 200.0

if roster_needs_this_position:
    base += 0.1       # bonus for strategic fit

if remaining_budget < min_needed_to_fill_roster:
    base -= 0.2       # penalty for financial danger

return base
```

**Range**: approximately [-0.7, +0.6]

**Logic**: The agent is rewarded proportionally to the discount it achieves on a player's fair value. Paying $20 for a $30 player earns +0.05; overpaying $10 for that same player earns -0.05. The positional need bonus incentivizes roster completion over hoarding value at a single position. The budget safety penalty teaches the agent to avoid overbidding early and running out of money for later picks.

### Terminal Reward (end of draft, season sim enabled)

Computed after simulating a full 17-week fantasy season:

```
standing_rewards = {1st: 5.0, 2nd: 3.0, 3rd: 2.0, 4th: 1.0, else: 0.0}
win_reward       = total_wins × 0.3
raw              = standing_rewards[final_place] + win_reward
normalized       = raw / 10.1          # ≈ [0, 1]
```

**Range**: approximately [0, 1] after normalization

**Logic**: The terminal reward ties the draft outcome to actual fantasy performance. A team that drafts efficiently but ends up with an unbalanced roster (e.g., no RBs) will get punished in the season sim. This long-horizon signal prevents the agent from optimizing only for per-pick discount without regard for roster composition.

### Reward Scaling

The mid-draft reward is designed to be on a similar magnitude to per-pick terminal reward components (wins × 0.03 per pick). This ensures that neither signal dominates during the PPO update.

---

## 9. Component 7 — Opponent Modeling

**Source**: `ffai/src/ffai/simulation/auction_draft_simulator.py`

Realistic opponent behavior is essential: an agent trained against random or uniform opponents would learn exploits that fail against real humans.

### Per-Manager Bidding Model

For each opponent team, the simulator looks up that manager's historical profile in `manager_tendencies.csv`:

1. Retrieve the manager's `bid_per_proj_pt_{position}` efficiency metric
2. Compute `max_bid = projected_points × (bid_per_proj_pt + small_noise)`
3. Fall back to league-average tendencies if no profile exists
4. Final fallback: `auction_value × (1 ± 10% random)`

This means managers who historically overpay for RBs will continue to do so in simulation, and managers who hunt for value will reflect that in their bids. The noise term prevents degenerate memorization of opponent thresholds.

### Team-to-Manager Mapping

The simulator extracts a mapping from generic "Team N" draft slots to actual `manager_id` values from the ESPN draft data. This allows per-manager profiles to be attached to each simulated opponent team correctly.

### Heuristic Bidding Bug Fix

The original `should_bid()` function capped opponents' willingness-to-pay at `player.get("auction_value", 1)`. Because the ESPN pre-draft API returns `auction_value = 0` for most bench players, teams were dropping out of nearly every auction at $1, leaving their full $200 budget unspent. The fix removes this cap — `should_bid()` now only checks budget safety (can the team afford to pay without running out for later picks?). The actual max willingness-to-pay is determined by `_opponent_max_bid()`, which removes teams from the active bidders set when they reach their ceiling.

### Self-Play Opponents (Phase 4)

In Phase 4, heuristic opponents are replaced by a mixture of:
- **Learned checkpoints** (70%): past snapshots of the RL policy, loaded from the `OpponentPool` checkpoint pool
- **Heuristic bidders** (30%): the per-manager behavioral models from `manager_tendencies.csv`

The pool is seeded with either a behavioral cloning (BC) reference model (Stage 2) or the phase3 final checkpoint. Every 50K training steps, a snapshot of the current policy is added to the pool (FIFO, max 10 checkpoints).

A learned policy trained to maximize season standing naturally learns to deploy its full $200 budget, because:
- Every unspent dollar represents forgone player value
- PPO penalizes strategies that leave budget on the table (→ worse rosters → worse standings)
- As checkpoint policies improve, they bid more competitively, raising market prices

This creates a virtuous cycle: better opponents → harder training → better RL policy → better opponents.

---

## 10. Training — Four-Phase Curriculum

**Source**: `ffai/scripts/train_puffer.py`, `ffai/scripts/train.py`

Training uses a curriculum to progressively increase difficulty and reward complexity.

| Phase | Timesteps | Season Sim | Season Sim Interval | Opponents | Num Envs | Purpose |
|-------|-----------|------------|---------------------|-----------|----------|---------|
| 1 | 100K | No | — | Heuristic | 1 | Draft mechanics warm-up; validate pipeline |
| 2 | 500K | Yes | Every 10 episodes | Heuristic | 4 | Introduce season sim; learn draft→season link |
| 3 | 1M | Yes | Every episode | Heuristic | 6 | Full season feedback; optimize for standing |
| 4 | 500K | Yes | Every episode | Pool (70% ckpt / 30% heuristic) | 6 | Self-play; competitive opponents |

### Phase 1: Draft Warm-Up

The agent receives only mid-draft rewards. This forces it to learn auction fundamentals:
- Don't overpay for any single player
- Complete the roster (don't run out of money)
- Recognize value vs. overprice

Without season feedback, Phase 1 is a purely financial optimization problem. The policy quickly learns to avoid extreme bids and roughly match fair market value.

### Phase 2: Season Sim Introduced

Every 10 episodes, the completed draft roster is fed into the season simulator, and a terminal reward based on final standings is added. This teaches the agent that:
- Winning a lot of cheap players doesn't help if they're all at the same position
- Roster balance (spread across positions) matters for season outcomes
- There is a direct connection between draft decisions and end-of-season standing

The 10-episode interval keeps the terminal signal sparse initially so it doesn't overwhelm the more frequent mid-draft reward.

### Phase 3: Full Season Feedback

Every draft episode generates a terminal reward. The agent is now optimizing directly for the long-horizon objective: win the league. It must balance per-pick value efficiency against roster composition and positional need.

### Phase 4: Self-Play

Opponents are drawn from `OpponentPool` — a FIFO pool of past policy checkpoints. Each episode samples 11 opponent policies (one per opposing team), with 70% drawn from the pool and 30% using the heuristic bidder for diversity. A snapshot of the current policy is added to the pool every 50K steps. The pool is seeded with either the BC reference model (human-realistic baseline) or the phase3 final checkpoint.

Learned opponents naturally learn to spend their full $200 budget (leaving money unspent means a weaker roster and worse standings), creating a more adversarial and realistic training environment than the static heuristic opponents in Phases 1–3.

### Checkpoint Loading

Phases are loaded sequentially:
```
phase1_final.pt  →  phase2 initialization
phase2_final.pt  →  phase3 initialization
phase3_final.pt  →  phase4 initialization + initial opponent pool seed
```

This warm-starting prevents wasted compute: each phase begins with the skills developed in previous phases.

### End-to-End Launcher

`ffai/scripts/train.py` chains all phases in sequence:
```bash
.venv/bin/python ffai/scripts/train.py              # all phases
.venv/bin/python ffai/scripts/train.py --from-phase 3  # resume from Phase 3
.venv/bin/python ffai/scripts/train.py --skip-bc    # skip BC reference training
```

---

## 11. Training Hyperparameters

**Source**: `ffai/src/ffai/config/puffer.ini`

| Parameter | Value | Notes |
|-----------|-------|-------|
| Optimizer | Adam | β₁=0.9, β₂=0.999, ε=1e-8 |
| Learning rate | 3e-4 | Annealed to 10% by end of training |
| Discount factor γ | 0.99 | Near-fully delayed reward |
| GAE λ | 0.95 | Advantage estimation smoothing |
| PPO clip ε | 0.2 | Standard conservative clip |
| Value function coef | 0.5 | Balances policy vs. value losses |
| Entropy coef | 0.01 | Encourages exploration of bid range |
| Update epochs | 4 | Reuses each batch 4× before discarding |
| Batch size | 3072 | Steps collected before each update |
| BPTT horizon | 64 | Sequence length for recurrent-style rollouts |
| Minibatch size | 512 | SGD minibatch within each update epoch |
| Max grad norm | 0.5 | Gradient clipping for stability |

**Batch sizing note**: The constraint `batch_size / bptt_horizon` must be evenly divisible by `num_envs` across all phases. LCM(1, 4, 6) = 12; 3072 / 64 = 48; 48 % 12 = 0. This is verified to hold so PufferLib's multiprocessing backend can evenly distribute segments across workers.

---

## 12. Episode Lifecycle

A single training episode corresponds to one complete 12-team auction draft.

```
env.reset()
    │
    ├─ Creates AuctionDraftSimulator (from cached ESPN data)
    ├─ Initializes 12 teams, available player pool, roster slots
    ├─ Starts draft_steps() generator
    └─ Receives first (state, player, bid) → builds 72-dim observation

env.step(action)  [repeated per RL bidding decision]
    │
    ├─ Decodes action: max_bid = clip(action[0], 0, 1) × remaining_budget
    ├─ Sends max_bid to generator: next state = gen.send(max_bid)
    ├─ Simulator runs bidding round internally (opponents bid, increment price)
    ├─ If RL team wins player: sets _step_reward = mid_draft_reward(...)
    └─ Returns (next_obs, reward, done, truncated, info)

On StopIteration (draft complete):
    │
    ├─ terminal = True
    ├─ [Optional] Run SeasonSimulator → add terminal_reward
    └─ Auto-reset: start next episode, store first obs

Typical episode length: ~50–200 RL decisions
  (depends on how often the RL team is the active bidder)
```

---

## 13. Using the Trained Model

### Offline Evaluation (simulate_draft.py)

Run a full simulated 12-team draft with per-manager opponent models and print a play-by-play transcript:

```bash
# All-heuristic baseline — per-manager opponents, no RL agent
.venv/bin/python ffai/scripts/simulate_draft.py --year 2024

# Inject a trained RL model as Team 1
.venv/bin/python ffai/scripts/simulate_draft.py \
  --year 2024 \
  --rl-model-path checkpoints/puffer/phase3_final.pt \
  --rl-team "Team 1"

# Save transcript to file
.venv/bin/python ffai/scripts/simulate_draft.py --year 2024 \
  --rl-model-path checkpoints/puffer/phase3_final.pt \
  --output results/sim_2024.txt
```

The transcript shows each nomination, winner, final price, and complete team rosters with projected points and spend.

### Advisory Mode (advisory_draft.py)

Monitors a live ESPN draft and prints bidding recommendations at each decision. You retain full control; the agent advises.

```bash
.venv/bin/python ffai/scripts/advisory_draft.py \
  --team-id 1 \
  --value-model checkpoints/value_model/best_model.pt
```

### Autonomous Mode (autonomous_draft.py)

Controls the ESPN draft room directly via Playwright browser automation. Always test in a mock league first.

```bash
.venv/bin/python ffai/scripts/autonomous_draft.py \
  --team-id 1 \
  --ppo-checkpoint checkpoints/puffer/phase3_final.pt
```

The browser window stays visible so you can monitor and intervene. A dead-man's switch falls back to advisory mode if the ESPN UI fails 10 consecutive interactions.

### Interpreting Model Decisions

At each bidding decision, the policy outputs a **bid fraction** from its Gaussian distribution. The mean of this distribution represents the agent's "ideal" bid as a fraction of its remaining budget. For example:
- Mean ≈ 0.05: "I'd pay up to 5% of my remaining budget for this player" — relatively passing
- Mean ≈ 0.25: "I'd pay up to 25% of my remaining budget" — aggressive pursuit
- Mean ≈ 0.01: "I'm effectively passing" — willing to let this player go

During inference (advisory/autonomous mode), the **mean** of the distribution is used directly (no sampling), making the policy deterministic for repeatable live-draft decisions.

---

## 14. Key Design Decisions

### Why Auction Draft, Not Snake Draft?

Auction drafts have a much richer decision space. In a snake draft, your only decision is which player to pick. In an auction, you must decide:
- How much to bid on each player (continuous action)
- Whether to stay in or drop out as price climbs (implicit in bid ceiling)
- When to save budget vs. spend aggressively

This makes auction drafts an ideal RL problem — the action space is meaningful, the consequences are delayed, and expert strategy is far from trivial.

### Why a Continuous Action Space?

A discrete "bid $N" formulation would require a large number of actions ($1–$200) and make credit assignment harder. Using a continuous bid fraction scales naturally with the budget, handles different budget scenarios uniformly, and is well-suited to Gaussian policy gradients.

### Why Curriculum Learning?

Starting with the full terminal reward (season standing) is difficult because the signal is sparse — you make ~100 picks over 30+ minutes, then receive one terminal reward. The agent would struggle to connect early bidding decisions to the late-season outcome.

The curriculum lets the agent first learn auction mechanics via dense per-pick rewards, then incrementally introduces the sparse long-horizon signal. This mirrors how a human might learn: first understand value vs. price, then think about roster strategy, then optimize for winning.

### Why Lookback-Safe Features?

All historical features use data strictly from years prior to the target year. This is essential for training validity: if features for the 2022 draft used 2022 performance data, the model would learn to use information that isn't available at draft time. Lookback safety ensures the model only uses what a drafter would actually know.

### Why Per-Manager Opponent Models?

Uniform random or average-bidder opponents would not reflect the structured patterns real managers exhibit — some always overpay for RBs, some hunt for $1 sleepers, some have fixed position priorities. Per-manager models produce a training environment closer to the real competitive landscape the agent will face at live draft time.

---

## 15. Extending the System

**SPACeR-style BC Anchoring (Stage 2)**: The BC reference model (`bc_reference.py`) trained on historical ESPN data can be used as an auxiliary PPO reward signal (`bc_alpha × log P(action | state, θ_ref)` added to mid-draft reward) to prevent the policy from drifting into degenerate bidding strategies. The `bc_alpha` parameter is configured in `puffer.ini [self_play]` (default 0.05). This stage is not yet integrated into `mid_draft_reward()` but the BC model and checkpoint export are implemented.

**Recurrent Policy (LSTM/Attention)**: The current MLP policy sees only the current state snapshot. An LSTM would allow the agent to reason about bidding dynamics over time within a single draft episode — e.g., "this opponent has bid aggressively on the last 3 players, so they're probably running low."

**Nomination Strategy**: The agent currently uses a heuristic to decide which player to nominate. Learning a nomination policy (which player to put up for auction and when) is a significant additional strategic layer.

**Positional Scarcity Forecasting**: Replace the current static market state with a learned scarcity predictor that estimates which positions will run scarce as the draft progresses.

**League Generalization**: The current model is trained on a single ESPN league's historical data. Training across multiple leagues with different scoring formats, roster requirements, and manager styles would produce a more robust policy.

**Real-Time Evaluation**: Close the feedback loop by tracking real-season outcomes and fine-tuning the model at the end of each fantasy season with actual performance data.
