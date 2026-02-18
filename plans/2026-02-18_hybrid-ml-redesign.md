# Fantasy Football Auction Draft AI — Redesign Plan

## Context

Two prior experimental repos exist:
- **ffai** (Feb 18-20, 2025) — RL simulation approach; newer, more architecturally ambitious (~1,700 lines), but RL not converging (finishing 6-10th of 12 teams). Has critical bugs. This is the **base repo**.
- **favrefignewton** (Feb 16-17, 2025) — Supervised learning approach; simpler, has a working trained model but no bidding strategy optimization.

Goal: Design a hybrid system that uses **supervised learning for player value estimation** and **PPO reinforcement learning for bidding strategy optimization**, deployable as either a real-time advisory tool (show human recommendations) or an autonomous agent (control the draft via browser automation).

**Base repo:** `/home/dave/code/davidcasterton/ffai/ffai/`

---

## Why RL Wasn't Converging (Critical Bugs to Fix First)

1. **Broken backprop**: `rl_model.py`'s `update()` does `loss = -reward` and calls `.backward()` on a scalar with no computational graph attached to model parameters. **Gradients are zero. Model weights never change.**
2. **State/network mismatch**: `prepare_state_tensor()` builds a 190-dim vector but `forward()` only consumes dims 0-44. 145 features are built and ignored.
3. **Wrong algorithm**: Action space is continuous (bid dollar amount). DQN requires discrete actions. Replace with PPO (Gaussian policy over continuous bids).
4. **No rollout buffer**: Per-pick updates cannot do credit assignment across a full draft. PPO must accumulate full episode rollouts before updating.
5. **pdb breakpoint**: `auction_draft_simulator.py:114` — blocks all training when round limit is hit.

---

## New Architecture

### Hybrid Pipeline

```
ESPN Historical Data (2009-2024)
         |
         v
[PlayerValueModel]  <-- supervised, two-head output:
  - expected_season_points
  - auction_dollar_value (PAR/VORP-scaled)
         |
         v
[PPO Agent]  <-- uses value model outputs as state features
  State: budget, roster needs, player PAR, opponent budgets, market scarcity
  Action: bid amount (continuous Gaussian policy)
  Reward: value_efficiency (mid-draft) + season standing (terminal)
         |
         v
[Interfaces]
  Advisory Mode: poll ESPN API, display recommendations in terminal
  Autonomous Mode: Playwright browser automation of ESPN draft room
```

---

## Directory Structure (in `ffai/src/ffai/`)

```
ffai/src/ffai/
├── config/
│   ├── league.yaml              # ESPN credentials + league_id (gitignored)
│   └── training.yaml            # hyperparameters
├── data/
│   ├── espn_scraper.py          # MODIFY: externalize credentials to config
│   └── preprocessor.py          # NEW: cherry-pick from favrefignewton, add PAR/VORP
├── value_model/
│   ├── player_value_model.py    # NEW: two-head supervised model (points + dollar value)
│   └── value_trainer.py         # NEW: training loop
├── rl/
│   ├── ppo_agent.py             # NEW: replaces rl_model.py; Actor-Critic PPO
│   ├── state_builder.py         # NEW: centralized state construction (~56 dims)
│   ├── reward.py                # NEW: all reward logic in one place
│   └── replay_buffer.py         # NEW: GAE rollout buffer
├── simulation/
│   ├── auction_draft_simulator.py  # MODIFY: fix pdb bug + position_counts key bug
│   └── season_simulator.py         # keep as-is
└── interfaces/
    ├── live_draft_reader.py     # NEW: poll ESPN REST API every 2-3s for draft state
    ├── advisory.py              # NEW: terminal UI showing recommendations to human
    └── autonomous.py            # NEW: Playwright agent to control ESPN draft room
```

**Scripts:**
- `scripts/collect_data.py` — multi-year data collection
- `scripts/train_value_model.py` — supervised training
- `scripts/train_rl.py` — PPO training with phase flag
- `scripts/advisory_draft.py` — launch advisory mode
- `scripts/autonomous_draft.py` — launch autonomous mode

---

## Key Components

### PlayerValueModel (supervised, `value_model/player_value_model.py`)
- **Architecture**: player_embedding (256-dim) + position_embedding (64-dim) + numerical_encoder MLP([5] → [128]) for [projected_pts, adp, year, position_scarcity_rank, etc]
- **Two heads**: `points_head` (expected season pts) + `value_head` (fair auction $)
- **Training data**: historical ESPN data 2009-2022 train, 2023 val, 2024 test
- **Target**: MSE on `total_points` and `bid_amount` from historical drafts
- **Do NOT load** the `favrefignewton/model_checkpoints/best_model.pt` — incompatible architecture; use it as a reference only

### PPO Agent (`rl/ppo_agent.py`)
- **Actor**: `AuctionDraftActor` — state_encoder MLP → bid_mean + bid_log_std (Gaussian) + nomination_logits
- **Critic**: `AuctionDraftCritic` — state_encoder MLP → V(s) for advantage estimation
- **Updates**: Collect full episode rollout, compute GAE advantages, PPO clip loss (ε=0.2), 4 mini-batch epochs per update

### State Vector (`rl/state_builder.py`, ~56 dims)
| Group | Dims | Content |
|-------|------|---------|
| Budget context | 4 | rl_budget/200, budget_per_slot, draft_progress, remaining_slots |
| Opponent pressure | 11 | opponent_budget_fractions |
| My roster state | 12 | per-position: [slots_filled_fraction, points_accumulated] |
| Market state | 18 | per-position: [avg_remaining_value, top_available_pts, scarcity_ratio] |
| Current player | 8 | value_model_points_hat, value_model_dollar_hat, PAR, VORP_dollar, position onehot |
| Bid context | 3 | current_bid/200, current_bid/fair_value, min_needed_budget |

### Reward Shaping (`rl/reward.py`)
- **Mid-draft** (per pick): `(fair_dollar_value - bid_amount) / 200` ∈ [-0.5, +0.5] + position need bonus (+0.1) + budget safety penalty (-0.2)
- **Terminal** (end of season): standing reward {1st: 5.0, 2nd: 3.0, 3rd: 2.0, 4th: 1.0} + wins × 0.3
- *All rewards normalized to same scale* — prior implementation had values ranging from 200 to 500 in the same reward space

### Live Draft Reader (`interfaces/live_draft_reader.py`)
- Poll: `GET https://lm-api-reads.fantasy.espn.com/apis/v3/games/ffl/seasons/{year}/segments/0/leagues/{league_id}?view=mDraftDetail`
- Auth via cookies (SWID, espn_s2) loaded from `config/league.yaml`
- Every 2-3 seconds (within 60-90s nomination timers)
- `convert_to_simulation_state()` maps API response to same dict format used in simulation

### Autonomous Mode (`interfaces/autonomous.py`)
- Playwright (not Selenium) — better async SPA support
- `headless=False` for monitoring
- Dead-man's switch: if DOM selectors fail 10 consecutive polls, alert + fall back to advisory mode
- ESPN UI selectors will break on UI updates; treat as fragile, log all interactions

---

## Training Sequence

### Stage 1 — Data Collection
```bash
python scripts/collect_data.py --years 2009-2024
```

### Stage 2 — Train Value Model
```bash
python scripts/train_value_model.py --train-years 2009-2022 --val-year 2023 --test-year 2024
```
Expected: RMSE ~5-6 pts for season points, MAE ~$5-8 for auction value.

### Stage 3 — Curriculum Phase 1: Heuristic Baseline (500 episodes)
Run value model output as heuristic (bid = `fair_value * 0.95`). No PPO learning. Confirm agent finishes top-6 consistently — validates value model and simulation integration before adding RL.

### Stage 4 — Curriculum Phase 2: PPO Training (2000 episodes)
```bash
python scripts/train_rl.py --phase 2 --episodes 2000 --ppo-lr 3e-4 --gamma 0.99 --gae-lambda 0.95
```
Monitor: average standing should move from ~6 → ~3-4 by episode 1500.

### Stage 5 — Curriculum Phase 3: Self-play (3000 episodes)
Replace some auto-draft opponents with frozen PPO checkpoints from earlier training. Prevents overfitting to the fixed ±10% ESPN opponent model.

---

## Verification

1. **Bug fixes**: Run `python scripts/auction.py` — should complete a full 12-team draft without hitting pdb or crashing on position_counts key
2. **Value model**: After training, spot-check `fair_dollar_value` for top players — Patrick Mahomes-tier QBs should price at $40-60, top RBs at $50-70
3. **PPO learning**: Plot `avg_standing_position` per episode over 2000 episodes — should show downward trend (better placement)
4. **Advisory mode**: Run against a historical 2024 draft replay — check that recommendations match intuitive auction values
5. **Autonomous mode**: Test against ESPN draft lobby in a mock/test league before using in real league

---

## Deferred / Out of Scope for Now
- Multi-league support (single league for training and deployment)
- Proper test suite (currently no tests in either repo)
- Web UI (terminal output is sufficient for advisory mode)
- Free agent / waiver wire strategy (draft focus only)
