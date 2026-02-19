# ffai — Fantasy Football Auction Draft AI

Hybrid ML system for ESPN fantasy football auction drafts. Uses supervised learning to estimate player value and PPO reinforcement learning (via PufferLib) to optimize bidding strategy. Can run as a real-time advisory tool or as an autonomous agent that controls the ESPN draft room via browser automation.

## Architecture

```
ESPN Historical Data (2009-2024)          nflverse Data (2009-2024)
         |                                          |
         v                                          v
[collect_data.py]                       [fetch_nfl_data.py]
  → data/{league_name}/                  → data/nflverse/*.parquet
         |                                          |
         +------------------+-----------------------+
                            |
                            v
                  [build_features.py]
                    → data/{league_name}_processed/
                      player_history_{id}.csv       (3yr pts avg, consistency, YoY)
                      manager_tendencies_{id}.csv   (per-manager bidding profiles)
                      position_strategy_{id}.csv    (ROI, winning budget shares)
                            |
                            v
                  [train_value_model.py]
                  [PlayerValueModel] — supervised, two-head output:
                    - expected_season_points
                    - fair_auction_dollar  (PAR/VORP-scaled)
                    → checkpoints/value_model/best_model.pt
                            |
                            v
                  [train_puffer.py] — PufferLib PPO, 3-phase curriculum
                  [AuctionDraftPolicy] — Gaussian policy, 72-dim state
                    State:  budget, roster needs, player PAR, opponent budgets,
                            market scarcity, player history, position strategy,
                            opponent tendencies
                    Action: bid fraction ∈ [0, 1]  →  max_bid = fraction × budget
                    Reward: value_efficiency (mid-draft) + season standing (terminal)
                    → checkpoints/puffer/phase{1,2,3}_final.pt
                            |
                            v
                  [Interfaces]
                    Advisory Mode:    poll ESPN API, display recommendations in terminal
                    Autonomous Mode:  Playwright browser automation of ESPN draft room
```

## Directory structure

```
ffai/
├── scripts/
│   ├── collect_data.py         # Stage 1: download ESPN draft/stats data
│   ├── fetch_nfl_data.py       # Stage 2: download nflverse parquets
│   ├── build_features.py       # Stage 3: build processed feature CSVs
│   ├── analyze_strategy.py     # exploratory: position ROI, manager tendencies, projection bias
│   ├── train_value_model.py    # Stage 4: train supervised value model
│   ├── train_puffer.py         # Stage 5: PufferLib PPO curriculum training
│   ├── simulate_draft.py       # post-training: full draft simulation with transcript output
│   ├── advisory_draft.py       # live draft advisory mode
│   └── autonomous_draft.py     # autonomous ESPN draft room control
└── src/ffai/
    ├── config/
    │   ├── league.yaml          # ESPN credentials + league_id (gitignored)
    │   ├── league.yaml.example  # template
    │   ├── training.yaml        # value model hyperparameters
    │   └── puffer.ini           # PufferLib PPO hyperparameters
    ├── data/
    │   ├── espn_scraper.py      # ESPN API client
    │   ├── preprocessor.py      # PAR/VORP calculation, 14-feature vector
    │   ├── feature_store.py     # unified loader for processed CSVs (with imputation)
    │   ├── feature_engineering/
    │   │   ├── player_history.py       # 3yr avg pts, YoY change, projection ratio
    │   │   ├── weekly_stats_parser.py  # consistency: CV, floor, ceiling
    │   │   ├── nfl_feature_builder.py  # targets/game, carries/game, snap %, draft round
    │   │   ├── manager_tendencies.py   # per-manager budget shares, bid rates
    │   │   └── position_strategy.py    # per-(position, year) ROI and winning patterns
    │   ├── {league_name}/       # ESPN raw data (gitignored, local only)
    │   ├── nflverse/            # nflverse parquets (git LFS)
    │   └── {league_name}_processed/   # derived feature CSVs (gitignored, local only)
    ├── value_model/
    │   ├── player_value_model.py  # two-head supervised model (points + dollar)
    │   └── value_trainer.py       # training loop with early stopping
    ├── rl/
    │   ├── puffer_env.py        # AuctionDraftEnv(PufferEnv) — PufferLib native env
    │   ├── puffer_policy.py     # AuctionDraftPolicy — Gaussian policy (72-dim state)
    │   ├── state_builder.py     # build_state() → 72-dim float32 tensor
    │   ├── reward.py            # mid_draft_reward(), terminal_reward()
    │   ├── ppo_agent.py         # legacy custom PPO (superseded by puffer pipeline)
    │   └── replay_buffer.py     # GAE rollout buffer (legacy)
    ├── simulation/
    │   ├── auction_draft_simulator.py  # 12-team auction draft engine + draft_steps() generator
    │   └── season_simulator.py         # 17-week season simulator
    └── interfaces/
        ├── live_draft_reader.py  # polls ESPN REST API during live drafts
        ├── advisory.py           # terminal UI showing recommendations
        └── autonomous.py         # Playwright browser control
```

## Setup

```bash
# Create and activate virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install ffai in editable mode
pip install -e ffai/

# Install PufferLib from source (PyPI version has PyTorch ABI mismatch)
pip install -e /path/to/PufferLib/

# Only needed for autonomous mode
playwright install chromium
```

**Configure credentials:**

```bash
cp ffai/src/ffai/config/league.yaml.example ffai/src/ffai/config/league.yaml
# Edit league.yaml with your ESPN league_id, league_name, SWID, and espn_s2 cookies.
# league_name determines the data directory (data/{league_name}/).
# Find SWID and espn_s2 in browser DevTools > Application > Cookies
# after logging into ESPN Fantasy Football.
```

## Training pipeline

All commands are run from the repo root (`/path/to/ffai/`) using the venv python.

### Stage 1 — Collect ESPN historical data

```bash
.venv/bin/python ffai/scripts/collect_data.py --years 2009-2024
```

Downloads draft results, player stats, pre-draft values, and weekly scoring from ESPN for each year. Cached under `ffai/src/ffai/data/{league_name}/` (where `league_name` comes from `config/league.yaml`).

### Stage 2 — Download nflverse data

```bash
.venv/bin/python ffai/scripts/fetch_nfl_data.py --seasons 2009-2024
```

Downloads 6 nflverse datasets (player stats, rosters, snap counts, draft picks, player ID crosswalk) as parquet files to `ffai/src/ffai/data/nflverse/`. These are tracked in git LFS so teammates get them automatically on clone.

### Stage 3 — Build feature CSVs

```bash
.venv/bin/python ffai/scripts/build_features.py --years 2009-2024
```

The `--league-id` argument defaults to the value in `config/league.yaml`. Produces 3 CSVs in `ffai/src/ffai/data/{league_name}_processed/`:
- `player_history_{id}.csv` — per-(player_id, year): 3yr avg points, YoY change, projection accuracy, weekly consistency
- `manager_tendencies_{id}.csv` — per-manager: RB/WR budget shares, bid aggressiveness, $1-bid rate
- `position_strategy_{id}.csv` — per-(position, year): ROI, projection accuracy, winning team budget shares

### Stage 3b — (Optional) Analyze strategy

```bash
.venv/bin/python ffai/scripts/analyze_strategy.py
```

Prints 4 analyses: position ROI over 16 years, manager efficiency table, ESPN projection bias by position, and Spearman correlation of budget allocation with final standing.

### Stage 4 — Train the value model

```bash
.venv/bin/python ffai/scripts/train_value_model.py \
  --train-years 2009-2022 \
  --val-year 2023 \
  --test-year 2024
```

Trains a two-head neural network on 14 features per player per year:
- **Points head**: predicts total season fantasy points
- **Value head**: predicts fair auction dollar value (derived from PAR/VORP)

Best checkpoint saved to `checkpoints/value_model/best_model.pt`.

### Stage 5 — Train the PPO agent (three-phase curriculum)

Training uses PufferLib's vectorized PPO with multiprocessing workers. Each episode is a full 12-team auction draft (~168 total nominations). The RL agent makes one decision per bidding round it participates in — it is polled every round a nomination is in progress and it hasn't dropped out. For contested players it may make several decisions as the price climbs; for players it passes on it makes zero. Total RL decisions per episode is typically in the range of 50-200. Phase checkpoints are loaded sequentially.

**Phase 1 — draft warm-up (100K steps, no season sim)**

```bash
.venv/bin/python ffai/scripts/train_puffer.py --curriculum-phase 1
```

Fast validation that the pipeline works. No season simulation — reward is draft-only value efficiency. Checkpoint saved to `checkpoints/puffer/phase1_final.pt`.

**Phase 2 — season sim introduced (500K steps)**

```bash
.venv/bin/python ffai/scripts/train_puffer.py --curriculum-phase 2 \
  --load-model-path checkpoints/puffer/phase1_final.pt
```

Season simulator runs every 10 episodes per worker (~20% of episodes get terminal reward). Checkpoint saved to `checkpoints/puffer/phase2_final.pt`.

**Phase 3 — full season sim (1M steps)**

```bash
.venv/bin/python ffai/scripts/train_puffer.py --curriculum-phase 3 \
  --load-model-path checkpoints/puffer/phase2_final.pt
```

Season simulator runs every episode. Checkpoint saved to `checkpoints/puffer/phase3_final.pt`.

**Quick smoke test (500 steps):**

```bash
.venv/bin/python ffai/scripts/train_puffer.py --curriculum-phase 1 --total-timesteps 500
```

## Usage

### Simulate a draft (post-training evaluation)

Run a full 12-team auction draft simulation and print a play-by-play transcript plus final rosters. Uses per-manager behavioral models from historical bid data for all opponent teams.

```bash
# All-heuristic baseline — per-manager opponents, no RL agent
.venv/bin/python ffai/scripts/simulate_draft.py --year 2024

# Inject a trained RL model as Team 1 (or any team slot)
.venv/bin/python ffai/scripts/simulate_draft.py \
  --year 2024 \
  --rl-model-path checkpoints/puffer/phase3_final.pt \
  --rl-team "Team 1"

# Save output to a file
.venv/bin/python ffai/scripts/simulate_draft.py --year 2024 \
  --rl-model-path checkpoints/puffer/phase3_final.pt \
  --output results/sim_2024.txt
```

The transcript shows each nomination, winner, and final price. The roster section shows every team's picks with projected points and spend. Verification checks: 168 rows in the transcript, all 12 teams with full rosters, total spend ≤ $200 per team.

### Advisory mode

The agent monitors the live draft and prints bidding recommendations. You control the draft; the agent advises.

```bash
.venv/bin/python ffai/scripts/advisory_draft.py \
  --team-id 1 \
  --value-model checkpoints/value_model/best_model.pt
```

### Autonomous mode

The agent controls the ESPN draft room directly via Playwright. Always test in a mock league first.

```bash
.venv/bin/python ffai/scripts/autonomous_draft.py \
  --team-id 1 \
  --ppo-checkpoint checkpoints/ppo/ppo_checkpoint_2000.pt
```

The browser window stays visible (`headless=False`) so you can monitor and intervene. A dead-man's switch falls back to advisory mode if ESPN's UI selectors fail 10 times consecutively.

## Plans

- `plans/2025-02-18_initial-design.md` — original RL-only design
- `plans/2026-02-18_hybrid-ml-redesign.md` — hybrid supervised + PPO architecture
- `plans/2026-02-18_feature-engineering-and-data-expansion.md` — nflverse integration, 72-dim state, feature store
- `plans/2026-02-18_draft-simulation-and-opponent-modeling.md` — simulate_draft.py, per-manager opponent modeling, self-play research synthesis
