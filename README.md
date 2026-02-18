# ffai — Fantasy Football Auction Draft AI

Hybrid ML system for ESPN fantasy football auction drafts. Uses supervised learning to estimate player value and PPO reinforcement learning to optimize bidding strategy. Can run as a real-time advisory tool (shows recommendations while you control the draft) or as an autonomous agent (controls the ESPN draft room via browser automation).

## Architecture

```
ESPN Historical Data (2009-2024)
         |
         v
[PlayerValueModel]   — supervised, two-head output:
  - expected_season_points
  - fair_auction_dollar  (PAR/VORP-scaled)
         |
         v
[PPO Agent]          — uses value model outputs as state features
  State:  budget, roster needs, player PAR, opponent budgets, market scarcity
  Action: bid amount (continuous Gaussian policy)
  Reward: value_efficiency (mid-draft) + season standing (terminal)
         |
         v
[Interfaces]
  Advisory Mode:    poll ESPN API, display recommendations in terminal
  Autonomous Mode:  Playwright browser automation of ESPN draft room
```

## Directory structure

```
ffai/src/ffai/
├── config/
│   ├── league.yaml          # ESPN credentials + league_id (gitignored — copy from league.yaml.example)
│   └── training.yaml        # hyperparameters
├── data/
│   ├── espn_scraper.py      # ESPN API client (credentials loaded from config)
│   └── preprocessor.py      # historical data processing, PAR/VORP calculation
├── value_model/
│   ├── player_value_model.py  # two-head supervised model (points + dollar value)
│   └── value_trainer.py       # training loop with early stopping
├── rl/
│   ├── ppo_agent.py         # Actor-Critic PPO with Gaussian continuous action
│   ├── state_builder.py     # centralized 56-dim state construction
│   ├── reward.py            # normalized reward shaping
│   └── replay_buffer.py     # GAE rollout buffer
├── simulation/
│   ├── auction_draft_simulator.py  # 12-team auction draft engine
│   └── season_simulator.py         # 17-week season simulator
└── interfaces/
    ├── live_draft_reader.py  # polls ESPN REST API during live drafts
    ├── advisory.py           # terminal UI showing recommendations
    └── autonomous.py         # Playwright browser control
```

## Setup

```bash
cd ffai
pip install -e .
pip install -r requirements.txt
playwright install chromium  # only needed for autonomous mode
```

**Configure credentials:**

```bash
cp ffai/src/ffai/config/league.yaml.example ffai/src/ffai/config/league.yaml
# Edit league.yaml with your ESPN league_id, SWID, and espn_s2 cookies.
# Find SWID and espn_s2 in browser DevTools > Application > Cookies
# after logging into ESPN Fantasy Football.
```

## Training pipeline

### Stage 1 — Collect historical data

```bash
python ffai/scripts/collect_data.py --years 2009-2024
```

Downloads draft results, player stats, and weekly scoring from ESPN for each year. Data is cached locally under `ffai/src/ffai/data/raw/`.

### Stage 2 — Train the value model

```bash
python ffai/scripts/train_value_model.py \
  --train-years 2009-2022 \
  --val-year 2023 \
  --test-year 2024
```

Trains a two-head neural network on historical data:
- **Points head**: predicts a player's total season fantasy points
- **Value head**: predicts fair auction dollar value (derived from PAR/VORP)

Target metrics: points RMSE ~5-6 pts, dollar MAE ~$5-8.
Best checkpoint saved to `checkpoints/value_model/best_model.pt`.

### Stage 3 — Train the PPO agent (three-phase curriculum)

**Phase 1 — heuristic baseline (500 episodes)**

No PPO learning. Uses value model output as a heuristic (`bid = fair_value × 0.95`). Validates the pipeline and confirms the agent finishes top-6 consistently before adding RL.

```bash
python ffai/scripts/train_rl.py --phase 1 --episodes 500
```

**Phase 2 — PPO training (2000 episodes)**

PPO learns to optimize bidding strategy. Average standing should improve from ~6th to ~3-4th by episode 1500.

```bash
python ffai/scripts/train_rl.py --phase 2 --episodes 2000 \
  --ppo-lr 3e-4 --gamma 0.99 --gae-lambda 0.95
```

**Phase 3 — self-play (3000 episodes)**

Some opponent teams are replaced with frozen PPO checkpoints from phase 2. Prevents overfitting to the fixed ESPN auto-draft opponents.

```bash
python ffai/scripts/train_rl.py --phase 3 --episodes 3000
```

## Usage

### Advisory mode

The agent monitors the live draft and prints bidding recommendations. You control the draft; the agent advises.

```bash
python ffai/scripts/advisory_draft.py \
  --team-id 1 \
  --value-model checkpoints/value_model/best_model.pt \
  --ppo-checkpoint checkpoints/ppo/ppo_checkpoint_2000.pt
```

### Autonomous mode

The agent controls the ESPN draft room directly via Playwright. Always test in a mock league first.

```bash
python ffai/scripts/autonomous_draft.py \
  --team-id 1 \
  --ppo-checkpoint checkpoints/ppo/ppo_checkpoint_2000.pt
```

The browser window stays visible (`headless=False`) so you can monitor and intervene. A dead-man's switch falls back to advisory mode if ESPN's UI selectors fail 10 times consecutively.

## Plans

- `plans/2025-02-18_initial-design.md` — original RL-only design
- `plans/2026-02-18_hybrid-ml-redesign.md` — current hybrid architecture (why RL wasn't converging + full redesign)
