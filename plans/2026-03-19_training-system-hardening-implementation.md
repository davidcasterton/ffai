# 2026-03-19 Training System Hardening — Implemented Changes

This document summarizes the implementation completed from the system design review follow-up.

## 1) Data + Value Model Training Integrity

- Wired engineered features into value-model training by loading and passing `FeatureStore` through train/val/test preprocessing.
  - File: `ffai/scripts/train_value_model.py`
- Added training run manifest output with core config, dataset counts, feature-store status, and test metrics.
  - File: `ffai/scripts/train_value_model.py`
- Fixed multi-year scaler behavior to fit on the full training corpus (not first processed year only), then re-transform all train-year feature matrices.
  - File: `ffai/src/ffai/data/preprocessor.py`
- Corrected validation metric naming in logs/history (`dollar_RMSE` rather than mislabeled MAE).
  - File: `ffai/src/ffai/value_model/value_trainer.py`

## 2) Reward + Simulator Consistency

- Unified mid-draft reward path so simulator uses `ffai.rl.reward.mid_draft_reward` directly, including overbid shaping.
  - File: `ffai/src/ffai/simulation/auction_draft_simulator.py`
- Improved opponent tendency state construction to use team-specific manager profiles for top-budget opponents (fallback to league average only when needed).
  - File: `ffai/src/ffai/simulation/auction_draft_simulator.py`

## 3) Multi-Year RL Training Sampling

- Added multi-year episode sampling support in env/trainer:
  - `AuctionDraftEnv` now accepts `years` and samples per-episode training year.
  - `train_puffer.py` now accepts `--train-years` (range or CSV) and passes it into env creation.
  - `train.py` now forwards `--train-years` through all curriculum phases.
  - Files: `ffai/src/ffai/rl/puffer_env.py`, `ffai/scripts/train_puffer.py`, `ffai/scripts/train.py`

## 4) BC Dataset Reconstruction Quality

- Replaced placeholder BC state generation with historical replay-style state reconstruction:
  - Tracks per-team budgets and roster slots across historical pick sequence.
  - Builds position needs/scarcity/market values from remaining player pool.
  - Computes draft progress from actual pick index.
  - Files: `ffai/scripts/train_bc_reference.py`

## 5) Evaluation Tooling

- Added backtest evaluator script for checkpoint/heuristic runs with year/seed sweeps and JSON reporting:
  - Metrics include average standing, win-rate estimate, top-4 rate, budget utilization, roster completion.
  - File: `ffai/scripts/eval_backtest.py`
- Added regression gate script to compare candidate vs baseline checkpoints against configurable thresholds:
  - File: `ffai/scripts/eval_regression.py`

## 6) Verification Performed

- Syntax/compile validation:
  - `python -m compileall ffai/scripts ffai/src/ffai` (passed)
- Smoke test:
  - `.venv/bin/python ffai/scripts/check_smoke.py` (passed)
- Evaluation script sanity run:
  - `.venv/bin/python ffai/scripts/eval_backtest.py --years 2024 --seeds 1` (passed)
