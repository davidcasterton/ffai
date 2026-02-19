# Draft Simulation & Faithful Opponent Modeling

**Date**: 2026-02-18

## Summary

Implemented three improvements:

1. `scripts/simulate_draft.py` — new script to run post-training draft simulations; injects a trained RL model into one team slot and prints a play-by-play transcript plus per-team final rosters.
2. Per-manager opponent modeling — replaced league-average heuristic in `_opponent_max_bid()` with per-team lookup via `_team_manager_map` (built from `draft_df` in both `__init__` and `_make_simulator()`).
3. `transcript_callback` hook added to `simulate_draft()` — fires after each player is won; used by the simulation script.

## Files Changed

| File | Change |
|------|--------|
| `ffai/scripts/simulate_draft.py` | **New** — simulation entry point with transcript + roster output |
| `ffai/src/ffai/simulation/auction_draft_simulator.py` | Added `_team_manager_map` in `__init__`; `transcript_callback` param to `simulate_draft()`; upgraded `_opponent_max_bid()` to per-manager lookup |
| `ffai/src/ffai/rl/puffer_env.py` | Set `sim._team_manager_map` in `_make_simulator()` |
| `README.md` | Added simulate_draft.py to Usage section and directory structure |

## Self-Play Research Assessment

| Paper | Verdict |
|-------|---------|
| Self-Play survey (2408.01072) — PSRO | Useful: diverse population of opponents prevents overfitting. Phase 3 could sample from {heuristic, per-manager BC clones, past checkpoints}. |
| HuggingFace Unit 7 — classic self-play | Not recommended: homogenizes all agents, destroying manager-specific behavioral diversity. |
| SeRL (2505.20347) — LLM self-play | Not applicable: designed for deterministic QA tasks with majority-vote rewards. |
| SPACeR (2510.18060) — imitation-anchored self-play | Most applicable long-term: BC opponents fine-tuned with RL + KL penalty anchoring to historical behavior. |

## Future Work (Phase B)

Behavioral cloning per manager — small regression model trained on ~200 historical bids per manager predicting `bid_amount` given `(player_features, roster_state, budget)`; SPACeR-style KL anchoring if opponents are then fine-tuned with RL.
