# Self-Play for Auction Draft RL

**Date**: 2026-02-19
**Plan file**: `plans/2026-02-19_self-play-auction-draft.md`

---

## Context

The current system trains one RL agent against 11 heuristic opponents whose bids are derived from historical `manager_tendencies.csv` profiles. The sim output reveals a clear failure mode: most heuristic teams spend well below their $200 budget (e.g., Team 6: $67, Team 10: $94, Team 9: $99), causing players to be won at unrealistically low prices. This happens because:

1. `bid_per_proj_pt × proj_pts` produces conservative max-bids relative to actual market prices
2. Teams pile up $1 nominations they win uncontested rather than bidding aggressively
3. Opponents don't adapt when the RL agent bids cheaply — they never respond to market conditions

The root cause of issue #2 is that `should_bid()` caps bidding at `player.get("auction_value", 1)`, and `auction_value` in the ESPN pre-draft data is often 0 (ESPN doesn't provide values for most bench players), causing teams to drop out of almost every auction immediately.

**Self-play solves this**: Learned policies naturally learn to deploy their full $200 budget because they're optimizing for season standing, and leaving money on the table means losing value. As the RL policy improves, opponents co-evolve, creating a progressively harder and more realistic training environment — matching what the SPACeR paper achieves via shared policy + behavioral anchoring.

---

## Approach: Two-Stage Self-Play

### Stage 1 — PSRO-lite Checkpoint Pool (lower complexity, immediate value)

Replace heuristic opponents with a mix of: past RL policy checkpoints (sampled from a pool) + heuristic baseline. This is minimal infrastructure change and immediately improves opponent quality.

### Stage 2 — SPACeR-style BC Anchoring (higher fidelity, human-realistic)

Train a behavioral cloning (BC) reference model on historical ESPN draft data. Use it as:
1. A strong opponent initialization (better than heuristics)
2. An auxiliary reward signal (`α × log P(action|state)`) that keeps the policy human-realistic and prevents strategy collapse

---

## Implementation Steps

### Step 1: Perspective-Aware State Builder [PREREQUISITE]

**File**: `ffai/src/ffai/rl/state_builder.py`

`build_state()` currently hardcodes the RL agent's team as the subject. Add `team_name: str` parameter so any team's perspective can be built. Required for checkpoint policies to observe the draft from their own point of view.

Add `get_state_for(team_name)` to `AuctionDraftSimulator` — mirrors `get_state()` but from any team's POV. `build_state()` adds a `team_name` keyword parameter for documentation clarity; callers pass the perspective-correct state dict from `get_state_for(team_name)`.

---

### Step 2: Checkpoint Opponent Pool

**New file**: `ffai/src/ffai/rl/opponent_pool.py`

```python
class OpponentPool:
    def __init__(self, heuristic_fraction: float = 0.3, max_pool_size: int = 10)
    def add_checkpoint(self, path: Path) -> None
    def sample_policy(self) -> Optional[LoadedCheckpointPolicy]
        # Returns None (→ use heuristic) with probability heuristic_fraction
        # Returns a loaded policy otherwise, uniformly sampled from pool
```

`LoadedCheckpointPolicy` wraps a loaded `AuctionDraftPolicy` checkpoint. It calls `build_state(sim.get_state_for(team_name), current_player=player, ...)` to get its observation and outputs `max_bid = policy(state) * remaining_budget`.

Keep 30% heuristic fraction to maintain diversity and training stability — never fully remove heuristic opponents.

---

### Step 3: Simulator Support for Learned Opponents

**File**: `ffai/src/ffai/simulation/auction_draft_simulator.py`

Add `opponent_policies: Optional[Dict[str, LoadedCheckpointPolicy]]` constructor parameter (defaults to `None`).

In `_opponent_max_bid(team_name, player, current_bid)`:
```python
if self._opponent_policies and team_name in self._opponent_policies:
    policy = self._opponent_policies[team_name]
    return policy.get_bid(self, team_name, player, current_bid)
# else: existing heuristic logic unchanged
```

This is a clean extension with zero change to existing behavior when `opponent_policies=None`.

---

### Step 4: Environment Integration

**File**: `ffai/src/ffai/rl/puffer_env.py`

At `_start_episode()`:
- Sample 11 opponent policies from `OpponentPool` (one per opponent team)
- Build `opponent_policies` dict and pass to `_make_simulator()`

Add checkpoint registration: after each training phase, or every 50K steps, call `opponent_pool.add_checkpoint(current_checkpoint_path)`.

`OpponentPool` instance lives on the environment or is passed in at construction time.

---

### Step 5: Behavioral Cloning Reference Model [Stage 2]

**New files**:
- `ffai/src/ffai/rl/bc_reference.py` — BC model class (same MLP architecture as policy)
- `ffai/scripts/train_bc_reference.py` — training script

**Data source**: Historical ESPN draft CSVs in `data/{league_name}/`. For each historical pick: reconstruct `(state_72_dim, bid_fraction = bid_amount / budget_at_time)` pairs using the perspective-aware state builder (Step 1 prerequisite).

**Architecture**: Reuse `AuctionDraftPolicy` encoder (72 → 256 → 128 → 1, sigmoid). Trained with MSE loss on bid fractions.

**Uses**:
1. Cold-start the checkpoint pool: Phase 4 can begin with BC-trained opponents rather than phase-3-trained checkpoints, giving a stronger and more human-realistic opponent baseline from the start.
2. Auxiliary reward in `reward.py`: `r_bc = α × log P(action | state, θ_ref)`, where α ≈ 0.05. Added to `mid_draft_reward()` to keep the learned policy from drifting into degenerate bidding strategies.

---

### Step 6: Updated Training Curriculum

**File**: `ffai/scripts/train_puffer.py`

Add Phase 4 after existing phases:

| Phase | Steps | Season Sim | Opponents | Num Envs | Notes |
|-------|-------|------------|-----------|----------|-------|
| 1 | 100K | No | Heuristic | 1 | Unchanged |
| 2 | 500K | 1/10 eps | Heuristic | 4 | Unchanged |
| 3 | 1M | Every ep | Heuristic | 6 | Unchanged |
| 4 | 500K | Every ep | Pool (70% ckpt / 30% heuristic) | 6 | Self-play |

Phase 4 loads from `phase3_final.pt`, then runs against the checkpoint pool. Pool is seeded with the BC reference model (if Stage 2 is implemented) and enriched with snapshots every 50K steps.

`puffer.ini` may need a new `[self_play]` section for `heuristic_fraction`, `checkpoint_interval`, `bc_alpha`.

---

### Step 7: Budget Utilization Fix (Parallel Quick Fix)

**Root cause**: `should_bid()` caps bids at `player.get("auction_value", 1)`. The ESPN pre-draft API returns `auction_value = 0` for most players (only provides values for top prospects), so `min(budget, 0) = 0` and teams immediately drop out of virtually every auction, leaving budgets unspent.

**Fix**:
1. Remove the `auction_value` cap from `should_bid()` — only check budget safety (`can_afford`)
2. In the bidding loop, remove a team from `active_bidders` when `max_bid <= current_bid` (i.e., they've hit their real ceiling from `_opponent_max_bid()`)

---

### Step 8: End-to-End Training Launcher

**New file**: `ffai/scripts/train.py`

A single entry point that chains all phases in sequence:

```bash
.venv/bin/python ffai/scripts/train.py            # run all phases
.venv/bin/python ffai/scripts/train.py --from-phase 3  # resume from phase 3
.venv/bin/python ffai/scripts/train.py --skip-bc       # skip BC reference training
```

Execution order:
1. Phase 1 (100K steps) — draft warm-up
2. Phase 2 (500K steps) — season sim introduced
3. Phase 3 (1M steps) — full season feedback
4. Train BC reference model (optional, `--skip-bc` to omit)
5. Phase 4 (500K steps) — self-play with checkpoint pool

---

## Critical Files

| File | Change Type |
|------|------------|
| `ffai/src/ffai/rl/state_builder.py` | Modify — add `team_name` param |
| `ffai/src/ffai/simulation/auction_draft_simulator.py` | Modify — `get_state_for()`, `opponent_policies`, fix `should_bid()`, fix bidding loop |
| `ffai/src/ffai/rl/puffer_env.py` | Modify — pool sampling at episode reset |
| `ffai/src/ffai/rl/opponent_pool.py` | **New** — checkpoint pool class |
| `ffai/src/ffai/rl/bc_reference.py` | **New** — BC reference model (Stage 2) |
| `ffai/scripts/train_bc_reference.py` | **New** — BC training script (Stage 2) |
| `ffai/scripts/train_puffer.py` | Modify — add Phase 4 |
| `ffai/scripts/train.py` | **New** — end-to-end launcher (all phases) |
| `ffai/src/ffai/config/puffer.ini` | Modify — `[self_play]` config section |
| `MODEL.md` | Update — document self-play architecture |
| `README.md` | Update — new training phase, new scripts |

---

## Why This Addresses the Under-Spending Problem

A learned policy trained to maximize season standing will naturally learn to deploy its full $200 budget, because:
- Every unspent dollar represents forgone player value
- PPO will penalize strategies that leave budget on the table (→ worse rosters → worse standings)
- As checkpoint policies improve, they bid more competitively, raising market prices closer to real auction dynamics

The immediate Step 7 fix also helps: removing the ESPN `auction_value` cap allows heuristic opponents to stay in bidding wars longer and actually spend their budgets.

---

## Verification

1. **State builder**: Unit test `build_state(sim.get_state_for(team_name), ...)` for 3 teams and verify budget/roster dims reflect the correct team's perspective.
2. **Opponent pool**: Verify `sample_policy()` returns `None` with ~30% frequency; verify loaded policies produce valid `max_bid` values (> 0, ≤ remaining budget).
3. **Simulator integration**: Run `simulate_draft.py` with all 11 opponents as checkpoint policies; verify all teams spend closer to their full $200 budgets.
4. **Training integration**: Run Phase 4 for 10K steps (smoke test); verify loss curves are stable and `episode_reward` improves over Phase 3 baseline.
5. **BC reference** (Stage 2): Verify BC model predicts bid fractions in [0, 1] and correlates with historical bid data; check auxiliary reward weight doesn't dominate PPO objective.
