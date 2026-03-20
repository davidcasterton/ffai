# Plan: Data Quality, Bench Value, and Bidding Improvements

**Date**: 2026-02-19
**Scope**: Three interrelated quality improvements to the ffai data pipeline, season simulation, and reward shaping.

---

## Context

1. **RL overspending / heuristic under-spending**: Simulation showed the RL agent bid $125 for a player worth ~$45 VORP_dollar, while heuristic teams (Team 6, Team 10) spent only $136–158 of $200 budgets.

   Root cause: `manager_tendencies.py` computes `bid_per_proj_pt` as `mean(bid_amount / proj_pts)` with no budget normalization. If favrefignewton used a $1000 auction budget in some seasons, those `bid_per_proj_pt` values would be 5× inflated, corrupting every per-manager profile used by heuristic opponents. The `high_bid_rate` threshold is hardcoded at `> $30` (15% of $200, but only 3% of $1000). The `budget` field in `league.yaml` is **never read by the data pipeline**.

2. **Bench player value**: `season_simulator.py` fills lineups optimally each week by projected points but never models bye weeks or injuries. Without bye weeks, bench players are never forced into starting slots. The user confirmed they want: **bye weeks + stochastic injuries**.

3. **Data sanity**: Multiple hardcoded $200 assumptions (`TOTAL_BUDGET = 200 * 12`, `DOLLAR_MAX = 80.0`, `> $30` threshold in manager_tendencies). No validation script exists to detect which years used non-standard budgets.

---

## Step 1: Sanity Check Script (diagnostic — run first, no model changes)

**New file**: `scripts/sanity_check_data.py`

Reads all available `draft_results_{league_id}_{year}.csv` files and produces:
- Per-year table: total spend per team, median team spend, inferred budget
- Per-year `bid_amount` distribution (p5, median, p95, max), fraction of $1 bids
- Per-manager: `bid_per_proj_pt` range across years (flag if any year's value > 3× median)
- Summary: which years have non-$200 budgets

**Budget detection logic**:
```python
team_spend = draft_df.groupby("manager_id")["bid_amount"].sum()
detected_budget = 1000 if team_spend.median() > 300 else 200
```

Writes `data/{league_name}_processed/budget_by_year.csv` (columns: `year`, `detected_budget`, `median_team_spend`) — consumed by Step 2.

---

## Step 2: Budget Normalization (foundational data fix)

**Philosophy**: normalize `bid_amount` to $200-equivalent at the earliest stage — in `build_features.py` before any feature engineering runs. Constants in `state_builder.py`, `reward.py`, and `preprocessor.py` all stay correct for $200 budgets and require no changes.

### `scripts/build_features.py` (modify)

Before calling any feature engineering module for a given year:
1. Load `budget_by_year.csv` (from Step 1); fallback to inline detection if missing
2. Compute `norm_factor = 200.0 / detected_budget` per year
3. Apply: `draft_df["bid_amount"] = (draft_df["bid_amount"] * norm_factor).round(2)`

This is the **only** place normalization needs to be applied.

### `data/feature_engineering/manager_tendencies.py` (modify)

One change only — `dollar_one_rate` threshold:
- Currently: `(grp["bid_amount"] == 1)` — correct for $200, but a $1 bid in a $1000 league normalizes to $0.20
- Change to: `(grp["bid_amount"] < 1.5)` to catch near-minimum bids in normalized data
- `high_bid_rate > 30` stays correct (30 = 15% of $200, same threshold after normalization)
- `bid_per_proj_pt` is automatically correct after normalization

### `data/feature_engineering/position_strategy.py`

No code changes needed — `avg_bid`, `median_bid`, `top5_avg_bid` computed from normalized `bid_amount` will be $200-calibrated automatically.

### `data/preprocessor.py`, `rl/state_builder.py`, `rl/reward.py`

No changes needed. All constants assume $200-equivalent data:
- `TOTAL_BUDGET = 200 * 12` ✓
- `DOLLAR_MAX = 80.0` ✓ (40% of $200 budget)
- `budget_max = 200.0` in reward ✓

**Expected effect**: After normalization and rebuilding features:
- `bid_per_proj_pt_rb` for a typical manager should be ~$0.20–0.45/pt
- Heuristic `_opponent_max_bid()` should produce $20–70 bids for RBs instead of ~$200
- VORP_dollar calibration improves → better RL reward signal → less overspending

---

## Step 3: Bye Weeks + Injury Simulation

**File**: `ffai/src/ffai/simulation/season_simulator.py`

### Bye Weeks

Add `_build_bye_week_map()` method called in `__init__`:

```python
def _build_bye_week_map(self) -> dict[str, int]:
    """Return {player_id: bye_week} for all players this season.

    A player's bye week is the week number in [1, 17] with no weekly_df entry,
    inferring that their team had no game that week.
    """
    all_weeks = set(range(1, 18))
    bye_map = {}
    for pid, grp in self.weekly_df.groupby("player_id"):
        played_weeks = set(grp["week"].astype(int).tolist())
        missing = all_weeks - played_weeks
        if len(missing) == 1:
            bye_map[str(pid)] = min(missing)
    return bye_map
```

Modify `optimize_weekly_roster(team_name, week)`:
- Before sorting `available_players`, exclude any player whose `player_id` is in `_bye_week_map` with value `== week`
- Bye players cannot start or sit — truly unavailable for the whole week

### Injury Simulation

Add `_simulate_injuries(players, week, team_name)` method:

```python
def _simulate_injuries(self, players: list, week: int, team_name: str) -> list:
    """Stochastically remove injured players (p=0.03 per player per week).

    Uses a deterministic seed per (team_name, week, year) for reproducibility.
    ~0.03 × 17 weeks × ~15 players ≈ 8-10 injury-missed starts per team per season.
    """
    seed = hash((team_name, week, self.year)) % (2**32)
    rng = np.random.default_rng(seed=seed)
    return [p for p in players if rng.random() > 0.03]
```

Modify `optimize_weekly_roster`:
1. Get `available_players` from roster (non-None)
2. Update weekly projected_points from `weekly_df` (existing logic)
3. Remove bye-week players: `[p for p if bye_map.get(str(p["player_id"])) != week]`
4. Apply injuries: `_simulate_injuries(available_players, week, team_name)`
5. Sort by projected_points and fill roster slots (existing logic)

**Effect**: With bye weeks + injuries, bench depth directly determines matchup outcomes. A 14-player roster sees ~1–2 bye-related forced bench starts plus ~8–10 injury-missed starts per season. Teams with no RB depth lose weeks when both starters are on bye the same week or are injured. The RL agent learns to value depth through the terminal standings reward implicitly — no state vector changes required.

---

## Step 4: Fix Auction Mechanics — Commit Reservation Prices

**File**: `ffai/src/ffai/simulation/auction_draft_simulator.py`

### The bug

In both `simulate_draft()` (line 260) and `draft_steps()` (line 355), `_opponent_max_bid()` is called **fresh every iteration** of the inner `while len(active_bidders) > 1` loop. Because `_opponent_max_bid()` adds Gaussian noise (`np.random.normal(0, bpppt * 0.15)`), each call can produce a different value. A team that computed max_bid=$47 in round 1 could compute max_bid=$43 in round 2 and drop out — even though the current_bid may only be $44. This is not how real auctions work.

In a real ascending auction, each participant has a **reservation price** they commit to at the start. They bid +$1 each time the price is below their reservation, and drop out exactly when price ≥ reservation. The winner pays second-highest reservation price + $1.

### The fix

Before the `while len(active_bidders) > 1` loop (in both `simulate_draft` and `draft_steps`), compute and cache each **heuristic** opponent's reservation price once:

```python
# Cache heuristic opponents' reservation prices at start of each nomination.
# This makes opponents commit to a price rather than re-randomizing each round.
_cached_max_bids: dict[str, float] = {}
for t in self.teams:
    if t == self.rl_team_name:
        continue
    # Only cache if this team uses the heuristic (not a checkpoint policy)
    if not (self._opponent_policies and self._opponent_policies.get(t) is not None):
        _cached_max_bids[t] = self._opponent_max_bid(t, nominated_player, 0)
```

Inside the loop, change heuristic opponent bid lookup from:
```python
max_bid = self._opponent_max_bid(team_name, nominated_player, current_bid)
```
to:
```python
max_bid = _cached_max_bids.get(team_name) or self._opponent_max_bid(team_name, nominated_player, current_bid)
```

**Checkpoint-policy opponents** (Phase 4) are re-polled each round because their policy is state-dependent (current_bid is in their observation). The RL agent is always re-polled (dynamic decision). Only heuristic opponents are cached.

### Effect

- Opponents commit to a consistent reservation price per player → winning bid correctly = second-highest reservation + $1
- Removes spurious early dropouts caused by noise fluctuation → heuristic teams bid up to their true limit
- Prices converge faster (fewer loop iterations needed)
- More realistic training signal for the RL agent (opponents don't randomly waffle)

---

## Step 5: Reward Refinement (secondary RL overspending fix)

**File**: `ffai/src/ffai/rl/reward.py`

Add a nonlinear overbid penalty to `mid_draft_reward()`:

```python
# Extra penalty for extreme overbidding (bid > 2× fair value)
overbid_ratio = bid_amount / max(fair_dollar_value, 1.0)
if overbid_ratio > 2.0:
    base -= 0.1 * min(overbid_ratio - 2.0, 3.0)  # max additional penalty: -0.3
```

Example: $125 bid on a $45 VORP_dollar player → ratio = 2.78, extra penalty = −0.078.

**Note**: This is a secondary guard. The primary fix for RL overbidding is Step 2 (better VORP_dollar calibration from normalized training data). Add this only after confirming that budget normalization alone doesn't fully resolve overspending.

---

## Step 6: Documentation

- `README.md`: add `sanity_check_data.py` to directory tree, document budget normalization in the data pipeline section, add bye weeks and injury simulation to the season sim description
- Save full plan to `plans/2026-02-19_data-quality-bench-value-bidding.md`

---

## Verification Checklist

1. **Sanity check**: Run `scripts/sanity_check_data.py`. Report clearly shows which years used non-$200 budgets. `budget_by_year.csv` is written.

2. **Normalized features**: After `build_features.py`, inspect `manager_tendencies_{id}.csv`:
   - `bid_per_proj_pt_rb` should be in ~[0.15, 0.50]
   - `bid_per_proj_pt_qb` should be higher (QBs cost more per point at QB scarcity)
   - If values are still > 1.0, budget detection needs adjustment

3. **Heuristic spending**: Run `simulate_draft.py` after rebuilding features. Heuristic teams should spend $175–200 (up from $136–158).

4. **Bye week coverage**: Verify `_build_bye_week_map()` produces ~30 entries (roughly 14 starters × ~2 players sharing bye weeks per team). Check that in the season sim, bench players start in bye weeks for each team.

5. **Auction commitment**: Enable DEBUG logging and run `simulate_draft.py`. For a single player's auction, verify the same team does not appear to raise and then drop below their prior bid. Inspect that winning prices match "second-highest bid + $1" pattern.

6. **Injury rate**: Run 5 full season simulations. Verify mean injury-missed starts is ~8–10 per team per season. (Too high → reduce p from 0.03; too low → increase.)

7. **Smoke test retraining**: Run `scripts/train.py --smoke-test` to confirm no import errors or shape mismatches.

---

## Critical Files

| File | Change | Purpose |
|------|--------|---------|
| `scripts/sanity_check_data.py` | **New** | Detect budget per year, report anomalies, write budget_by_year.csv |
| `scripts/build_features.py` | Modify | Load budget_by_year.csv, apply bid_amount normalization per year |
| `data/feature_engineering/manager_tendencies.py` | Modify | Change dollar_one_rate threshold from `== 1` to `< 1.5` |
| `simulation/auction_draft_simulator.py` | Modify | Cache heuristic reservation prices before bidding loop (both `simulate_draft` and `draft_steps`) |
| `simulation/season_simulator.py` | Modify | Add `_build_bye_week_map()`, `_simulate_injuries()`, update `optimize_weekly_roster()` |
| `rl/reward.py` | Modify | Add overbid protection penalty (secondary fix) |
| `README.md` | Update | Document new script, bye weeks, injury model |
| `plans/2026-02-19_data-quality-bench-value-bidding.md` | **New** | Permanent plan record |

---

## Follow-on (not in this plan)

**Bench depth in state vector**: Expanding STATE_DIM 72→78 to include "best bench player projected points per position" would let the RL agent explicitly reason about depth during the draft. This requires retraining from scratch. Recommend only after validating that the above fixes produce realistic simulation behavior.
