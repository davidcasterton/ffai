#!/usr/bin/env python3
"""
Fast smoke test for the ffai pipeline.

Verifies that all key modules import correctly, the environment initializes,
reset/step work with the correct shapes, and one full draft runs without error.
Completes in ~5-15 seconds. Use instead of --total-timesteps 500 for validation.

Usage:
    .venv/bin/python ffai/scripts/check_smoke.py
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

PASS = "[PASS]"
FAIL = "[FAIL]"


_SENTINEL = object()


def check(label: str, fn):
    """Run fn(), print pass/fail. Returns fn()'s result, or _SENTINEL on exception."""
    t0 = time.monotonic()
    try:
        result = fn()
        elapsed = time.monotonic() - t0
        print(f"  {PASS} {label} ({elapsed:.2f}s)")
        return result
    except Exception as exc:
        elapsed = time.monotonic() - t0
        print(f"  {FAIL} {label} ({elapsed:.2f}s): {exc}")
        return _SENTINEL


def main():
    failures = []

    print("\n=== ffai smoke test ===\n")

    # -----------------------------------------------------------------------
    # 1. Module imports
    # -----------------------------------------------------------------------
    print("[1/4] Imports")

    r = check("ffai.rl.reward", lambda: __import__("ffai.rl.reward", fromlist=["mid_draft_reward"]))
    if r is _SENTINEL: failures.append("import reward")

    r = check("ffai.rl.state_builder", lambda: __import__("ffai.rl.state_builder", fromlist=["build_state"]))
    if r is _SENTINEL: failures.append("import state_builder")

    r = check("ffai.simulation.auction_draft_simulator",
              lambda: __import__("ffai.simulation.auction_draft_simulator",
                                 fromlist=["AuctionDraftSimulator"]))
    if r is _SENTINEL: failures.append("import auction_draft_simulator")

    r = check("ffai.simulation.season_simulator",
              lambda: __import__("ffai.simulation.season_simulator", fromlist=["SeasonSimulator"]))
    if r is _SENTINEL: failures.append("import season_simulator")

    r = check("ffai.rl.puffer_env",
              lambda: __import__("ffai.rl.puffer_env", fromlist=["AuctionDraftEnv"]))
    if r is _SENTINEL: failures.append("import puffer_env")

    r = check("ffai.rl.puffer_policy",
              lambda: __import__("ffai.rl.puffer_policy", fromlist=["AuctionDraftPolicy"]))
    if r is _SENTINEL: failures.append("import puffer_policy")

    r = check("ffai.data.feature_store",
              lambda: __import__("ffai.data.feature_store", fromlist=["FeatureStore"]))
    if r is _SENTINEL: failures.append("import feature_store")

    # -----------------------------------------------------------------------
    # 2. Reward function shapes
    # -----------------------------------------------------------------------
    print("\n[2/4] Reward function")

    def _check_reward():
        from ffai.rl.reward import mid_draft_reward, terminal_reward
        r1 = mid_draft_reward(40.0, 40.0, True, True)
        r2 = mid_draft_reward(40.0, 120.0, True, True)   # overbid: ratio=3 → extra penalty
        r3 = terminal_reward(0, 14)
        assert isinstance(r1, float), f"expected float, got {type(r1)}"
        assert isinstance(r2, float)
        assert r2 < r1, f"overbid should have lower reward than fair bid: {r2} vs {r1}"
        assert r3 > 0, f"1st place + 14 wins should be positive: {r3}"
        # Overbid penalty: $120 on $40 player → ratio=3, extra = -0.1*(3-2)= -0.1
        expected_extra = -0.1 * min(3.0 - 2.0, 3.0)
        base_without_penalty = (40.0 - 120.0) / 200.0 + 0.1
        assert abs(r2 - (base_without_penalty + expected_extra)) < 1e-6, \
            f"overbid penalty mismatch: {r2} vs {base_without_penalty + expected_extra}"
        return True

    r = check("reward shapes + overbid penalty", _check_reward)
    if r is _SENTINEL: failures.append("reward shapes")

    # -----------------------------------------------------------------------
    # 3. Environment: init, reset, step
    # -----------------------------------------------------------------------
    print("\n[3/4] Environment (init → reset → 5 steps)")

    def _make_env():
        from ffai.rl.puffer_env import AuctionDraftEnv
        from ffai.rl.state_builder import STATE_DIM
        env = AuctionDraftEnv(year=2024, enable_season_sim=False, season_sim_interval=999)
        obs, _ = env.reset()
        # PufferLib may add a leading batch dim: (1, STATE_DIM) or (STATE_DIM,)
        assert obs.shape[-1] == STATE_DIM, f"obs last dim {obs.shape[-1]} != {STATE_DIM}"
        for _ in range(5):
            action = env.action_space.sample()
            result = env.step(action)
            step_obs = result[0]
            step_reward = result[1]
            assert step_obs.shape[-1] == STATE_DIM
            assert isinstance(float(step_reward), float)
        env.close()
        return True

    env = check("AuctionDraftEnv (phase 1, no season sim)", _make_env)
    if env is _SENTINEL: failures.append("env init/step")

    # -----------------------------------------------------------------------
    # 4. Full draft simulation (auction mechanics end-to-end)
    # -----------------------------------------------------------------------
    print("\n[4/4] Full draft simulation (no RL model)")

    def _simulate():
        from ffai.simulation.auction_draft_simulator import AuctionDraftSimulator
        sim = AuctionDraftSimulator(year=2024)
        completed, teams, _ = sim.simulate_draft()
        assert completed, "draft did not complete"
        for team_name, team in teams.items():
            spent = 200 - team["current_budget"]
            assert 0 < spent <= 200, f"{team_name} spent ${spent} (out of range)"
        return teams

    teams = check("simulate_draft() — all-heuristic 12-team draft", _simulate)
    if teams is _SENTINEL: failures.append("simulate_draft")
    elif isinstance(teams, dict):
        # Print a brief spending summary
        spends = {t: 200 - d["current_budget"] for t, d in teams.items()}
        print(f"         team spend range: ${min(spends.values())}–${max(spends.values())}")

    # -----------------------------------------------------------------------
    # Summary
    # -----------------------------------------------------------------------
    print(f"\n{'=' * 40}")
    if failures:
        print(f"SMOKE TEST FAILED ({len(failures)} error(s)):")
        for f in failures:
            print(f"  - {f}")
        sys.exit(1)
    else:
        print("SMOKE TEST PASSED — all checks OK")
    print(f"{'=' * 40}\n")


if __name__ == "__main__":
    main()
