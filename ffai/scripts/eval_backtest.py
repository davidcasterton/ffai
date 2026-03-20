#!/usr/bin/env python3
"""
Backtest a policy checkpoint across years/seeds using draft + season simulation.

Usage:
  .venv/bin/python ffai/scripts/eval_backtest.py --checkpoint checkpoints/puffer/phase4_final.pt --years 2022-2024
  .venv/bin/python ffai/scripts/eval_backtest.py --years 2024 --seeds 5
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np
import torch

_SCRIPT_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _SCRIPT_DIR.parent.parent
sys.path.insert(0, str(_REPO_ROOT / "ffai" / "src"))


def _parse_years(s: str) -> list[int]:
    if "-" in s and "," not in s:
        start, end = s.split("-")
        return list(range(int(start), int(end) + 1))
    return [int(x.strip()) for x in s.split(",") if x.strip()]


def _seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _load_policy(checkpoint_path: Path):
    from ffai.rl.puffer_policy import AuctionDraftPolicy
    from ffai.rl.state_builder import STATE_DIM
    import gymnasium

    class _FakeEnv:
        single_observation_space = gymnasium.spaces.Box(
            low=-np.inf, high=np.inf, shape=(STATE_DIM,), dtype=np.float32
        )
        single_action_space = gymnasium.spaces.Box(
            low=0.0, high=1.0, shape=(1,), dtype=np.float32
        )

    policy = AuctionDraftPolicy(_FakeEnv())
    state_dict = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if isinstance(state_dict, dict) and "state_dict" in state_dict:
        state_dict = state_dict["state_dict"]
    elif isinstance(state_dict, dict) and "model_state_dict" in state_dict:
        state_dict = state_dict["model_state_dict"]
    policy.load_state_dict(state_dict, strict=False)
    policy.eval()
    return policy


@dataclass
class EpisodeResult:
    year: int
    seed: int
    standing: int
    wins: int
    budget_utilization: float
    roster_complete: int


def _run_episode(year: int, seed: int, checkpoint: Path | None, rl_team: str) -> EpisodeResult:
    from ffai.simulation.auction_draft_simulator import AuctionDraftSimulator
    from ffai.simulation.season_simulator import SeasonSimulator
    from ffai.rl.state_builder import build_state

    _seed_all(seed)
    sim = AuctionDraftSimulator(year=year, rl_team=rl_team)

    if checkpoint is not None:
        policy = _load_policy(checkpoint)
        feature_store = sim.feature_store
        original_fn = sim._opponent_max_bid

        def _patched(team_name: str, player: dict, current_bid: float = 0.0) -> float:
            if team_name != rl_team:
                return original_fn(team_name, player, current_bid)
            state = sim.get_state()
            budget = float(sim.teams[rl_team]["current_budget"])
            obs = build_state(
                state,
                current_player=player,
                current_bid=float(current_bid),
                feature_store=feature_store,
                year=year,
            ).unsqueeze(0)
            with torch.no_grad():
                dist, _ = policy.forward_eval(obs)
                bid_fraction = float(dist.mean.item())
            return max(1.0, bid_fraction * budget)

        sim._opponent_max_bid = _patched  # type: ignore[method-assign]

    completed, teams, _ = sim.simulate_draft()
    draft_results = sim.get_draft_results()

    season = SeasonSimulator(draft_results=draft_results["teams"], year=year)
    season.simulate_season()
    standings = season.get_standings()
    standing_idx = next(i for i, (team, _) in enumerate(standings) if team == rl_team)
    wins = int(season.standings[rl_team])

    spent = 200.0 - float(teams[rl_team]["current_budget"])
    utilization = max(0.0, min(1.0, spent / 200.0))

    return EpisodeResult(
        year=year,
        seed=seed,
        standing=standing_idx + 1,
        wins=wins,
        budget_utilization=utilization,
        roster_complete=int(completed),
    )


def evaluate(checkpoint: Path | None, years: list[int], num_seeds: int, rl_team: str) -> dict:
    episodes: list[EpisodeResult] = []
    for year in years:
        for i in range(num_seeds):
            seed = 1000 + (year * 10) + i
            episodes.append(_run_episode(year=year, seed=seed, checkpoint=checkpoint, rl_team=rl_team))

    standings = np.array([e.standing for e in episodes], dtype=np.float32)
    wins = np.array([e.wins for e in episodes], dtype=np.float32)
    utilization = np.array([e.budget_utilization for e in episodes], dtype=np.float32)
    complete = np.array([e.roster_complete for e in episodes], dtype=np.float32)

    metrics = {
        "episodes": len(episodes),
        "avg_standing": float(np.mean(standings)),
        "top4_rate": float(np.mean(standings <= 4)),
        "win_rate_estimate": float(np.mean(wins / 17.0)),
        "avg_wins": float(np.mean(wins)),
        "avg_budget_utilization": float(np.mean(utilization)),
        "roster_completion_rate": float(np.mean(complete)),
    }
    return {
        "checkpoint": str(checkpoint) if checkpoint else None,
        "years": years,
        "num_seeds": num_seeds,
        "rl_team": rl_team,
        "metrics": metrics,
        "episodes_detail": [asdict(e) for e in episodes],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Backtest RL checkpoint on draft+season simulation")
    parser.add_argument("--checkpoint", type=Path, default=None, help="Policy checkpoint .pt (omit for heuristic baseline)")
    parser.add_argument("--years", type=str, default="2024", help="Year range/list, e.g. 2022-2024 or 2022,2023")
    parser.add_argument("--seeds", type=int, default=3, help="Number of random seeds per year")
    parser.add_argument("--rl-team", type=str, default="Team 1", help="Team slot to evaluate")
    parser.add_argument("--output", type=Path, default=None, help="Optional output JSON path")
    args = parser.parse_args()

    years = _parse_years(args.years)
    report = evaluate(checkpoint=args.checkpoint, years=years, num_seeds=args.seeds, rl_team=args.rl_team)
    text = json.dumps(report, indent=2)
    print(text)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n")
        print(f"\nSaved report to {args.output}")


if __name__ == "__main__":
    main()
