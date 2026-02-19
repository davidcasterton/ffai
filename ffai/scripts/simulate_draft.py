"""
simulate_draft.py — Run a complete auction draft simulation for a given year.

Optionally injects a trained AuctionDraftPolicy checkpoint into one team slot.
Prints a play-by-play transcript and per-team final rosters.

Usage:
    # All-heuristic baseline (per-manager behavioral models, no RL)
    .venv/bin/python ffai/scripts/simulate_draft.py --year 2024

    # Inject RL model as Team 3
    .venv/bin/python ffai/scripts/simulate_draft.py \\
        --year 2024 \\
        --rl-model-path checkpoints/puffer/phase3_final.pt \\
        --rl-team "Team 3"

    # Save output to file
    .venv/bin/python ffai/scripts/simulate_draft.py --year 2024 \\
        --rl-model-path checkpoints/puffer/phase3_final.pt \\
        --output results/sim_2024.txt
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

# Resolve project root (repo_root/ffai/scripts/ → repo_root)
_SCRIPT_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _SCRIPT_DIR.parent.parent
sys.path.insert(0, str(_REPO_ROOT / "ffai" / "src"))


def _load_policy(checkpoint_path: str):
    """Load AuctionDraftPolicy from a .pt checkpoint. Returns the model in eval mode."""
    from ffai.rl.puffer_policy import AuctionDraftPolicy
    from ffai.rl.state_builder import STATE_DIM
    import gymnasium

    # Build a minimal env-like object just for AuctionDraftPolicy.__init__
    class _FakeEnv:
        single_observation_space = gymnasium.spaces.Box(
            low=-np.inf, high=np.inf, shape=(STATE_DIM,), dtype=np.float32
        )
        single_action_space = gymnasium.spaces.Box(
            low=0.0, high=1.0, shape=(1,), dtype=np.float32
        )

    policy = AuctionDraftPolicy(_FakeEnv())
    state_dict = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    # Handle checkpoint dicts that wrap state_dict under a key
    if isinstance(state_dict, dict) and "state_dict" in state_dict:
        state_dict = state_dict["state_dict"]
    elif isinstance(state_dict, dict) and "model_state_dict" in state_dict:
        state_dict = state_dict["model_state_dict"]
    policy.load_state_dict(state_dict, strict=False)
    policy.eval()
    return policy


class _RLBidder:
    """Thin wrapper: given (obs_tensor) → bid_fraction via AuctionDraftPolicy."""

    def __init__(self, policy, feature_store, year: int):
        self.policy = policy
        self.feature_store = feature_store
        self.year = year

    def get_max_bid(self, sim_state: dict, player: dict, current_bid: int, budget: float) -> float:
        """Return the RL team's maximum willing bid for this player."""
        from ffai.rl.state_builder import build_state

        obs = build_state(
            sim_state,
            current_player=player,
            current_bid=float(current_bid),
            feature_store=self.feature_store,
            year=self.year,
        ).unsqueeze(0)  # (1, STATE_DIM)

        with torch.no_grad():
            dist, _ = self.policy.forward_eval(obs)
            fraction = dist.mean.item()  # use mean for deterministic eval

        return fraction * budget


def _build_team_manager_display(sim) -> dict[str, str]:
    """Return {team_name: display_name} using draft_df manager names.

    Uses sim._team_manager_map (slot → manager_id) to look up each manager's
    display name, since draft_df["team_name"] contains ESPN team names, not
    the generic "Team N" slot names used internally by the simulator.
    """
    display = {}
    df = getattr(sim, "draft_df", None)
    team_manager_map = getattr(sim, "_team_manager_map", {})
    if df is None or "manager_id" not in df.columns:
        return display

    for team_name in sim.teams:
        manager_id = team_manager_map.get(team_name)
        if not manager_id:
            display[team_name] = ""
            continue
        rows = df[df["manager_id"] == manager_id]
        if rows.empty:
            display[team_name] = ""
            continue
        row = rows.iloc[0]
        if "manager_display_name" in df.columns and row.get("manager_display_name"):
            display[team_name] = str(row["manager_display_name"])
        elif "manager_first_name" in df.columns:
            first = str(row.get("manager_first_name", ""))
            last = str(row.get("manager_last_name", ""))
            display[team_name] = f"{first} {last}".strip()
        else:
            display[team_name] = ""
    return display


def _run_simulation(
    year: int,
    rl_model_path: str | None,
    rl_team: str,
    output_path: str | None,
) -> None:
    from ffai.simulation.auction_draft_simulator import AuctionDraftSimulator

    # --- Load RL policy if requested ---
    rl_bidder = None
    if rl_model_path:
        print(f"Loading RL model from: {rl_model_path}")
        policy = _load_policy(rl_model_path)
        # Feature store will be set after simulator is created
        rl_bidder_factory = policy  # stored temporarily
    else:
        rl_bidder_factory = None

    # --- Create simulator ---
    sim = AuctionDraftSimulator(year=year, rl_team=rl_team)

    if rl_bidder_factory is not None:
        rl_bidder = _RLBidder(rl_bidder_factory, sim.feature_store, year)

    # --- Build manager display names ---
    team_display = _build_team_manager_display(sim)

    # --- Transcript storage ---
    transcript: list[tuple[int, str, dict, str, int]] = []  # (pick#, nominator, player, winner, bid)

    def _on_pick(pick_num, nominator, player, winner, bid):
        transcript.append((pick_num, nominator, player, winner, bid))

    # --- Inject RL model into the bidding loop ---
    # We drive the draft using simulate_draft() with transcript_callback,
    # and monkey-patch _opponent_max_bid to intercept RL team decisions.
    if rl_bidder:
        _orig_opponent_max_bid = sim._opponent_max_bid

        def _patched_opponent_max_bid(team_name: str, player: dict, current_bid: float = 0.0) -> float:
            if team_name == rl_team:
                state = sim.get_state()
                budget = float(sim.teams[rl_team]["current_budget"])
                return rl_bidder.get_max_bid(state, player, 1, budget)
            return _orig_opponent_max_bid(team_name, player, current_bid)

        sim._opponent_max_bid = _patched_opponent_max_bid  # type: ignore[method-assign]

    # Run the draft
    sim.simulate_draft(transcript_callback=_on_pick)

    # --- Format output ---
    lines: list[str] = []

    header = f"=== {year} AUCTION DRAFT SIMULATION ==="
    lines.append(header)
    if rl_bidder:
        mgr_name = team_display.get(rl_team, "")
        lines.append(f"RL team: {rl_team}" + (f" ({mgr_name})" if mgr_name else ""))
    lines.append("")

    # Transcript table
    lines.append("--- Draft Transcript ---")
    col_w = {"pick": 4, "player": 22, "pos": 5, "nominator": 16, "winner": 18, "bid": 5}
    header_row = (
        f"{'#':>{col_w['pick']}}  "
        f"{'Player':<{col_w['player']}}  "
        f"{'Pos':<{col_w['pos']}}  "
        f"{'Nominated By':<{col_w['nominator']}}  "
        f"{'Winner':<{col_w['winner']}}  "
        f"{'$':>{col_w['bid']}}"
    )
    lines.append(header_row)
    lines.append("-" * len(header_row))

    for (pick_num, nominator, player, winner, bid) in transcript:
        nom_label = f"{nominator} [RL]" if nominator == rl_team and rl_bidder else nominator
        win_label = f"{winner} [RL]" if winner == rl_team and rl_bidder else winner
        lines.append(
            f"{pick_num:>{col_w['pick']}}  "
            f"{player['name']:<{col_w['player']}}  "
            f"{player['position']:<{col_w['pos']}}  "
            f"{nom_label:<{col_w['nominator']}}  "
            f"{win_label:<{col_w['winner']}}  "
            f"${bid:>{col_w['bid'] - 1}}"
        )

    lines.append("")
    lines.append("--- Final Rosters ---")
    lines.append("")

    # Per-team rosters — RL team first, then others
    team_order = [rl_team] + [t for t in sim.teams if t != rl_team] if rl_bidder else list(sim.teams.keys())

    for team_name in team_order:
        team = sim.teams[team_name]
        mgr_name = team_display.get(team_name, "")
        spent = 200 - team["current_budget"]
        t_label = f"{team_name} [RL]" if team_name == rl_team and rl_bidder else team_name
        header_parts = [t_label]
        if mgr_name:
            header_parts.append(f"({mgr_name})")
        header_parts.append(f"Spent: ${spent} / $200")
        lines.append("  ".join(header_parts))

        total_proj = 0.0
        for slot, player in team["roster"].items():
            if player is None:
                lines.append(f"  {slot:<10}  (empty)")
            else:
                proj = float(player.get("projected_points", 0.0))
                total_proj += proj
                lines.append(
                    f"  {slot:<10}  {player['name']:<25}  "
                    f"${player['bid_amount']:<5}  proj: {proj:.0f} pts"
                )
        lines.append(f"  Total projected: {total_proj:,.0f} pts")
        lines.append("")

    output_text = "\n".join(lines)

    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        Path(output_path).write_text(output_text)
        print(f"Simulation saved to: {output_path}")
    else:
        print(output_text)


def main():
    parser = argparse.ArgumentParser(
        description="Run an auction draft simulation for a given year."
    )
    parser.add_argument("--year", type=int, default=2024, help="Draft year (default: 2024)")
    parser.add_argument(
        "--rl-model-path",
        default=None,
        help="Path to a trained AuctionDraftPolicy .pt checkpoint to inject as the RL team.",
    )
    parser.add_argument(
        "--rl-team",
        default="Team 1",
        help="Team slot name for the RL agent (default: 'Team 1'). Only used with --rl-model-path.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Write simulation output to this file instead of stdout.",
    )
    args = parser.parse_args()

    _run_simulation(
        year=args.year,
        rl_model_path=args.rl_model_path,
        rl_team=args.rl_team,
        output_path=args.output,
    )


if __name__ == "__main__":
    main()
