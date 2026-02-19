#!/usr/bin/env python3
"""
Train a Behavioral Cloning (BC) reference model on historical ESPN draft data.

For each historical pick, reconstructs:
    (state_72_dim, bid_fraction = bid_amount / budget_at_time)
using the perspective-aware state builder so that the BC model learns from
the manager's actual point of view at each decision.

The trained model is saved to checkpoints/bc/bc_reference.pt and can be:
  1. Exported as an opponent checkpoint to seed the Phase 4 self-play pool
  2. Used as an auxiliary reward signal (bc_alpha * log_prob) in Phase 4

Usage:
    .venv/bin/python ffai/scripts/train_bc_reference.py
    .venv/bin/python ffai/scripts/train_bc_reference.py --years 2019-2024
    .venv/bin/python ffai/scripts/train_bc_reference.py --export-checkpoint
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from ffai.rl.bc_reference import BCReferenceModel
from ffai.rl.state_builder import build_state
from ffai.data.espn_scraper import ESPNDraftScraper, load_league_config

CHECKPOINT_DIR = Path("checkpoints/bc")
BUDGET = 200.0


# ---------------------------------------------------------------------------
# Data extraction
# ---------------------------------------------------------------------------


def build_bc_dataset(years: list[int]) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Extract (state, bid_fraction) pairs from historical ESPN draft data.

    For each pick in each year, reconstructs the draft state from the
    picking team's perspective at the time of that pick and computes the
    bid fraction = bid_amount / budget_at_start_of_round.

    Returns:
        obs_tensor: (N, 72) float32 tensor
        target_tensor: (N, 1) float32 tensor, bid fractions ∈ (0, 1)
    """
    scraper = ESPNDraftScraper()
    cfg = load_league_config()
    league_name = cfg["league"]["league_name"]

    try:
        from ffai.data.feature_store import FeatureStore
        feature_store = FeatureStore()
    except Exception:
        feature_store = None

    all_obs: list[np.ndarray] = []
    all_targets: list[float] = []

    for year in years:
        print(f"  Processing year {year}...")
        try:
            draft_df, stats_df, weekly_df, predraft_df, settings = scraper.load_or_fetch_data(year)
        except Exception as e:
            print(f"    Skipping {year}: {e}")
            continue

        if draft_df is None or len(draft_df) == 0:
            print(f"    No draft data for {year}, skipping")
            continue

        # Sort by pick_number to process picks in order
        if "pick_number" in draft_df.columns:
            draft_df = draft_df.sort_values("pick_number")

        # Track each team's remaining budget as picks are processed
        team_budgets: dict[str, float] = {}
        team_position_counts: dict[str, dict[str, int]] = {}

        for _, pick in draft_df.iterrows():
            team_name = str(pick.get("team_name", "unknown"))
            bid_amount = float(pick.get("bid_amount", 1))
            position = str(pick.get("position", ""))

            # Initialize team budget tracking
            if team_name not in team_budgets:
                team_budgets[team_name] = BUDGET
                team_position_counts[team_name] = {}

            remaining_budget = team_budgets[team_name]
            if remaining_budget <= 0:
                continue

            # bid_fraction = fraction of budget at time of pick
            bid_fraction = float(np.clip(bid_amount / remaining_budget, 0.01, 0.99))

            # Build a minimal state dict for this pick
            # We use simplified state since we don't have full sim state for historical data
            state_dict = _build_minimal_state(
                team_name=team_name,
                remaining_budget=remaining_budget,
                bid_amount=bid_amount,
                pick_row=pick,
                draft_df=draft_df,
                predraft_df=predraft_df,
            )

            # Build current player dict
            player_id = str(pick.get("player_id", ""))
            player_dict = _build_player_dict(pick, predraft_df, player_id)

            obs = build_state(
                state_dict,
                team_name=team_name,
                current_player=player_dict,
                current_bid=float(bid_amount - 1),  # bid before winning
                feature_store=feature_store,
                year=year,
            ).numpy()

            all_obs.append(obs)
            all_targets.append(bid_fraction)

            # Update budget
            team_budgets[team_name] = max(0.0, remaining_budget - bid_amount)
            team_position_counts[team_name][position] = (
                team_position_counts[team_name].get(position, 0) + 1
            )

        print(f"    Extracted {len(all_obs)} picks from {year}")

    if not all_obs:
        raise ValueError("No valid picks extracted. Check data availability.")

    obs_tensor = torch.tensor(np.stack(all_obs), dtype=torch.float32)
    target_tensor = torch.tensor(all_targets, dtype=torch.float32).unsqueeze(1)
    print(f"\nTotal BC dataset: {len(obs_tensor)} picks")
    print(f"Bid fraction stats: mean={target_tensor.mean():.3f}, std={target_tensor.std():.3f}")
    return obs_tensor, target_tensor


def _build_minimal_state(
    team_name: str,
    remaining_budget: float,
    bid_amount: float,
    pick_row,
    draft_df,
    predraft_df,
) -> dict:
    """Build a minimal state dict compatible with build_state() for a historical pick."""
    # Estimate position needs from how many roster slots remain
    remaining_slots = max(1, int(remaining_budget))  # rough proxy

    # Position needs: use standard roster as simplified approximation
    position_needs = {"QB": 1, "RB": 2, "WR": 2, "TE": 1, "D/ST": 1, "K": 1}

    # Position values from predraft data
    position_values = {}
    for pos in ["QB", "RB", "WR", "TE", "D/ST", "K"]:
        pos_players = predraft_df[predraft_df["position"] == pos] if predraft_df is not None else None
        if pos_players is not None and len(pos_players) > 0:
            position_values[pos] = {
                "avg_value": float(pos_players.get("auction_value", 0).mean() if "auction_value" in pos_players.columns else 0),
                "avg_points": float(pos_players.get("projected_points", 0).mean() if "projected_points" in pos_players.columns else 0),
            }

    return {
        "rl_team_budget": remaining_budget,
        "opponent_budgets": [BUDGET * 0.7] * 11,  # simplified approximation
        "draft_turn": 0,
        "teams": [f"Team {i+1}" for i in range(12)],
        "predicted_points_per_slot": {},
        "position_needs": position_needs,
        "position_counts": {},
        "position_scarcity": {pos: 0.0 for pos in ["QB", "RB", "WR", "TE", "D/ST", "K"]},
        "position_values": position_values,
        "remaining_budget_per_need": remaining_budget / max(1, remaining_slots),
        "draft_progress": 0.5,  # simplified
        "total_team_points": 0.0,
        "opponent_tendencies": [],
    }


def _build_player_dict(pick_row, predraft_df, player_id: str) -> dict:
    """Build a player dict from a draft pick row."""
    player = {
        "player_id": player_id,
        "name": str(pick_row.get("player_name", pick_row.get("name", "unknown"))),
        "position": str(pick_row.get("position", "")),
        "projected_points": float(pick_row.get("projected_points", 0)),
        "auction_value": float(pick_row.get("auction_value", 1)),
        "PAR": float(pick_row.get("PAR", 0)),
        "VORP_dollar": float(pick_row.get("VORP_dollar", pick_row.get("auction_value", 1))),
    }

    # Try to enrich from predraft_df
    if predraft_df is not None and "player_id" in predraft_df.columns:
        matches = predraft_df[predraft_df["player_id"].astype(str) == player_id]
        if len(matches) > 0:
            row = matches.iloc[0]
            for col in ["projected_points", "auction_value", "PAR", "VORP_dollar"]:
                if col in row.index and row[col] is not None:
                    player[col] = float(row[col])

    return player


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------


def train_bc(obs: torch.Tensor, targets: torch.Tensor, args) -> BCReferenceModel:
    """Train BCReferenceModel with MSE loss."""
    n = len(obs)
    val_size = max(1, int(n * 0.1))
    train_obs, val_obs = obs[:-val_size], obs[-val_size:]
    train_tgt, val_tgt = targets[:-val_size], targets[-val_size:]

    train_ds = TensorDataset(train_obs, train_tgt)
    val_ds = TensorDataset(val_obs, val_tgt)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size)

    model = BCReferenceModel()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    best_val_loss = float("inf")
    patience_count = 0

    print(f"\nTraining BCReferenceModel for up to {args.epochs} epochs")
    print(f"  Train: {len(train_obs)} picks, Val: {len(val_obs)} picks")

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_losses = []
        for obs_batch, tgt_batch in train_loader:
            optimizer.zero_grad()
            pred = model(obs_batch)
            loss = nn.MSELoss()(pred, tgt_batch)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            optimizer.step()
            train_losses.append(loss.item())

        model.eval()
        val_losses = []
        with torch.no_grad():
            for obs_batch, tgt_batch in val_loader:
                pred = model(obs_batch)
                val_losses.append(nn.MSELoss()(pred, tgt_batch).item())

        train_loss = np.mean(train_losses)
        val_loss = np.mean(val_losses)

        if epoch % 10 == 0 or epoch == 1:
            print(f"  Epoch {epoch:3d}: train={train_loss:.4f}, val={val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_count = 0
            model.save(CHECKPOINT_DIR / "bc_reference_best.pt")
        else:
            patience_count += 1
            if patience_count >= args.patience:
                print(f"  Early stopping at epoch {epoch} (patience={args.patience})")
                break

    print(f"\nBest validation loss: {best_val_loss:.4f}")
    return model


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def parse_years(years_str: str) -> list[int]:
    if "-" in years_str:
        start, end = years_str.split("-")
        return list(range(int(start), int(end) + 1))
    return [int(y) for y in years_str.split(",")]


def main():
    parser = argparse.ArgumentParser(
        description="Train BC reference model on historical ESPN draft data"
    )
    parser.add_argument(
        "--years", type=str, default="2019-2024",
        help="Years to use for training (e.g. '2019-2024' or '2021,2022,2023')",
    )
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument(
        "--export-checkpoint", action="store_true",
        help="After training, export as AuctionDraftPolicy checkpoint for OpponentPool",
    )
    args = parser.parse_args()

    years = parse_years(args.years)
    print(f"Building BC dataset from years: {years}")

    obs, targets = build_bc_dataset(years)
    model = train_bc(obs, targets, args)

    # Save final model
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    final_path = CHECKPOINT_DIR / "bc_reference.pt"
    model.save(final_path)
    print(f"\nBC reference model saved to {final_path}")

    if args.export_checkpoint:
        export_path = CHECKPOINT_DIR / "bc_as_policy.pt"
        policy = model.as_checkpoint_policy(export_path)
        print(f"Exported as checkpoint policy to {export_path}")
        print("Add to OpponentPool with: pool.add_checkpoint(Path('checkpoints/bc/bc_as_policy.pt'))")


if __name__ == "__main__":
    main()
