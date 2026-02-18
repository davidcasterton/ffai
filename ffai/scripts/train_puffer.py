#!/usr/bin/env python3
"""
PufferLib PPO training for AuctionDraftEnv — 3-phase curriculum.

Phase 1  (draft-only warm-up, 100K steps):
    No season simulation. Fast validation that the pipeline works.
    Target: loss decreases, no crashes.

Phase 2  (season-sim introduced, 500K steps):
    Season simulator runs every 10 episodes per worker (~20% of episodes).
    Target: agent starts winning more weekly matchups.

Phase 3  (full season-sim, 1M steps):
    Season simulator runs every episode. Frozen opponent policies optional.
    Target: standing < 6 on average, wins > 8.

Usage:
    # Phase 1 (smoke test — 500 timesteps):
    python scripts/train_puffer.py --curriculum-phase 1 --total-timesteps 500

    # Full Phase 1:
    python scripts/train_puffer.py --curriculum-phase 1

    # Phase 2 (load from Phase 1 checkpoint):
    python scripts/train_puffer.py --curriculum-phase 2 \\
        --load-model-path checkpoints/puffer/phase1_final.pt

    # Phase 3:
    python scripts/train_puffer.py --curriculum-phase 3 \\
        --load-model-path checkpoints/puffer/phase2_final.pt
"""

import argparse
import configparser
import sys
from pathlib import Path

# Ensure src/ is on the path when run from the repo root
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import torch

import pufferlib
import pufferlib.vector
import pufferlib.pytorch
from pufferlib.pufferl import PuffeRL

from ffai.rl.puffer_env import AuctionDraftEnv
from ffai.rl.puffer_policy import AuctionDraftPolicy

# ---------------------------------------------------------------------------
# Phase configs (override puffer.ini defaults)
# ---------------------------------------------------------------------------

PHASE_CONFIGS = {
    1: {
        "total_timesteps": 100_000,
        "enable_season_sim": False,
        "season_sim_interval": 10,
        "num_envs": 1,
        "description": "draft-only warm-up (no season sim)",
    },
    2: {
        "total_timesteps": 500_000,
        "enable_season_sim": True,
        "season_sim_interval": 10,
        "num_envs": 4,
        "description": "season-sim introduced every 10 episodes",
    },
    3: {
        "total_timesteps": 1_000_000,
        "enable_season_sim": True,
        "season_sim_interval": 1,
        "num_envs": 8,
        "description": "full season-sim every episode",
    },
}

CONFIG_PATH = Path(__file__).parent.parent / "src/ffai/config/puffer.ini"
CHECKPOINT_DIR = Path("checkpoints/puffer")


# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------

def load_config(phase: int, total_timesteps_override: int | None = None) -> dict:
    """Load puffer.ini and apply phase overrides."""
    parser = configparser.ConfigParser()
    parser.read(CONFIG_PATH)

    train_section = dict(parser["train"]) if "train" in parser else {}

    def _coerce(v: str):
        """Parse INI string to Python type."""
        v = v.strip()
        # Remove underscores from numeric literals (e.g. 100_000)
        stripped = v.replace("_", "")
        try:
            return int(stripped)
        except ValueError:
            pass
        try:
            return float(stripped)
        except ValueError:
            pass
        if v.lower() in ("true", "yes"):
            return True
        if v.lower() in ("false", "no"):
            return False
        if v.lower() in ("auto", "none"):
            return v.lower()
        return v

    config = {k: _coerce(v) for k, v in train_section.items()}

    # Apply phase overrides
    phase_cfg = PHASE_CONFIGS[phase]
    config["total_timesteps"] = phase_cfg["total_timesteps"]

    # CLI override takes highest priority
    if total_timesteps_override is not None:
        config["total_timesteps"] = total_timesteps_override

    # Required keys that puffer.ini might not set explicitly
    config.setdefault("env", "auction_draft")
    config.setdefault("batch_size", 2048)
    config.setdefault("bptt_horizon", 64)
    config.setdefault("minibatch_size", 512)
    config.setdefault("max_minibatch_size", 2048)
    config.setdefault("use_rnn", False)

    # Ensure data_dir exists
    Path(str(config.get("data_dir", "checkpoints/puffer"))).mkdir(parents=True, exist_ok=True)

    return config


# ---------------------------------------------------------------------------
# Environment factory
# ---------------------------------------------------------------------------

def make_env_creator(phase: int):
    """Return an env_creator callable suitable for pufferlib.vector."""
    phase_cfg = PHASE_CONFIGS[phase]

    def env_creator(buf=None, seed=None):
        return AuctionDraftEnv(
            year=2024,
            enable_season_sim=phase_cfg["enable_season_sim"],
            season_sim_interval=phase_cfg["season_sim_interval"],
            buf=buf,
            seed=seed,
        )

    return env_creator


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train(args):
    phase = args.curriculum_phase
    phase_cfg = PHASE_CONFIGS[phase]
    num_envs = args.num_envs if args.num_envs is not None else phase_cfg["num_envs"]

    print(f"\n=== Phase {phase}: {phase_cfg['description']} ===")
    print(f"    num_envs={num_envs}, total_timesteps={args.total_timesteps or phase_cfg['total_timesteps']}")

    config = load_config(phase, total_timesteps_override=args.total_timesteps)
    config["env"] = f"auction_draft_phase{phase}"

    # Build vecenv (Serial for phase 1 smoke tests, Multiprocessing otherwise)
    env_creator = make_env_creator(phase)
    env_creators = [env_creator] * num_envs
    env_args = [[] for _ in range(num_envs)]
    env_kwargs = [{} for _ in range(num_envs)]

    if num_envs == 1 or args.serial:
        vecenv = pufferlib.vector.Serial(
            env_creators=env_creators,
            env_args=env_args,
            env_kwargs=env_kwargs,
            num_envs=num_envs,
        )
    else:
        vecenv = pufferlib.vector.Multiprocessing(
            env_creators=env_creators,
            env_args=env_args,
            env_kwargs=env_kwargs,
            num_envs=num_envs,
        )

    # Build policy
    policy = AuctionDraftPolicy(env=vecenv).to(config["device"])

    # Optionally load weights from a previous phase checkpoint
    if args.load_model_path:
        load_path = Path(args.load_model_path)
        if load_path.exists():
            state_dict = torch.load(load_path, map_location=config["device"])
            policy.load_state_dict(state_dict)
            print(f"    Loaded model from {load_path}")
        else:
            print(f"    WARNING: --load-model-path {load_path} not found, starting fresh")

    # Run training
    trainer = PuffeRL(config=config, vecenv=vecenv, policy=policy)

    done_training = False
    while not done_training:
        trainer.evaluate()
        logs = trainer.train()
        if logs is not None:
            done_training = trainer.global_step >= config["total_timesteps"]

    # Save final checkpoint
    final_path = CHECKPOINT_DIR / f"phase{phase}_final.pt"
    final_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(trainer.uncompiled_policy.state_dict(), final_path)
    print(f"\nPhase {phase} complete. Model saved to {final_path}")

    trainer.close()
    return final_path


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Train AuctionDraft PPO agent via PufferLib"
    )
    parser.add_argument(
        "--curriculum-phase", type=int, choices=[1, 2, 3], default=1,
        help="Training phase: 1=draft-only, 2=season-sim/10, 3=season-sim/1",
    )
    parser.add_argument(
        "--total-timesteps", type=int, default=None,
        help="Override total timesteps from config (useful for smoke tests)",
    )
    parser.add_argument(
        "--num-envs", type=int, default=None,
        help="Override number of parallel environments",
    )
    parser.add_argument(
        "--load-model-path", type=Path, default=None,
        help="Path to .pt file to load policy weights from (for curriculum handoff)",
    )
    parser.add_argument(
        "--serial", action="store_true",
        help="Force Serial backend even with multiple envs (useful for debugging)",
    )
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
