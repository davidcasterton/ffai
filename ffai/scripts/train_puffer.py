#!/usr/bin/env python3
"""
PufferLib PPO training for AuctionDraftEnv — 4-phase curriculum.

Phase 1  (draft-only warm-up, 100K steps):
    No season simulation. Fast validation that the pipeline works.
    Target: loss decreases, no crashes.

Phase 2  (season-sim introduced, 500K steps):
    Season simulator runs every 10 episodes per worker (~20% of episodes).
    Target: agent starts winning more weekly matchups.

Phase 3  (full season-sim, 1M steps):
    Season simulator runs every episode. Frozen opponent policies optional.
    Target: standing < 6 on average, wins > 8.

Phase 4  (self-play, 500K steps):
    Season simulator every episode. Opponents are a mix of past RL checkpoints
    (70%) and heuristic bidders (30%), sampled per episode from OpponentPool.
    Pool is seeded with phase3_final.pt and enriched every 50K steps.
    Target: budget utilization > 90%, standing < 5 on average.

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

    # Phase 4 (self-play — seed pool with phase3 checkpoint):
    python scripts/train_puffer.py --curriculum-phase 4 \\
        --load-model-path checkpoints/puffer/phase3_final.pt \\
        --seed-pool-path checkpoints/puffer/phase3_final.pt
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
from ffai.rl.opponent_pool import OpponentPool

# ---------------------------------------------------------------------------
# Phase configs (override puffer.ini defaults)
# ---------------------------------------------------------------------------

PHASE_CONFIGS = {
    1: {
        "total_timesteps": 100_000,
        "enable_season_sim": False,
        "season_sim_interval": 10,
        "num_envs": 1,
        "self_play": False,
        "description": "draft-only warm-up (no season sim)",
    },
    2: {
        "total_timesteps": 500_000,
        "enable_season_sim": True,
        "season_sim_interval": 10,
        "num_envs": 4,
        "self_play": False,
        "description": "season-sim introduced every 10 episodes",
    },
    3: {
        "total_timesteps": 1_000_000,
        "enable_season_sim": True,
        "season_sim_interval": 1,
        "num_envs": 6,
        "self_play": False,
        "description": "full season-sim every episode",
    },
    4: {
        "total_timesteps": 500_000,
        "enable_season_sim": True,
        "season_sim_interval": 1,
        "num_envs": 6,
        "self_play": True,
        "description": "self-play vs. checkpoint pool (70% learned / 30% heuristic)",
    },
}

CONFIG_PATH = Path(__file__).parent.parent / "src/ffai/config/puffer.ini"
CHECKPOINT_DIR = Path("checkpoints/puffer")


# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------

def load_config(phase: int, total_timesteps_override: int | None = None, self_play_section: bool = False) -> dict:
    """Load puffer.ini and apply phase overrides."""
    parser = configparser.ConfigParser()
    parser.read(CONFIG_PATH)

    train_section = dict(parser["train"]) if "train" in parser else {}

    # Read self_play section if requested (Phase 4)
    self_play_overrides: dict = {}
    if self_play_section and "self_play" in parser:
        def _coerce_ini(v: str):
            v = v.strip()
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
            return v
        self_play_overrides = {k: _coerce_ini(v) for k, v in dict(parser["self_play"]).items()}

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

    # Apply self_play section overrides on top of train section
    config.update(self_play_overrides)

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

def make_env_creator(phase: int, opponent_pool: OpponentPool | None = None):
    """Return an env_creator callable suitable for pufferlib.vector."""
    phase_cfg = PHASE_CONFIGS[phase]

    def env_creator(buf=None, seed=None):
        return AuctionDraftEnv(
            year=2024,
            enable_season_sim=phase_cfg["enable_season_sim"],
            season_sim_interval=phase_cfg["season_sim_interval"],
            opponent_pool=opponent_pool,
            buf=buf,
            seed=seed,
        )

    return env_creator


# ---------------------------------------------------------------------------
# Self-play helpers
# ---------------------------------------------------------------------------


def _build_opponent_pool(phase_cfg: dict, config: dict, args) -> OpponentPool | None:
    """Build and optionally seed an OpponentPool for Phase 4 self-play."""
    if not phase_cfg["self_play"]:
        return None

    sp_heuristic_frac = float(config.get("heuristic_fraction", 0.3))
    sp_max_pool = int(config.get("max_pool_size", 10))
    pool = OpponentPool(
        heuristic_fraction=sp_heuristic_frac,
        max_pool_size=sp_max_pool,
        device=str(config.get("device", "cpu")),
    )

    # Seed pool with BC checkpoint or previous phase checkpoint
    seed_path: Path | None = None
    if args.seed_pool_path:
        seed_path = Path(args.seed_pool_path)
    elif args.load_model_path:
        seed_path = Path(args.load_model_path)

    if seed_path is not None and seed_path.exists():
        pool.add_checkpoint(seed_path)
        print(f"    Seeded opponent pool with {seed_path}")

    print(f"    OpponentPool: heuristic_fraction={sp_heuristic_frac}, max_pool={sp_max_pool}")
    return pool


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train(args):
    phase = args.curriculum_phase
    phase_cfg = PHASE_CONFIGS[phase]
    num_envs = args.num_envs if args.num_envs is not None else phase_cfg["num_envs"]

    print(f"\n=== Phase {phase}: {phase_cfg['description']} ===")
    print(f"    num_envs={num_envs}, total_timesteps={args.total_timesteps or phase_cfg['total_timesteps']}")

    config = load_config(
        phase,
        total_timesteps_override=args.total_timesteps,
        self_play_section=(phase == 4),
    )
    config["env"] = f"auction_draft_phase{phase}"

    # Phase 4: build OpponentPool and optionally seed it
    opponent_pool = _build_opponent_pool(phase_cfg, config, args)

    # Build vecenv (Serial for phase 1 smoke tests, Multiprocessing otherwise)
    env_creator = make_env_creator(phase, opponent_pool=opponent_pool)
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
            state_dict = torch.load(load_path, map_location=config["device"], weights_only=True)
            policy.load_state_dict(state_dict)
            print(f"    Loaded model from {load_path}")
        else:
            print(f"    WARNING: --load-model-path {load_path} not found, starting fresh")

    # Self-play checkpoint interval (Phase 4 only)
    checkpoint_interval_steps = int(config.get("checkpoint_interval_steps", 50_000))

    # Run training
    trainer = PuffeRL(config=config, vecenv=vecenv, policy=policy)
    last_pool_update_step = 0

    done_training = False
    while not done_training:
        trainer.evaluate()
        logs = trainer.train()
        if logs is not None:
            done_training = trainer.global_step >= config["total_timesteps"]

            # Phase 4: periodically snapshot current policy into opponent pool
            if opponent_pool is not None:
                steps_since_update = trainer.global_step - last_pool_update_step
                if steps_since_update >= checkpoint_interval_steps:
                    snap_path = CHECKPOINT_DIR / f"phase4_snap_{trainer.global_step}.pt"
                    snap_path.parent.mkdir(parents=True, exist_ok=True)
                    torch.save(trainer.uncompiled_policy.state_dict(), snap_path)
                    opponent_pool.add_checkpoint(snap_path)
                    last_pool_update_step = trainer.global_step
                    print(f"    [self-play] Added snapshot at step {trainer.global_step}")

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
        "--curriculum-phase", type=int, choices=[1, 2, 3, 4], default=1,
        help="Training phase: 1=draft-only, 2=season-sim/10, 3=season-sim/1, 4=self-play",
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
        "--seed-pool-path", type=Path, default=None,
        help="Phase 4: path to checkpoint used to seed the OpponentPool at startup",
    )
    parser.add_argument(
        "--serial", action="store_true",
        help="Force Serial backend even with multiple envs (useful for debugging)",
    )
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
