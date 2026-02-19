#!/usr/bin/env python3
"""
End-to-end training launcher for the auction draft RL pipeline.

Chains all training phases in sequence so the full pipeline can be run
with a single command. Each phase loads from the previous phase's final
checkpoint automatically. Crashed runs can be resumed with --from-phase N.

Execution order:
  Phase 1 (100K steps)  — draft warm-up, no season sim
  Phase 2 (500K steps)  — season sim introduced every 10 episodes
  Phase 3 (1M steps)    — full season sim every episode
  [BC reference model]  — optional, trains on historical ESPN data
  Phase 4 (500K steps)  — self-play vs. checkpoint pool

Usage:
    # Full pipeline from scratch
    .venv/bin/python ffai/scripts/train.py

    # Resume from Phase 3 (skips 1, 2)
    .venv/bin/python ffai/scripts/train.py --from-phase 3

    # Skip BC reference training (faster, slightly weaker opponent init)
    .venv/bin/python ffai/scripts/train.py --skip-bc

    # Smoke test: 500 steps per phase, serial backend
    .venv/bin/python ffai/scripts/train.py --smoke-test

    # Phase 4 only with specific seed
    .venv/bin/python ffai/scripts/train.py --from-phase 4 \\
        --seed-pool-path checkpoints/bc/bc_as_policy.pt
"""

import argparse
import subprocess
import sys
from pathlib import Path

# Ensure src/ is on the path when run from the repo root
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

CHECKPOINT_DIR = Path("checkpoints/puffer")
BC_CHECKPOINT = Path("checkpoints/bc/bc_reference.pt")
BC_POLICY_CHECKPOINT = Path("checkpoints/bc/bc_as_policy.pt")
PYTHON = sys.executable


def phase_checkpoint(phase: int) -> Path:
    return CHECKPOINT_DIR / f"phase{phase}_final.pt"


def run(cmd: list[str], description: str) -> None:
    """Run a subprocess command, raising on failure."""
    print(f"\n{'=' * 60}")
    print(f"  {description}")
    print(f"  {' '.join(str(c) for c in cmd)}")
    print(f"{'=' * 60}\n")
    result = subprocess.run(cmd)
    if result.returncode != 0:
        print(f"\nERROR: command exited with code {result.returncode}")
        sys.exit(result.returncode)


def main():
    parser = argparse.ArgumentParser(
        description="End-to-end auction draft RL training launcher"
    )
    parser.add_argument(
        "--from-phase", type=int, choices=[1, 2, 3, 4], default=1,
        help="Start from this phase (earlier phases must already have checkpoints)",
    )
    parser.add_argument(
        "--skip-bc", action="store_true",
        help="Skip behavioral cloning reference model training",
    )
    parser.add_argument(
        "--seed-pool-path", type=Path, default=None,
        help="Phase 4: checkpoint to seed the opponent pool (default: BC policy or phase3)",
    )
    parser.add_argument(
        "--smoke-test", action="store_true",
        help="Run only 500 timesteps per phase for fast end-to-end validation",
    )
    parser.add_argument(
        "--bc-years", type=str, default="2019-2024",
        help="Years to use for BC reference training (e.g. '2019-2024')",
    )
    args = parser.parse_args()

    ts_override = ["--total-timesteps", "500"] if args.smoke_test else []
    serial_flag = ["--serial"] if args.smoke_test else []

    # -----------------------------------------------------------------------
    # Phase 1
    # -----------------------------------------------------------------------
    if args.from_phase <= 1:
        if args.smoke_test or not phase_checkpoint(1).exists():
            run(
                [PYTHON, "ffai/scripts/train_puffer.py", "--curriculum-phase", "1"]
                + ts_override + serial_flag,
                "Phase 1 — draft warm-up (no season sim)",
            )
        else:
            print(f"\n[skip] Phase 1 checkpoint exists: {phase_checkpoint(1)}")

    # -----------------------------------------------------------------------
    # Phase 2
    # -----------------------------------------------------------------------
    if args.from_phase <= 2:
        if args.smoke_test or not phase_checkpoint(2).exists():
            run(
                [
                    PYTHON, "ffai/scripts/train_puffer.py",
                    "--curriculum-phase", "2",
                    "--load-model-path", str(phase_checkpoint(1)),
                ]
                + ts_override + serial_flag,
                "Phase 2 — season sim introduced every 10 episodes",
            )
        else:
            print(f"\n[skip] Phase 2 checkpoint exists: {phase_checkpoint(2)}")

    # -----------------------------------------------------------------------
    # Phase 3
    # -----------------------------------------------------------------------
    if args.from_phase <= 3:
        if args.smoke_test or not phase_checkpoint(3).exists():
            run(
                [
                    PYTHON, "ffai/scripts/train_puffer.py",
                    "--curriculum-phase", "3",
                    "--load-model-path", str(phase_checkpoint(2)),
                ]
                + ts_override + serial_flag,
                "Phase 3 — full season sim every episode",
            )
        else:
            print(f"\n[skip] Phase 3 checkpoint exists: {phase_checkpoint(3)}")

    # -----------------------------------------------------------------------
    # BC Reference Model (optional Stage 2)
    # -----------------------------------------------------------------------
    if not args.skip_bc:
        if not BC_POLICY_CHECKPOINT.exists():
            run(
                [
                    PYTHON, "ffai/scripts/train_bc_reference.py",
                    "--years", args.bc_years,
                    "--export-checkpoint",
                ]
                + (["--epochs", "5"] if args.smoke_test else []),
                "BC Reference Model — behavioral cloning on historical ESPN data",
            )
        else:
            print(f"\n[skip] BC checkpoint exists: {BC_POLICY_CHECKPOINT}")
    else:
        print("\n[skip] BC reference training (--skip-bc)")

    # -----------------------------------------------------------------------
    # Phase 4 — Self-play
    # -----------------------------------------------------------------------
    if args.from_phase <= 4:
        # Determine pool seed: BC policy > explicit seed > phase3 checkpoint
        if args.seed_pool_path:
            seed_pool = args.seed_pool_path
        elif BC_POLICY_CHECKPOINT.exists():
            seed_pool = BC_POLICY_CHECKPOINT
        else:
            seed_pool = phase_checkpoint(3)

        cmd = [
            PYTHON, "ffai/scripts/train_puffer.py",
            "--curriculum-phase", "4",
            "--load-model-path", str(phase_checkpoint(3)),
            "--seed-pool-path", str(seed_pool),
        ] + ts_override + serial_flag

        if args.smoke_test or not phase_checkpoint(4).exists():
            run(cmd, f"Phase 4 — self-play (seed: {seed_pool})")
        else:
            print(f"\n[skip] Phase 4 checkpoint exists: {phase_checkpoint(4)}")

    print(f"\n{'=' * 60}")
    print("  Training complete!")
    print(f"  Final checkpoint: {phase_checkpoint(4)}")
    print(f"{'=' * 60}\n")


if __name__ == "__main__":
    main()
