#!/usr/bin/env python3
"""
Regression gate: compare candidate checkpoint vs baseline on fixed backtest setup.

Exits non-zero when candidate underperforms baseline by more than allowed drop.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_SCRIPT_DIR))

from eval_backtest import evaluate, _parse_years  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Regression gate for policy checkpoints")
    parser.add_argument("--candidate", type=Path, required=True, help="Candidate checkpoint .pt")
    parser.add_argument("--baseline", type=Path, required=True, help="Baseline checkpoint .pt")
    parser.add_argument("--years", type=str, default="2024", help="Years to evaluate")
    parser.add_argument("--seeds", type=int, default=3, help="Seeds per year")
    parser.add_argument("--max-winrate-drop", type=float, default=0.01, help="Allowed absolute drop in win-rate estimate")
    parser.add_argument("--max-standing-increase", type=float, default=0.2, help="Allowed increase in average standing")
    parser.add_argument("--output", type=Path, default=None, help="Optional JSON report path")
    args = parser.parse_args()

    years = _parse_years(args.years)
    cand = evaluate(checkpoint=args.candidate, years=years, num_seeds=args.seeds, rl_team="Team 1")
    base = evaluate(checkpoint=args.baseline, years=years, num_seeds=args.seeds, rl_team="Team 1")

    cand_m = cand["metrics"]
    base_m = base["metrics"]

    winrate_drop = float(base_m["win_rate_estimate"] - cand_m["win_rate_estimate"])
    standing_increase = float(cand_m["avg_standing"] - base_m["avg_standing"])

    passed = True
    reasons = []
    if winrate_drop > args.max_winrate_drop:
        passed = False
        reasons.append(
            f"win-rate dropped by {winrate_drop:.4f} (allowed {args.max_winrate_drop:.4f})"
        )
    if standing_increase > args.max_standing_increase:
        passed = False
        reasons.append(
            f"avg standing increased by {standing_increase:.4f} (allowed {args.max_standing_increase:.4f})"
        )

    report = {
        "passed": passed,
        "candidate": str(args.candidate),
        "baseline": str(args.baseline),
        "years": years,
        "seeds": args.seeds,
        "thresholds": {
            "max_winrate_drop": args.max_winrate_drop,
            "max_standing_increase": args.max_standing_increase,
        },
        "deltas": {
            "winrate_drop": winrate_drop,
            "standing_increase": standing_increase,
        },
        "candidate_metrics": cand_m,
        "baseline_metrics": base_m,
        "fail_reasons": reasons,
    }

    text = json.dumps(report, indent=2)
    print(text)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n")
        print(f"\nSaved report to {args.output}")

    if not passed:
        sys.exit(1)


if __name__ == "__main__":
    main()
