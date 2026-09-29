from __future__ import annotations

import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from dynafm_demo import RealisticConfig, run_realistic_experiment
from dynafm_demo.realistic import create_realistic_figures


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the realistic dynamical-family benchmark.")
    parser.add_argument("--output", default="results", help="Output directory relative to the project root.")
    parser.add_argument("--seed", type=int, default=12)
    args = parser.parse_args()
    output = PROJECT_ROOT / args.output
    metrics = run_realistic_experiment(output, RealisticConfig(seed=args.seed))
    create_realistic_figures(output)

    print("Realistic benchmark completed")
    for name, result in metrics["rmse"].items():
        print(f"{name:16s}: {result['mean']:.5f} +/- {result['std']:.5f}")
    comparison = metrics["active_vs_random"]
    print(f"Active win rate:         {100 * comparison['active_win_rate']:.1f}%")
    print(f"Relative RMSE reduction: {100 * comparison['relative_mean_reduction']:.1f}%")
    print(f"Results: {output}")


if __name__ == "__main__":
    main()
