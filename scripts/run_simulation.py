"""
Run WC2026 Monte Carlo simulation and save results to data/simulation_results.json.
Usage: python scripts/run_simulation.py [--n 10000] [--seed 42]
"""

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.simulate import run_monte_carlo

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=10_000, help="Number of simulations")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    print(f"Running {args.n:,} WC2026 simulations (seed={args.seed})...")
    t0 = time.time()
    probs = run_monte_carlo(n_simulations=args.n, seed=args.seed)
    elapsed = time.time() - t0

    # Sort by win probability
    probs_sorted = dict(sorted(probs.items(), key=lambda x: x[1], reverse=True))

    out_dir = Path(__file__).parent.parent / "data"
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / "simulation_results.json"
    with open(out_path, "w") as f:
        json.dump({"n_simulations": args.n, "seed": args.seed, "win_probs": probs_sorted}, f, indent=2)

    print(f"\nCompleted in {elapsed:.1f}s. Results saved to {out_path}\n")
    print(f"{'Team':<20} {'Win %':>7}")
    print("-" * 29)
    for team, prob in list(probs_sorted.items())[:16]:
        if prob > 0:
            print(f"{team:<20} {prob*100:>6.2f}%")


if __name__ == "__main__":
    main()
