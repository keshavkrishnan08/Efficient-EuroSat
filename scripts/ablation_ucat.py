"""Ablation study: effect of UCAT loss weight (lambda_ucat).

Trains the EfficientEuroSAT model with different lambda_ucat values
to determine the optimal UCAT loss weight.
"""

import os
import sys
import subprocess

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

LAMBDAS = [0.0, 0.01, 0.05, 0.1, 0.2, 0.5]
SEEDS = [42, 123, 456]


def main():
    for lam in LAMBDAS:
        for seed in SEEDS:
            name = f"ucat_lambda_{lam}_seed_{seed}"
            print(f"\n{'='*60}")
            print(f"Running: lambda_ucat={lam}, seed={seed}")
            print(f"{'='*60}\n")

            cmd = [
                sys.executable,
                os.path.join(os.path.dirname(__file__), "train.py"),
                "--lambda_ucat", str(lam),
                "--seed", str(seed),
                "--experiment_name", name,
                "--no_wandb",
            ]
            subprocess.run(cmd, check=False)

    print("\nUCAT ablation complete.")


if __name__ == "__main__":
    main()
