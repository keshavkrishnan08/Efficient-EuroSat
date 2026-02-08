#!/usr/bin/env python3
"""
Master training orchestrator for EfficientEuroSAT ablation experiments.

Defines all 10 experiment configurations from the ablation study plan and
runs them sequentially by invoking train.py via subprocess. Each experiment
gets its own checkpoint directory and can be selectively run or resumed.

Experiments:
    1. baseline           -- Baseline ViT (no modifications)
    2. ucat_only          -- +UCAT loss only
    3. early_exit_only    -- +Early Exit only
    4. dropout_only       -- +Learned Dropout only
    5. residual_only      -- +Learned Residual only
    6. annealing_only     -- +Temperature Annealing only
    7. all_combined_s42   -- All modifications combined, seed 42
    8. all_no_ucat        -- All modifications minus UCAT, seed 42
    9. all_combined_s123  -- All modifications combined, seed 123
   10. all_combined_s456  -- All modifications combined, seed 456

Usage:
    # Run all experiments
    python run_experiments.py

    # Run specific experiments by number
    python run_experiments.py --run 1 2 7

    # Resume (skip experiments whose checkpoints already exist)
    python run_experiments.py --resume

    # Custom epochs, batch size, data root
    python run_experiments.py --epochs 50 --batch_size 32 --data_root /data/eurosat

    # Enable Weights & Biases logging
    python run_experiments.py --use_wandb
"""

import argparse
import os
import subprocess
import sys
import time


# ---------------------------------------------------------------------------
# Experiment definitions
# ---------------------------------------------------------------------------

EXPERIMENTS = [
    {
        "number": 1,
        "name": "baseline",
        "description": "Baseline ViT",
        "flags": [
            "--model", "baseline",
            "--seed", "42",
        ],
    },
    {
        "number": 2,
        "name": "ucat_only",
        "description": "+UCAT only",
        "flags": [
            "--model", "efficient_eurosat",
            "--no_early_exit",
            "--no_learned_dropout",
            "--no_learned_residual",
            "--no_temp_annealing",
            "--lambda_ucat", "0.1",
            "--seed", "42",
        ],
    },
    {
        "number": 3,
        "name": "early_exit_only",
        "description": "+Early Exit only",
        "flags": [
            "--model", "efficient_eurosat",
            "--no_learned_temp",
            "--no_learned_dropout",
            "--no_learned_residual",
            "--no_temp_annealing",
            "--lambda_ucat", "0.0",
            "--seed", "42",
        ],
    },
    {
        "number": 4,
        "name": "dropout_only",
        "description": "+Learned Dropout only",
        "flags": [
            "--model", "efficient_eurosat",
            "--no_learned_temp",
            "--no_early_exit",
            "--no_learned_residual",
            "--no_temp_annealing",
            "--lambda_ucat", "0.0",
            "--seed", "42",
        ],
    },
    {
        "number": 5,
        "name": "residual_only",
        "description": "+Learned Residual only",
        "flags": [
            "--model", "efficient_eurosat",
            "--no_learned_temp",
            "--no_early_exit",
            "--no_learned_dropout",
            "--no_temp_annealing",
            "--lambda_ucat", "0.0",
            "--seed", "42",
        ],
    },
    {
        "number": 6,
        "name": "annealing_only",
        "description": "+Temp Annealing only",
        "flags": [
            "--model", "efficient_eurosat",
            "--no_early_exit",
            "--no_learned_dropout",
            "--no_learned_residual",
            "--lambda_ucat", "0.0",
            "--seed", "42",
        ],
    },
    {
        "number": 7,
        "name": "all_combined_s42",
        "description": "All combined, seed 42",
        "flags": [
            "--model", "efficient_eurosat",
            "--lambda_ucat", "0.1",
            "--seed", "42",
        ],
    },
    {
        "number": 8,
        "name": "all_no_ucat",
        "description": "All - UCAT",
        "flags": [
            "--model", "efficient_eurosat",
            "--lambda_ucat", "0.0",
            "--seed", "42",
        ],
    },
    {
        "number": 9,
        "name": "all_combined_s123",
        "description": "All combined, seed 123",
        "flags": [
            "--model", "efficient_eurosat",
            "--lambda_ucat", "0.1",
            "--seed", "123",
        ],
    },
    {
        "number": 10,
        "name": "all_combined_s456",
        "description": "All combined, seed 456",
        "flags": [
            "--model", "efficient_eurosat",
            "--lambda_ucat", "0.1",
            "--seed", "456",
        ],
    },
    # --- Decomposition ablation (E14 A-D) ---
    {
        "number": 11,
        "name": "decomp_input_dep_only",
        "description": "Input-dep tau_a, fixed tau_e",
        "flags": [
            "--model", "efficient_eurosat",
            "--use_decomposition",
            "--lambda_ucat", "0.1",
            "--lambda_aleatoric", "0.0",
            "--lambda_epistemic", "0.0",
            "--seed", "42",
        ],
    },
    {
        "number": 12,
        "name": "decomp_no_losses",
        "description": "Decomp, no decomp losses",
        "flags": [
            "--model", "efficient_eurosat",
            "--use_decomposition",
            "--lambda_ucat", "0.1",
            "--lambda_aleatoric", "0.0",
            "--lambda_epistemic", "0.0",
            "--seed", "42",
        ],
    },
    {
        "number": 13,
        "name": "decomp_with_losses",
        "description": "Decomp + all losses",
        "flags": [
            "--model", "efficient_eurosat",
            "--use_decomposition",
            "--lambda_ucat", "0.1",
            "--lambda_aleatoric", "0.05",
            "--lambda_epistemic", "0.05",
            "--seed", "42",
        ],
    },
    {
        "number": 14,
        "name": "decomp_with_losses_s123",
        "description": "Decomp + losses, seed 123",
        "flags": [
            "--model", "efficient_eurosat",
            "--use_decomposition",
            "--lambda_ucat", "0.1",
            "--lambda_aleatoric", "0.05",
            "--lambda_epistemic", "0.05",
            "--seed", "123",
        ],
    },
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def parse_args():
    """Parse command-line arguments for the orchestrator."""
    parser = argparse.ArgumentParser(
        description="Run all EfficientEuroSAT ablation experiments",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--run", type=int, nargs="+", default=None,
        help="Experiment numbers to run (e.g. --run 1 2 7). Default: run all.",
    )
    parser.add_argument(
        "--resume", action="store_true",
        help="Skip experiments whose checkpoint directory already contains a checkpoint file.",
    )
    parser.add_argument(
        "--epochs", type=int, default=100,
        help="Number of training epochs per experiment.",
    )
    parser.add_argument(
        "--batch_size", type=int, default=64,
        help="Training batch size.",
    )
    parser.add_argument(
        "--data_root", type=str, default="./data",
        help="Root directory for the EuroSAT dataset.",
    )
    parser.add_argument(
        "--use_wandb", action="store_true",
        help="Enable Weights & Biases logging (disabled by default).",
    )
    return parser.parse_args()


def _checkpoint_exists(save_dir):
    """Return True if *save_dir* already contains at least one .pt / .pth file."""
    if not os.path.isdir(save_dir):
        return False
    for fname in os.listdir(save_dir):
        if fname.endswith(".pt") or fname.endswith(".pth"):
            return True
    return False


def _format_duration(seconds):
    """Return a human-readable duration string."""
    if seconds < 60:
        return f"{seconds:.1f}s"
    minutes = seconds / 60.0
    if minutes < 60:
        return f"{minutes:.1f}m"
    hours = minutes / 60.0
    return f"{hours:.1f}h"


def _print_header(experiment, idx, total):
    """Print a prominent header before each experiment."""
    width = 72
    print()
    print("=" * width)
    print(
        f"  EXPERIMENT {experiment['number']}/{total}  |  "
        f"{experiment['name']}  |  {experiment['description']}"
    )
    print("=" * width)
    print()


def _print_summary_table(results):
    """Print a formatted summary table of all experiment outcomes."""
    width = 88
    print()
    print("=" * width)
    print("  EXPERIMENT SUMMARY")
    print("=" * width)

    header = (
        f"  {'#':<4} {'Name':<22} {'Description':<26} {'Status':<12} {'Duration':<10}"
    )
    print(header)
    print("  " + "-" * (width - 4))

    for row in results:
        status = row["status"]
        duration = row["duration"]
        duration_str = _format_duration(duration) if duration is not None else "--"

        print(
            f"  {row['number']:<4} {row['name']:<22} {row['description']:<26} "
            f"{status:<12} {duration_str:<10}"
        )

    print("=" * width)

    # Counts
    completed = sum(1 for r in results if r["status"] == "COMPLETED")
    failed = sum(1 for r in results if r["status"] == "FAILED")
    skipped = sum(1 for r in results if r["status"] == "SKIPPED")
    not_run = sum(1 for r in results if r["status"] == "NOT RUN")

    parts = []
    if completed:
        parts.append(f"{completed} completed")
    if failed:
        parts.append(f"{failed} failed")
    if skipped:
        parts.append(f"{skipped} skipped (resumed)")
    if not_run:
        parts.append(f"{not_run} not selected")
    print("  " + ", ".join(parts))
    print("=" * width)
    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    # Resolve path to train.py relative to this script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    train_script = os.path.join(script_dir, "train.py")

    if not os.path.isfile(train_script):
        print(f"ERROR: train.py not found at {train_script}", file=sys.stderr)
        sys.exit(1)

    python = sys.executable

    # Determine which experiments to run
    if args.run is not None:
        valid_numbers = {e["number"] for e in EXPERIMENTS}
        for num in args.run:
            if num not in valid_numbers:
                print(
                    f"ERROR: experiment number {num} is not valid. "
                    f"Choose from {sorted(valid_numbers)}.",
                    file=sys.stderr,
                )
                sys.exit(1)
        selected_numbers = set(args.run)
    else:
        selected_numbers = {e["number"] for e in EXPERIMENTS}

    total_selected = len(selected_numbers)

    # Banner
    print()
    print("#" * 72)
    print("#  EfficientEuroSAT  --  Experiment Orchestrator")
    print("#" * 72)
    print(f"  Python:       {python}")
    print(f"  train.py:     {train_script}")
    print(f"  Epochs:       {args.epochs}")
    print(f"  Batch size:   {args.batch_size}")
    print(f"  Data root:    {args.data_root}")
    print(f"  W&B logging:  {'enabled' if args.use_wandb else 'disabled'}")
    print(f"  Resume mode:  {'on' if args.resume else 'off'}")
    print(f"  Experiments:  {total_selected} selected")
    print()

    # Run experiments
    results = []
    run_idx = 0
    overall_start = time.time()

    for experiment in EXPERIMENTS:
        row = {
            "number": experiment["number"],
            "name": experiment["name"],
            "description": experiment["description"],
            "status": "NOT RUN",
            "duration": None,
        }

        if experiment["number"] not in selected_numbers:
            results.append(row)
            continue

        run_idx += 1
        save_dir = os.path.join(".", "checkpoints", experiment["name"])

        # Resume check
        if args.resume and _checkpoint_exists(save_dir):
            print(
                f"[{run_idx}/{total_selected}] Skipping '{experiment['name']}' "
                f"-- checkpoint already exists in {save_dir}"
            )
            row["status"] = "SKIPPED"
            results.append(row)
            continue

        _print_header(experiment, run_idx, total_selected)

        # Build command
        cmd = [
            python, train_script,
            "--epochs", str(args.epochs),
            "--batch_size", str(args.batch_size),
            "--data_root", args.data_root,
            "--save_dir", save_dir,
            "--experiment_name", experiment["name"],
        ]

        if not args.use_wandb:
            cmd.append("--no_wandb")

        cmd.extend(experiment["flags"])

        print("  Command:")
        print(f"    {' '.join(cmd)}")
        print()

        exp_start = time.time()
        result = subprocess.run(cmd, check=False)
        exp_duration = time.time() - exp_start

        row["duration"] = exp_duration

        if result.returncode == 0:
            row["status"] = "COMPLETED"
            print(
                f"\n  Experiment '{experiment['name']}' completed in "
                f"{_format_duration(exp_duration)}."
            )
        else:
            row["status"] = "FAILED"
            print(
                f"\n  Experiment '{experiment['name']}' FAILED "
                f"(exit code {result.returncode}) after "
                f"{_format_duration(exp_duration)}.",
                file=sys.stderr,
            )

        results.append(row)

    overall_duration = time.time() - overall_start

    # Summary
    _print_summary_table(results)
    print(f"  Total wall-clock time: {_format_duration(overall_duration)}")
    print()


if __name__ == "__main__":
    main()
