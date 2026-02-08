#!/usr/bin/env python3
"""UCAT (Uncertainty-Calibrated Attention Temperature) analysis script.

Performs two key analyses on a trained EfficientEuroSAT checkpoint:

  A2 -- Temperature-Entropy Correlation
        Scatter plot of learned attention temperature (tau) versus
        prediction entropy, with a regression line and Pearson r.

  A3 -- Correct vs Wrong
        Compare tau distributions for correct versus incorrect
        predictions using summary statistics and a box plot.

Usage:
    python scripts/analyze_ucat.py --checkpoint ./checkpoints/best_model.pth
    python scripts/analyze_ucat.py --checkpoint ./checkpoints/best_model.pth --save_dir ./analysis_results/ucat
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import argparse
import json

import numpy as np
import torch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.data.eurosat import get_eurosat_dataloaders, EUROSAT_CLASS_NAMES
from src.models.efficient_vit import EfficientEuroSATViT, create_efficient_eurosat_tiny
from src.models.baseline import BaselineViT
from src.utils.helpers import set_seed, get_device
from src.evaluation.ucat_analysis import (
    compute_temperature_entropy_correlation,
    analyze_correct_vs_incorrect,
    plot_temperature_entropy,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="UCAT analysis: temperature-entropy correlation and correct-vs-incorrect comparison",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--checkpoint", type=str, required=True,
        help="Path to model checkpoint",
    )
    parser.add_argument(
        "--data_root", type=str, default="./data",
        help="Root directory for dataset",
    )
    parser.add_argument(
        "--save_dir", type=str, default="./analysis_results/ucat",
        help="Directory to save analysis results and plots",
    )
    parser.add_argument(
        "--device", type=str, default=None,
        help="Device to use (auto-detected if not specified)",
    )
    parser.add_argument(
        "--batch_size", type=int, default=64,
        help="Evaluation batch size",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed",
    )
    return parser.parse_args()


def load_model_from_checkpoint(checkpoint_path, device):
    """Load a model from a saved checkpoint."""
    print(f"Loading checkpoint from: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Extract model config from checkpoint
    model_config = checkpoint.get("model_config", {})
    model_type = model_config.get("model_type", "efficient_eurosat")

    if model_type == "efficient_eurosat":
        model = EfficientEuroSATViT(
            img_size=model_config.get("img_size", 224),
            num_classes=model_config.get("num_classes", 10),
            use_learned_temp=model_config.get("use_learned_temp", True),
            use_early_exit=model_config.get("use_early_exit", True),
            use_learned_dropout=model_config.get("use_learned_dropout", True),
            use_learned_residual=model_config.get("use_learned_residual", True),
            use_temp_annealing=model_config.get("use_temp_annealing", True),
            use_decomposition=model_config.get("use_decomposition", False),
            tau_min=model_config.get("tau_min", 0.1),
            dropout_max=model_config.get("dropout_max", 0.3),
            exit_threshold=model_config.get("exit_threshold", 0.9),
            exit_min_layer=model_config.get("exit_min_layer", 4),
        )
    elif model_type == "baseline":
        model = BaselineViT(
            img_size=model_config.get("img_size", 224),
            num_classes=model_config.get("num_classes", 10),
        )
    else:
        raise ValueError(f"Unknown model type in checkpoint: {model_type}")

    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()
    if hasattr(model, 'early_exit_enabled'):
        model.early_exit_enabled = False

    print(f"  Model type: {model_type}")
    print(f"  Epoch: {checkpoint.get('epoch', 'unknown')}")
    print(f"  Best val accuracy: {checkpoint.get('best_val_accuracy', 'unknown')}")

    return model, model_type, model_config


def plot_correct_vs_incorrect_boxplot(results, save_path):
    """Create a box plot comparing tau distributions for correct vs incorrect predictions.

    Parameters
    ----------
    results : dict
        Output of ``compute_temperature_entropy_correlation`` containing
        ``'temperatures'`` and ``'correct'`` arrays.
    save_path : str
        File path for the saved figure.
    """
    temps = results["temperatures"]
    correct = results["correct"]

    mask_correct = correct == 1.0
    mask_incorrect = correct == 0.0

    temps_correct = temps[mask_correct]
    temps_incorrect = temps[mask_incorrect]

    fig, ax = plt.subplots(figsize=(8, 6))

    box_data = []
    box_labels = []

    if len(temps_correct) > 0:
        box_data.append(temps_correct)
        box_labels.append(f"Correct\n(n={len(temps_correct)})")
    if len(temps_incorrect) > 0:
        box_data.append(temps_incorrect)
        box_labels.append(f"Incorrect\n(n={len(temps_incorrect)})")

    if len(box_data) == 0:
        print("  Warning: no data for box plot, skipping.")
        plt.close(fig)
        return

    bp = ax.boxplot(box_data, labels=box_labels, patch_artist=True, widths=0.5)

    colors = ["#4CAF50", "#F44336"]  # green for correct, red for incorrect
    for patch, color in zip(bp["boxes"], colors[: len(box_data)]):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)

    ax.set_ylabel("Attention Temperature (\u03c4)", fontsize=12)
    ax.set_title("A3: Temperature Distribution -- Correct vs Incorrect", fontsize=14)
    ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"  Saved correct-vs-incorrect box plot to: {save_path}")


def main():
    args = parse_args()
    set_seed(args.seed)

    # Setup device
    if args.device is not None:
        device = torch.device(args.device)
    else:
        device = get_device()

    print("=" * 70)
    print("UCAT Analysis")
    print("=" * 70)
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Device:     {device}")

    # Create save directory
    os.makedirs(args.save_dir, exist_ok=True)

    # Load model
    model, model_type, model_config = load_model_from_checkpoint(
        args.checkpoint, device
    )

    # Load EuroSAT test data
    print("\nLoading EuroSAT test data...")
    _, _, test_loader, _ = get_eurosat_dataloaders(
        root=args.data_root,
        img_size=model_config.get("img_size", 224),
        batch_size=args.batch_size,
    )
    print(f"  Test batches: {len(test_loader)}")

    # ------------------------------------------------------------------
    # A2: Temperature-Entropy Correlation
    # ------------------------------------------------------------------
    print("\n--- A2: Temperature-Entropy Correlation ---")
    correlation_results = compute_temperature_entropy_correlation(
        model, test_loader, device
    )

    corr = correlation_results["correlation"]
    temps = correlation_results["temperatures"]
    entropies = correlation_results["entropies"]

    print(f"  Pearson correlation (tau vs entropy): {corr:.4f}")
    print(f"  Temperature -- mean: {temps.mean():.4f}, std: {temps.std():.4f}")
    print(f"  Entropy     -- mean: {entropies.mean():.4f}, std: {entropies.std():.4f}")
    print(f"  Num samples: {len(temps)}")

    # Scatter plot
    scatter_path = os.path.join(args.save_dir, "a2_temperature_entropy_scatter.png")
    plot_temperature_entropy(correlation_results, save_path=scatter_path)

    # ------------------------------------------------------------------
    # A3: Correct vs Wrong
    # ------------------------------------------------------------------
    print("\n--- A3: Correct vs Incorrect Temperature Comparison ---")
    cvi_results = analyze_correct_vs_incorrect(correlation_results)

    temp_correct = cvi_results["temp_correct"]
    temp_incorrect = cvi_results["temp_incorrect"]
    ratio = cvi_results["ratio"]

    correct_arr = correlation_results["correct"]
    n_correct = int(correct_arr.sum())
    n_incorrect = int(len(correct_arr) - n_correct)

    print(f"  Correct predictions:   n={n_correct}, mean tau={temp_correct:.4f}")
    print(f"  Incorrect predictions: n={n_incorrect}, mean tau={temp_incorrect:.4f}")
    print(f"  Ratio (incorrect / correct): {ratio:.4f}")

    # Box plot
    boxplot_path = os.path.join(args.save_dir, "a3_correct_vs_incorrect_boxplot.png")
    plot_correct_vs_incorrect_boxplot(correlation_results, save_path=boxplot_path)

    # ------------------------------------------------------------------
    # Save results JSON
    # ------------------------------------------------------------------
    summary = {
        "checkpoint": args.checkpoint,
        "model_type": model_type,
        "num_samples": len(temps),
        "a2_temperature_entropy_correlation": {
            "pearson_r": float(corr),
            "temperature_mean": float(temps.mean()),
            "temperature_std": float(temps.std()),
            "entropy_mean": float(entropies.mean()),
            "entropy_std": float(entropies.std()),
        },
        "a3_correct_vs_incorrect": {
            "num_correct": n_correct,
            "num_incorrect": n_incorrect,
            "mean_tau_correct": float(temp_correct),
            "mean_tau_incorrect": float(temp_incorrect),
            "ratio_incorrect_over_correct": float(ratio),
        },
    }

    results_path = os.path.join(args.save_dir, "ucat_analysis_results.json")
    with open(results_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n  Results JSON saved to: {results_path}")

    # ------------------------------------------------------------------
    # Print summary
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("UCAT ANALYSIS SUMMARY")
    print("=" * 70)
    print(f"  Model:                      {model_type}")
    print(f"  Samples evaluated:          {len(temps)}")
    print(f"  [A2] Pearson r (tau, H):    {corr:.4f}")
    print(f"  [A3] Mean tau (correct):    {temp_correct:.4f}")
    print(f"  [A3] Mean tau (incorrect):  {temp_incorrect:.4f}")
    print(f"  [A3] Ratio (incorr/corr):   {ratio:.4f}")
    print(f"  Scatter plot:               {scatter_path}")
    print(f"  Box plot:                   {boxplot_path}")
    print(f"  Results JSON:               {results_path}")
    print("=" * 70)


if __name__ == "__main__":
    main()
